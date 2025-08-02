// rune_cuda_infer.cu (优化版)
#include "rune_infer.hpp"
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <algorithm>

#define CUDA_CHECK(call) do {                                                \
    cudaError_t err = call;                                                  \
    if (err != cudaSuccess) {                                                \
        fprintf(stderr, "[CUDA ERROR] %s:%d: %s\n",                          \
                __FILE__, __LINE__, cudaGetErrorString(err));                \
        exit(EXIT_FAILURE);                                                  \
    }                                                                         \
} while (0)

static constexpr int INPUT_W        = 480;
static constexpr int INPUT_H        = 480;
static constexpr int NUM_CLASSES    = 2;
static constexpr int NUM_COLORS     = 2;
static constexpr int NUM_POINTS     = 5;
static constexpr float MERGE_CONF_ERROR = 0.15f;
static constexpr float MERGE_MIN_IOU     = 0.9f;

namespace rune_cuda_infer {

//--------------------------------------
// 在 GPU 上初始化 grid & stride 数组
//--------------------------------------
GPUGridAndStride* init_grid_strides_on_gpu(
    int in_w, int in_h,
    const std::vector<int>& strides,
    size_t& device_grid_count)
{
    std::vector<GPUGridAndStride> host;
    for (int s : strides) {
        int gw = in_w / s, gh = in_h / s;
        for (int y = 0; y < gh; ++y)
            for (int x = 0; x < gw; ++x)
                host.push_back({x,y,s});
    }
    device_grid_count = host.size();
    if (!device_grid_count) return nullptr;
    GPUGridAndStride* dptr = nullptr;
    CUDA_CHECK(cudaMalloc(&dptr, device_grid_count*sizeof(GPUGridAndStride)));
    CUDA_CHECK(cudaMemcpy(dptr, host.data(),
                          device_grid_count*sizeof(GPUGridAndStride),
                          cudaMemcpyHostToDevice));
    return dptr;
}

//---------------------------
// 双线性插值
//---------------------------
__device__ float bilinear_interpolate(
    const unsigned char* img, int w, int h,
    float x, float y, int c)
{
    x = fminf(fmaxf(x,0.f), w-1.f);
    y = fminf(fmaxf(y,0.f), h-1.f);
    int x0=floorf(x), x1=min(x0+1,w-1);
    int y0=floorf(y), y1=min(y0+1,h-1);
    float dx=x-x0, dy=y-y0;
    float v00=img[(y0*w+x0)*3+c],
          v01=img[(y0*w+x1)*3+c],
          v10=img[(y1*w+x0)*3+c],
          v11=img[(y1*w+x1)*3+c];
    return (1-dx)*(1-dy)*v00 + dx*(1-dy)*v01 + (1-dx)*dy*v10 + dx*dy*v11;
}

//-------------------------------------------
// letterbox 预处理 kernel
//-------------------------------------------
__global__ void letterbox_kernel(
    const unsigned char* in, int iw, int ih,
    float* out, int ow, int oh,
    float scale, int pt, int pl)
{
    int x=blockIdx.x*blockDim.x+threadIdx.x;
    int y=blockIdx.y*blockDim.y+threadIdx.y;
    if (x>=ow||y>=oh) return;
    float ix=(x-pl)/scale, iy=(y-pt)/scale;
    bool pad=(ix<0||iy<0||ix>=iw||iy>=ih);
    for(int c=0;c<3;++c){
        int oc=2-c;
        float v=pad?114.f:bilinear_interpolate(in,iw,ih,ix,iy,c);
        out[oc*oh*ow + y*ow + x] = v;
    }
}

//---------------------------
// argmax
//---------------------------
__device__ int argmax(const float* ptr, int len) {
    float m=ptr[0]; int idx=0;
    for(int i=1;i<len;++i) if(ptr[i]>m){m=ptr[i]; idx=i;}
    return idx;
}

//--------------------------------------------------
// decode 核函数：5 点 + conf + color/type
//--------------------------------------------------
__global__ void decode_rune_kernel(
    const float* output,
    const GPUGridAndStride* gs,
    int total, size_t grid_count,
    const float* t,
    GPURuneObject* objs,
    float conf_th)
{
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i>=total||i>=(int)grid_count) return;

    const float* det = output + i*(NUM_POINTS*2 + 1 + NUM_COLORS + NUM_CLASSES);
    float conf = det[NUM_POINTS*2];
    if(conf<conf_th) return;

    int gx=gs[i].grid0, gy=gs[i].grid1, s=gs[i].stride;
    __shared__ float tfm[9];
    if(threadIdx.x<9) tfm[threadIdx.x]=t[threadIdx.x];
    __syncthreads();

    GPURuneObject obj{};
    for(int k=0;k<NUM_POINTS;++k){
        float px=(det[2*k]   + gx)*s;
        float py=(det[2*k+1] + gy)*s;
        float x_=tfm[0]*px + tfm[1]*py + tfm[2];
        float y_=tfm[3]*px + tfm[4]*py + tfm[5];
        float w_=tfm[6]*px + tfm[7]*py + tfm[8];
        obj.x[k] = x_/w_;
        obj.y[k] = y_/w_;
    }

    obj.confidence = conf;
    obj.color_id   = argmax(det + NUM_POINTS*2 + 1, NUM_COLORS);
    obj.type_id    = argmax(det + NUM_POINTS*2 + 1 + NUM_COLORS, NUM_CLASSES);
    obj.valid      = 1;
    obj.num_pts    = NUM_POINTS;
    objs[i] = obj;
}

//-------------------------------------------
// 清除非 Top-k
//-------------------------------------------
__global__ void clear_invalid_topk(
    GPURuneObject* objs, int N, int k)
{
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i< N && i>=k) objs[i].valid=0;
}

//-------------------------------------------
// NMS 核函数
//-------------------------------------------
__global__ void nms_kernel(
    GPURuneObject* objs, int k, float th)
{
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i>=k||!objs[i].valid) return;
    GPURuneObject a=objs[i];
    for(int j=0;j<i;++j){
        if(j>=k) break;
        GPURuneObject &b=objs[j];
        if(!b.valid) continue;
        if(a.color_id==b.color_id && a.type_id==b.type_id){
            // 计算两多边形点集的包围框 IoU
            float ax0=a.x[0], ay0=a.y[0], ax1=ax0, ay1=ay0;
            float bx0=b.x[0], by0=b.y[0], bx1=bx0, by1=by0;
            for(int u=1;u<NUM_POINTS;++u){
                ax0=min(ax0,a.x[u]); ay0=min(ay0,a.y[u]);
                ax1=max(ax1,a.x[u]); ay1=max(ay1,a.y[u]);
                bx0=min(bx0,b.x[u]); by0=min(by0,b.y[u]);
                bx1=max(bx1,b.x[u]); by1=max(by1,b.y[u]);
            }
            float x1=max(ax0,bx0), y1=max(ay0,by0);
            float x2=min(ax1,bx1), y2=min(ay1,by1);
            float inter=max(0.f,x2-x1)*max(0.f,y2-y1);
            float areaA=(ax1-ax0)*(ay1-ay0),
                  areaB=(bx1-bx0)*(by1-by0);
            float u=areaA+areaB-inter+1e-6f;
            float iou=inter/u;
            if(iou>th){ objs[i].valid=0; break; }
            if(iou>MERGE_MIN_IOU
               && fabsf(a.confidence-b.confidence)<MERGE_CONF_ERROR)
            {
                if(b.num_pts+NUM_POINTS <= NUM_POINTS*2){
                    for(int u=0;u<NUM_POINTS;++u){
                        b.x[b.num_pts+u]=a.x[u];
                        b.y[b.num_pts+u]=a.y[u];
                    }
                    b.num_pts += NUM_POINTS;
                }
            }
        }
    }
}

// 主机排序比较器
struct HostCmpRune {
    bool operator()(const GPURuneObject&a,
                    const GPURuneObject&b) const {
        return a.confidence > b.confidence;
    }
};


CudaInfer::CudaInfer() = default;
CudaInfer::~CudaInfer() { release(); }

void CudaInfer::init(GPUGridAndStride* grid_strides, size_t img_bytes, int max_N,size_t grid_count) {
    d_grid_strides_ = grid_strides;
    buf_image_bytes_ = img_bytes;
    buf_max_N_ = max_N;
    CUDA_CHECK(cudaMalloc(&d_input_bgr_, buf_image_bytes_));
    CUDA_CHECK(cudaMalloc(&d_nchw_, INPUT_W * INPUT_H * 3 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_objs_, buf_max_N_ * sizeof(GPURuneObject)));
    CUDA_CHECK(cudaMalloc(&d_tf_, 9 * sizeof(float)));
}

void CudaInfer::release() {
    if (d_grid_strides_) cudaFree(d_grid_strides_), d_grid_strides_ = nullptr;
    if (d_input_bgr_) cudaFree(d_input_bgr_), d_input_bgr_ = nullptr;
    if (d_nchw_) cudaFree(d_nchw_), d_nchw_ = nullptr;
    if (d_objs_) cudaFree(d_objs_), d_objs_ = nullptr;
    if (d_tf_) cudaFree(d_tf_), d_tf_ = nullptr;
}

float* CudaInfer::preprocess(const unsigned char* input_bgr_host,
    int img_w, int img_h, Eigen::Matrix3f& tf_matrix,
    cudaStream_t stream)
{
    float scale = fminf(INPUT_W / (float)img_w, INPUT_H / (float)img_h);
    int rw = round(img_w * scale), rh = round(img_h * scale);
    int pad_l = (INPUT_W - rw) / 2, pad_t = (INPUT_H - rh) / 2;

    tf_matrix << 1.f / scale, 0, -pad_l / scale,
                 0, 1.f / scale, -pad_t / scale,
                 0, 0, 1;

    size_t img_size = img_w * img_h * 3;
    CUDA_CHECK(cudaMemcpyAsync(d_input_bgr_, input_bgr_host, img_size,
        cudaMemcpyHostToDevice, stream));

    dim3 threads(32, 32);
    dim3 blocks((INPUT_W + 31) / 32, (INPUT_H + 31) / 32);
    letterbox_kernel<<<blocks, threads, 0, stream>>>(d_input_bgr_, img_w, img_h,
        d_nchw_, INPUT_W, INPUT_H, scale, pad_t, pad_l);
    CUDA_CHECK(cudaGetLastError());
    return d_nchw_;
}
std::vector<GPURuneObject> CudaInfer::postprocess(
    const float* output,
    int N,
    const Eigen::Matrix3f& tfm,
    float conf_th,
    float nms_th,
    int top_k)
{
    if (!output || !d_grid_strides_ || !d_tf_ || !d_objs_) {
        fprintf(stderr, "[ERROR] postprocess null ptr\n");
        return {};
    }

    // 限制 N
    int validN = std::min<int>({N, (int)grid_count_, buf_max_N_});
    if (validN <= 0) return {};

    // 上传变换矩阵 (RowMajor)
    float h_tf[9];
    Eigen::Map<Eigen::Matrix<float,3,3,Eigen::RowMajor>>(h_tf,3,3) = tfm;
    CUDA_CHECK(cudaMemcpy(d_tf_, h_tf, sizeof(h_tf), cudaMemcpyHostToDevice));

    // 启动 kernels
    int threads = 256;
    int blocks  = (validN + threads - 1) / threads;

    decode_rune_kernel<<<blocks, threads>>>(
        output, d_grid_strides_, validN, grid_count_, d_tf_, d_objs_, conf_th);
    clear_invalid_topk<<<blocks, threads>>>(d_objs_, validN, top_k);
    nms_kernel<<<blocks, threads>>>(d_objs_, top_k, nms_th);
    CUDA_CHECK(cudaGetLastError());

    // 拷回到主机
    std::vector<GPURuneObject> host(validN);
    CUDA_CHECK(cudaMemcpy(host.data(),
                          d_objs_,
                          validN * sizeof(GPURuneObject),
                          cudaMemcpyDeviceToHost));

    // 主机排序 & 裁剪
    std::sort(host.begin(), host.end(), HostCmpRune());
    if ((int)host.size() > top_k)
        host.resize(top_k);

    return host;
}
std::vector<GPURuneObject> CudaInfer::process_trt(
        nvinfer1::IExecutionContext* context,
        void* device_buffers[2],
        int input_idx_,
        int output_idx_,
        const unsigned char* input_bgr_host,
        int img_w,
        int img_h,
        Eigen::Matrix3f& tf_matrix,
        cudaStream_t stream,
        int N,
        float conf_th,
        float nms_th,
        int top_k
    )
{

    void* input_tensor_ptr = preprocess(
        input_bgr_host,
        img_w,
        img_h,
        tf_matrix,
        stream
    );

    context->setTensorAddress("images", input_tensor_ptr);
    context->setTensorAddress("output", device_buffers[output_idx_]);


    if (!context->enqueueV3(stream)) {
        std::cerr<<"enqueueV3 failed!";
        return {};
    }

    std::vector<GPURuneObject> results = postprocess(
        (float*)device_buffers[output_idx_],
        N,
        tf_matrix,
        conf_th,
        nms_th,
        top_k
    );
    return results;
}


}