// rune_infer.cu
// Optimized for robustness, RAII resource management, and proper error handling.

#include "rune_infer.hpp"
#include <thrust/device_ptr.h>
#include <thrust/sort.h>
#include <cstdio>
#include <cmath>
#include <stdexcept>
#include <vector>
static constexpr int INPUT_W = 480; // Width of input
static constexpr int INPUT_H = 480; // Height of input
static constexpr int NUM_CLASSES = 2; // Number of classes
static constexpr int NUM_COLORS = 2; // Number of color
static constexpr int NUM_POINTS = 5;
static constexpr int NUM_POINTS_2 = 2 * NUM_POINTS;
static constexpr float MERGE_CONF_ERROR = 0.15;
static constexpr float MERGE_MIN_IOU = 0.9;
static constexpr int MAX_MERGE_POINTS =10;
// Exception for CUDA errors
struct CudaException : public std::runtime_error {
    CudaException(const char* func, cudaError_t err)
      : std::runtime_error(std::string(func) + " failed: " + cudaGetErrorString(err)) {}
};

inline void checkCuda(cudaError_t err, const char* func) {
    if (err != cudaSuccess) throw CudaException(func, err);
}

// RAII wrapper for device buffers
template<typename T>
class DeviceBuffer {
public:
    DeviceBuffer() : ptr_(nullptr), size_(0) {}
    ~DeviceBuffer() { free(); }

    void allocate(size_t count) {
        free();
        size_ = count;
        checkCuda(cudaMalloc(&ptr_, size_ * sizeof(T)), "cudaMalloc");
    }
    void free() {
        if (ptr_) cudaFree(ptr_);
        ptr_ = nullptr; size_ = 0;
    }
    T* ptr() const { return ptr_; }
    size_t size() const { return size_; }
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;
    DeviceBuffer(DeviceBuffer&& o) noexcept : ptr_(o.ptr_), size_(o.size_) { o.ptr_ = nullptr; o.size_ = 0; }
    DeviceBuffer& operator=(DeviceBuffer&& o) noexcept { free(); ptr_=o.ptr_; size_=o.size_; o.ptr_=nullptr; o.size_=0; return *this; }

private:
    T*    ptr_;
    size_t size_;
};

namespace rune_cuda_infer {

// Initialize grid+stride on GPU
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
                host.push_back({x, y, s});
    }
    device_grid_count = host.size();
    if (!device_grid_count) return nullptr;

    GPUGridAndStride* d_ptr = nullptr;
    checkCuda(cudaMalloc(&d_ptr, device_grid_count * sizeof(GPUGridAndStride)), "cudaMalloc(grid)");
    checkCuda(cudaMemcpy(d_ptr, host.data(), device_grid_count * sizeof(GPUGridAndStride), cudaMemcpyHostToDevice), "cudaMemcpy(grid)");
    return d_ptr;
}

// Bilinear interpolation
__device__ float bilinear_interpolate(const unsigned char* img, int w, int h, float x, float y, int c) {
    x = fminf(fmaxf(x,0.f), w-1.f);
    y = fminf(fmaxf(y,0.f), h-1.f);
    int x0=floorf(x), x1=min(x0+1,w-1), y0=floorf(y), y1=min(y0+1,h-1);
    float dx=x-x0, dy=y-y0;
    float v00 = img[(y0*w+x0)*3+c];
    float v01 = img[(y0*w+x1)*3+c];
    float v10 = img[(y1*w+x0)*3+c];
    float v11 = img[(y1*w+x1)*3+c];
    return (1-dx)*(1-dy)*v00 + dx*(1-dy)*v01 + (1-dx)*dy*v10 + dx*dy*v11;
}

// Letterbox kernel
__global__ void letterbox_kernel(const unsigned char* input, int iw,int ih, float* output, int ow,int oh, float scale,int pt,int pl) {
    int x = blockIdx.x*blockDim.x+threadIdx.x;
    int y = blockIdx.y*blockDim.y+threadIdx.y;
    if(x>=ow||y>=oh) return;
    float in_x=(x-pl)/scale, in_y=(y-pt)/scale;
    bool pad=(in_x<0||in_y<0||in_x>=iw||in_y>=ih);
    for(int c=0;c<3;++c) {
        int oc=2-c;
        float v=pad?114.f:bilinear_interpolate(input,iw,ih,in_x,in_y,c);
        output[oc*oh*ow + y*ow + x]=v;
    }
}

// Decode kernel with shared TF matrix
__global__ void decode_rune_kernel(const float* out, const GPUGridAndStride* gs, int N, const float* tf, GPURuneObject* objs, float conf_th){
    extern __shared__ float tfm[];
    if(threadIdx.x==0) for(int i=0;i<9;++i) tfm[i]=tf[i];
    __syncthreads();
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i>=N) return;
    const float* det=out + i* (NUM_POINTS_2 +1+NUM_COLORS+NUM_CLASSES);
    float conf=det[NUM_POINTS_2];
    if(conf<conf_th){ objs[i].valid=0; return; }
    auto g=gs[i];
    float nx[NUM_POINTS], ny[NUM_POINTS];
    for(int j=0;j<NUM_POINTS;++j){ nx[j]=(det[2*j]+g.grid0)*g.stride; ny[j]=(det[2*j+1]+g.grid1)*g.stride; }
    for(int j=0;j<NUM_POINTS;++j){ float x=nx[j],y=ny[j], w=tfm[6]*x+tfm[7]*y+tfm[8]; if(fabsf(w)<1e-6f) w=1e-6f;
        objs[i].x[j]=(tfm[0]*x+tfm[1]*y+tfm[2])/w;
        objs[i].y[j]=(tfm[3]*x+tfm[4]*y+tfm[5])/w;
    }
    // color
    const float* cs=det+NUM_POINTS_2+1;
    int cid=0; for(int j=1;j<NUM_COLORS;++j) if(cs[j]>cs[cid]) cid=j;
    // type
    const float* ts=cs+NUM_COLORS;
    int tid=0; for(int j=1;j<NUM_CLASSES;++j) if(ts[j]>ts[tid]) tid=j;
    auto& o=objs[i];
    for(int j=0;j<NUM_POINTS;++j){ o.x[j]=objs[i].x[j]; o.y[j]=objs[i].y[j]; }
    o.confidence=conf; o.color_id=cid; o.type_id=tid; o.valid=1; o.num_pts=NUM_POINTS;
}

// Clear beyond top-k
__global__ void clear_invalid_topk(GPURuneObject* objs,int N,int k){int i=blockIdx.x*blockDim.x+threadIdx.x; if(i<N&&i>=k) objs[i].valid=0;}

// NMS kernel
__global__ void nms_kernel(GPURuneObject* objs, int N, float th) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N || !objs[i].valid) return;

    auto& a = objs[i];
    float axmin = a.x[0], axmax = a.x[0], aymin = a.y[0], aymax = a.y[0];
    for (int j = 1; j < a.num_pts; ++j) {
        axmin = fminf(axmin, a.x[j]);
        axmax = fmaxf(axmax, a.x[j]);
        aymin = fminf(aymin, a.y[j]);
        aymax = fmaxf(aymax, a.y[j]);
    }

    for (int j = 0; j < i; ++j) {
        auto& b = objs[j];
        if (!b.valid || a.color_id != b.color_id || a.type_id != b.type_id)
            continue;

        float bxmin = b.x[0], bxmax = b.x[0], bymin = b.y[0], bymax = b.y[0];
        for (int k = 1; k < b.num_pts; ++k) {
            bxmin = fminf(bxmin, b.x[k]);
            bxmax = fmaxf(bxmax, b.x[k]);
            bymin = fminf(bymin, b.y[k]);
            bymax = fmaxf(bymax, b.y[k]);
        }

        float x1 = fmaxf(axmin, bxmin), y1 = fmaxf(aymin, bymin);
        float x2 = fminf(axmax, bxmax), y2 = fminf(aymax, bymax);
        float inter = fmaxf(0.f, x2 - x1) * fmaxf(0.f, y2 - y1);
        float areaA = (axmax - axmin) * (aymax - aymin);
        float areaB = (bxmax - bxmin) * (bymax - bymin);
        float iou = inter / (areaA + areaB - inter + 1e-6f);

        if (iou > th) {
            a.valid = 0;
            break;
        }

        if (iou > MERGE_MIN_IOU && fabsf(a.confidence - b.confidence) < MERGE_CONF_ERROR) {
            if (b.num_pts + a.num_pts <= MAX_MERGE_POINTS) {
                for (int k = 0; k < a.num_pts; ++k) {
                    b.x[b.num_pts + k] = a.x[k];
                    b.y[b.num_pts + k] = a.y[k];
                }
                b.num_pts += a.num_pts;
            }
        }
    }
}


// Pimpl implementation
class CudaInfer::Impl {
public:
    GPUGridAndStride* d_grid_ = nullptr;
    DeviceBuffer<unsigned char> buf_img_;
    DeviceBuffer<float> buf_nchw_;
    DeviceBuffer<GPURuneObject> buf_objs_;
    DeviceBuffer<float> buf_tf_;
    size_t img_bytes_ = 0;
    int max_N_ = 0;

    ~Impl() { release(); }

    void init(GPUGridAndStride* grid, size_t img_bytes, int max_N) {
        d_grid_ = grid; img_bytes_ = img_bytes; max_N_ = max_N;
        buf_nchw_.allocate(INPUT_W*INPUT_H*3);
        buf_objs_.allocate(max_N_);
        buf_tf_.allocate(9);
    }

    void release() {
        if(d_grid_) cudaFree(d_grid_);
        buf_img_.free(); buf_nchw_.free(); buf_objs_.free(); buf_tf_.free();
    }

    float* preprocess(const unsigned char* in, int iw,int ih, Eigen::Matrix3f& tfm, cudaStream_t s) {
        if(iw<=0||ih<=0) throw std::invalid_argument("Invalid image size");
        img_bytes_ = size_t(iw)*ih*3; buf_img_.allocate(img_bytes_);
        float scale=fminf(INPUT_W/(float)iw, INPUT_H/(float)ih);
        int rw=int(round(iw*scale)), rh=int(round(ih*scale));
        int pl=(INPUT_W-rw)/2, pt=(INPUT_H-rh)/2;
        tfm<<1/scale,0,-pl/scale,0,1/scale,-pt/scale,0,0,1;
        checkCuda(cudaMemcpyAsync(buf_img_.ptr(), in, img_bytes_, cudaMemcpyHostToDevice, s), "cudaMemcpyAsync(img)");
        dim3 t(32,32), b((INPUT_W+31)/32,(INPUT_H+31)/32);
        letterbox_kernel<<<b,t,0,s>>>(buf_img_.ptr(),iw,ih, buf_nchw_.ptr(), INPUT_W,INPUT_H, scale,pt,pl);
        checkCuda(cudaGetLastError(), "letterbox_kernel");
        return buf_nchw_.ptr();
    }

    std::vector<GPURuneObject> postprocess(const float* out, int N, const Eigen::Matrix3f& tfm, float conf_th, float nms_th, int top_k) {
        if(!out||!d_grid_) throw std::runtime_error("Null pointer in postprocess");
        if(N<=0||top_k<=0||top_k>max_N_||N>max_N_) throw std::invalid_argument("Invalid N/top_k");
        float tf_arr[9] = { tfm(0,0),tfm(0,1),tfm(0,2), tfm(1,0),tfm(1,1),tfm(1,2), tfm(2,0),tfm(2,1),tfm(2,2) };
        checkCuda(cudaMemcpy(buf_tf_.ptr(), tf_arr, sizeof(tf_arr), cudaMemcpyHostToDevice), "cudaMemcpy(tf)");
        int thr=256; int blk=(N+thr-1)/thr;
        checkCuda(cudaMemset(buf_objs_.ptr(),0, max_N_*sizeof(GPURuneObject)), "cudaMemset(objs)");
        decode_rune_kernel<<<blk,thr,9*sizeof(float)>>>(out, d_grid_, N, buf_tf_.ptr(), buf_objs_.ptr(), conf_th);
        checkCuda(cudaGetLastError(), "decode_rune_kernel");
        thrust::device_ptr<GPURuneObject> dp(buf_objs_.ptr());
        thrust::sort(thrust::device, dp, dp+N, ConfidenceComparator<GPURuneObject>());
        int sortN = std::min(N, max_N_);
        int blk2=(sortN+thr-1)/thr;
        clear_invalid_topk<<<blk2,thr>>>(buf_objs_.ptr(), sortN, top_k);
        checkCuda(cudaGetLastError(), "clear_invalid_topk");
        nms_kernel<<<blk2,thr>>>(buf_objs_.ptr(), top_k, nms_th);
        checkCuda(cudaGetLastError(), "nms_kernel");
        std::vector<GPURuneObject> outv(top_k);
        checkCuda(cudaMemcpy(outv.data(), buf_objs_.ptr(), top_k*sizeof(GPURuneObject), cudaMemcpyDeviceToHost), "cudaMemcpy(out)");
        outv.erase(std::remove_if(outv.begin(), outv.end(), [](auto& o){return o.valid==0;}), outv.end());
        return outv;
    }
};

CudaInfer::CudaInfer() : impl_(new Impl()) {}
CudaInfer::~CudaInfer() {};

void CudaInfer::init(GPUGridAndStride* g, size_t b, int n) { impl_->init(g,b,n); }
void CudaInfer::release() { impl_->release(); }
float* CudaInfer::preprocess(const unsigned char* i,int w,int h,Eigen::Matrix3f& m,cudaStream_t s) { return impl_->preprocess(i,w,h,m,s); }
std::vector<GPURuneObject> CudaInfer::postprocess(const float* o,int N,const Eigen::Matrix3f& m,float c,float n,int k){ return impl_->postprocess(o,N,m,c,n,k); }
std::vector<GPURuneObject> CudaInfer::process_trt(
    nvinfer1::IExecutionContext* ctx, void* bufs[2], int in_idx, int out_idx,
    const unsigned char* ib, int w,int h, Eigen::Matrix3f& m, cudaStream_t s,int N,float ct,float nt,int k)
{
    float* dev_in = preprocess(ib,w,h,m,s);
    cudaStreamSynchronize(s);
    ctx->setTensorAddress("images", dev_in);
    ctx->setTensorAddress("output", bufs[out_idx]);
    if(!ctx->enqueueV3(s)) throw std::runtime_error("enqueueV3 failed");
    return postprocess(reinterpret_cast<const float*>(bufs[out_idx]),N,m,ct,nt,k);
}

} // namespace rune_cuda_infer
