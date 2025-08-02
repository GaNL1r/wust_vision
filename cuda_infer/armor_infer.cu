// armor_cuda_infer.cu
#include "armor_infer.hpp"
#include <thrust/device_ptr.h>
#include <thrust/sort.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <stdexcept>

#define CUDA_CHECK(call) do {                                                      \
    cudaError_t err = call;                                                        \
    if (err != cudaSuccess) {                                                      \
        fprintf(stderr, "[CUDA ERROR] %s:%d: %s\n",                                \
                __FILE__, __LINE__, cudaGetErrorString(err));                      \
        exit(EXIT_FAILURE);                                                        \
    }                                                                               \
} while (0)

static const int INPUT_W        = 416;
static const int INPUT_H        = 416;
static constexpr int NUM_CLASSES = 8;
static constexpr int NUM_COLORS  = 4;
static constexpr float MERGE_CONF_ERROR = 0.15f;
static constexpr float MERGE_MIN_IOU     = 0.9f;

namespace armor_cuda_infer {

// ----------------------------------------------------------------------------
// 全局函数：在 GPU 上分配并初始化 grid & stride 数组
// ----------------------------------------------------------------------------
GPUGridAndStride* init_grid_strides_on_gpu(
    int in_w,
    int in_h,
    const std::vector<int>& strides,
    size_t& device_grid_count)
{
    std::vector<GPUGridAndStride> host_grid_strides;
    for (int s : strides) {
        int gw = in_w / s;
        int gh = in_h / s;
        for (int y = 0; y < gh; ++y) {
            for (int x = 0; x < gw; ++x) {
                host_grid_strides.push_back({x, y, s});
            }
        }
    }
    device_grid_count = host_grid_strides.size();
    if (device_grid_count == 0) {
        fprintf(stderr, "[WARN] init_grid_strides_on_gpu: no grids\n");
        return nullptr;
    }

    GPUGridAndStride* d_ptr = nullptr;
    CUDA_CHECK(cudaMalloc(&d_ptr, device_grid_count * sizeof(GPUGridAndStride)));
    CUDA_CHECK(cudaMemcpy(d_ptr,
                          host_grid_strides.data(),
                          device_grid_count * sizeof(GPUGridAndStride),
                          cudaMemcpyHostToDevice));
    return d_ptr;
}

// -------------------- Kernels & Helpers --------------------

__device__ float bilinear_interpolate(const unsigned char* img,
                                      int w, int h,
                                      float x, float y, int c) {
    x = fminf(fmaxf(x, 0.f), w - 1.f);
    y = fminf(fmaxf(y, 0.f), h - 1.f);
    int x0 = floorf(x), x1 = min(x0 + 1, w - 1);
    int y0 = floorf(y), y1 = min(y0 + 1, h - 1);
    float dx = x - x0, dy = y - y0;
    float v00 = img[(y0 * w + x0) * 3 + c];
    float v01 = img[(y0 * w + x1) * 3 + c];
    float v10 = img[(y1 * w + x0) * 3 + c];
    float v11 = img[(y1 * w + x1) * 3 + c];
    return (1 - dx)*(1 - dy)*v00 + dx*(1 - dy)*v01 + (1 - dx)*dy*v10 + dx*dy*v11;
}

__global__ void letterbox_kernel(const unsigned char* input_bgr,
                                 int img_w, int img_h,
                                 float* output_nchw,
                                 int out_w, int out_h,
                                 float scale,
                                 int pad_top, int pad_left) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= out_w || y >= out_h) return;
    float in_x = (x - pad_left) / scale;
    float in_y = (y - pad_top ) / scale;
    bool pad = (in_x < 0 || in_y < 0 || in_x >= img_w || in_y >= img_h);
    for (int c = 0; c < 3; ++c) {
        int oc = 2 - c;
        float v = pad ? 114.f :
                  bilinear_interpolate(input_bgr, img_w, img_h, in_x, in_y, c);
        output_nchw[oc * out_h * out_w + y * out_w + x] = v;
    }
}

__device__ int argmax(const float* ptr, int len) {
    float m = ptr[0]; int idx = 0;
    for (int i = 1; i < len; ++i) {
        if (ptr[i] > m) { m = ptr[i]; idx = i; }
    }
    return idx;
}

__global__ void decode_kernel(const float* output,
                              const GPUGridAndStride* grid_strides,
                              int total,
                              size_t grid_count,
                              const float* t,
                              GPUArmorObject* objs,
                              float conf_thresh) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= total || i >= (int)grid_count) return;

    const float* det = output + i * (9 + NUM_COLORS + NUM_CLASSES);
    GPUArmorObject obj{};  
    if (det[8] < conf_thresh) {
        objs[i] = obj;
        return;
    }

    int gx = grid_strides[i].grid0;
    int gy = grid_strides[i].grid1;
    int s  = grid_strides[i].stride;
    __shared__ float tfm[9];
    if (threadIdx.x < 9) tfm[threadIdx.x] = t[threadIdx.x];
    __syncthreads();

    for (int j = 0; j < 4; ++j) {
        float px = (det[2*j]     + gx) * s;
        float py = (det[2*j + 1] + gy) * s;
        float w  = tfm[6]*px + tfm[7]*py + tfm[8];
        if (fabsf(w) < 1e-6f) w = 1e-6f;
        obj.x[j] = (tfm[0]*px + tfm[1]*py + tfm[2]) / w;
        obj.y[j] = (tfm[3]*px + tfm[4]*py + tfm[5]) / w;
    }

    obj.confidence = det[8];
    obj.color_id   = argmax(det + 9, NUM_COLORS);
    obj.number_id  = argmax(det + 9 + NUM_COLORS, NUM_CLASSES);
    obj.valid      = 1;
    obj.num_pts    = 4;
    objs[i]        = obj;
}

__global__ void clear_invalid_topk(GPUArmorObject* objs, int total, int k) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < total && i >= k) {
        objs[i].valid = 0;
    }
}

__global__ void nms_kernel(GPUArmorObject* objs, int k, float thresh) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= k || !objs[i].valid) return;
    GPUArmorObject a = objs[i];
    for (int j = 0; j < i; ++j) {
        if (j >= k) break;
        GPUArmorObject& b = objs[j];
        if (!b.valid) continue;
        if (a.color_id == b.color_id && a.number_id == b.number_id) {
            float x1 = fmaxf(fminf(a.x[0],a.x[2]), fminf(b.x[0],b.x[2]));
            float y1 = fmaxf(fminf(a.y[0],a.y[2]), fminf(b.y[0],b.y[2]));
            float x2 = fminf(fmaxf(a.x[0],a.x[2]), fmaxf(b.x[0],b.x[2]));
            float y2 = fminf(fmaxf(a.y[0],a.y[2]), fmaxf(b.y[0],b.y[2]));
            float inter = fmaxf(0.f, x2 - x1) * fmaxf(0.f, y2 - y1);
            float area_a = fabsf((a.x[2]-a.x[0])*(a.y[2]-a.y[0]));
            float area_b = fabsf((b.x[2]-b.x[0])*(b.y[2]-b.y[0]));
            float u = area_a + area_b - inter + 1e-6f;
            float iou = inter / u;
            if (iou > thresh) {
                objs[i].valid = 0;
                return;
            }
            if (iou > MERGE_MIN_IOU && fabsf(a.confidence - b.confidence) < MERGE_CONF_ERROR) {
                if (b.num_pts + 4 <= 4) {
                    for (int m = 0; m < 4; ++m) {
                        b.x[b.num_pts + m] = a.x[m];
                        b.y[b.num_pts + m] = a.y[m];
                    }
                    b.num_pts += 4;
                }
            }
        }
    }
}

// ----------------------------------------------------------------------------
// CudaInfer 方法实现
// ----------------------------------------------------------------------------

CudaInfer::CudaInfer() = default;
CudaInfer::~CudaInfer() noexcept { release(); }

void CudaInfer::init(GPUGridAndStride* grid_strides,
                     size_t img_bytes,
                     int max_N,
                     size_t grid_count) {
    if (!grid_strides || img_bytes == 0 || max_N <= 0 || grid_count == 0) {
        fprintf(stderr, "[ERROR] CudaInfer::init invalid arguments\n");
        return;
    }
    d_grid_strides_  = grid_strides;
    grid_count_      = grid_count;
    buf_image_bytes_ = img_bytes;
    buf_max_N_       = max_N;
    CUDA_CHECK(cudaMalloc(&d_input_bgr_, buf_image_bytes_));
    CUDA_CHECK(cudaMalloc(&d_nchw_, INPUT_W * INPUT_H * 3 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_objs_, buf_max_N_ * sizeof(GPUArmorObject)));
    CUDA_CHECK(cudaMalloc(&d_tf_, 9 * sizeof(float)));
}

void CudaInfer::release() {
    if (d_grid_strides_) cudaFree(d_grid_strides_), d_grid_strides_ = nullptr;
    if (d_input_bgr_)    cudaFree(d_input_bgr_),    d_input_bgr_    = nullptr;
    if (d_nchw_)         cudaFree(d_nchw_),         d_nchw_         = nullptr;
    if (d_objs_)         cudaFree(d_objs_),         d_objs_         = nullptr;
    if (d_tf_)           cudaFree(d_tf_),           d_tf_           = nullptr;
}

float* CudaInfer::preprocess(const unsigned char* input_bgr_host,
                             int img_w, int img_h,
                             Eigen::Matrix3f& tf_matrix,
                             cudaStream_t stream) {
    if (!input_bgr_host || !d_input_bgr_ || !d_nchw_) {
        fprintf(stderr, "[ERROR] CudaInfer::preprocess null ptr\n");
        return nullptr;
    }
    if (img_w <= 0 || img_h <= 0) {
        fprintf(stderr, "[ERROR] CudaInfer::preprocess invalid size %d x %d\n", img_w, img_h);
        return nullptr;
    }
    float scale = fminf(INPUT_W / float(img_w), INPUT_H / float(img_h));
    int rw = int(round(img_w * scale)), rh = int(round(img_h * scale));
    int pad_l = (INPUT_W - rw) / 2, pad_t = (INPUT_H - rh) / 2;
    tf_matrix << 1.f/scale, 0, -pad_l/scale,
                 0, 1.f/scale, -pad_t/scale,
                 0, 0, 1;
    size_t img_size = size_t(img_w) * img_h * 3;
    CUDA_CHECK(cudaMemcpyAsync(d_input_bgr_, input_bgr_host, img_size, cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaGetLastError());
    dim3 threads(32, 32);
    dim3 blocks((INPUT_W + 31) / 32, (INPUT_H + 31) / 32);
    letterbox_kernel<<<blocks, threads, 0, stream>>>(d_input_bgr_, img_w, img_h,
        d_nchw_, INPUT_W, INPUT_H, scale, pad_t, pad_l);
    CUDA_CHECK(cudaGetLastError());
    return d_nchw_;
}

std::vector<GPUArmorObject> CudaInfer::postprocess(
    const float* output,
    int N,
    const Eigen::Matrix3f& tfm,
    float conf_th,
    float nms_th,
    int top_k) {

    if (!output || !d_grid_strides_ || !d_tf_ || !d_objs_) {
        fprintf(stderr, "[ERROR] CudaInfer::postprocess null ptr\n");
        return {};
    }
    int valid_N = std::min<int>({N, static_cast<int>(grid_count_), buf_max_N_});
    if (valid_N <= 0) {
        fprintf(stderr, "[WARN] no proposals (N=%d, grid_count=%zu)\n", N, grid_count_);
        return {};
    }

    // Upload transform
    float h_tf[9] = {
        tfm(0,0), tfm(0,1), tfm(0,2),
        tfm(1,0), tfm(1,1), tfm(1,2),
        tfm(2,0), tfm(2,1), tfm(2,2)
    };
    CUDA_CHECK(cudaMemcpy(d_tf_, h_tf, sizeof(h_tf), cudaMemcpyHostToDevice));

    int threads = 256;
    int blocks = (valid_N + threads - 1) / threads;

    // Decode
    decode_kernel<<<blocks, threads>>>(output, d_grid_strides_, valid_N, grid_count_, d_tf_, d_objs_, conf_th);
    CUDA_CHECK(cudaGetLastError());

    // Clear beyond top_k
    clear_invalid_topk<<<blocks, threads>>>(d_objs_, valid_N, top_k);
    CUDA_CHECK(cudaGetLastError());

    // NMS
    nms_kernel<<<blocks, threads>>>(d_objs_, top_k, nms_th);
    CUDA_CHECK(cudaGetLastError());

    // Copy back and host-side sort
    std::vector<GPUArmorObject> host_objs(valid_N);
    CUDA_CHECK(cudaMemcpy(host_objs.data(), d_objs_, valid_N * sizeof(GPUArmorObject), cudaMemcpyDeviceToHost));

    std::sort(host_objs.begin(), host_objs.end(),
        [](const GPUArmorObject &a, const GPUArmorObject &b) {
            return a.confidence > b.confidence;
        });

    if ((int)host_objs.size() > top_k) {
        host_objs.resize(top_k);
    }
    return host_objs;
}


std::vector<GPUArmorObject> CudaInfer::process_trt(nvinfer1::IExecutionContext* ctx,
                                                   void* device_buffers[2],
                                                   int input_idx, int output_idx,
                                                   const unsigned char* input_bgr_host,
                                                   int img_w, int img_h,
                                                   Eigen::Matrix3f& tf_matrix,
                                                   cudaStream_t stream,
                                                   int N, float conf_th,
                                                   float nms_th, int top_k) {
    if (!ctx || !device_buffers[input_idx] || !device_buffers[output_idx]) {
        fprintf(stderr, "[ERROR] CudaInfer::process_trt invalid args\n");
        return {};
    }
    float* nchw = preprocess(input_bgr_host, img_w, img_h, tf_matrix, stream);
    if (!nchw) return {};
    if (!ctx->setTensorAddress("images", nchw) ||
        !ctx->setTensorAddress("output", device_buffers[output_idx])) {
        fprintf(stderr, "[ERROR] CudaInfer::process_trt setTensorAddress failed\n");
        return {};
    }
    if (!ctx->enqueueV3(stream)) {
        fprintf(stderr, "[ERROR] CudaInfer::process_trt enqueueV3 failed\n");
        return {};
    }
    return postprocess(reinterpret_cast<const float*>(device_buffers[output_idx]),
                       N, tf_matrix, conf_th, nms_th, top_k);
}

} // namespace armor_cuda_infer
