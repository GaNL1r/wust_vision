// armor_cuda_infer.cu
#include "armor_infer.hpp"
#include <thrust/device_ptr.h>
#include <thrust/sort.h>
#include <cstdio>
#include <cmath>

#define CUDA_CHECK(call) do {                                \
    cudaError_t err = call;                                  \
    if (err != cudaSuccess) {                                \
        fprintf(stderr, "CUDA error at %s:%d: %s\n",         \
                __FILE__, __LINE__, cudaGetErrorString(err));\
        exit(EXIT_FAILURE);                                  \
    }                                                        \
} while (0)
static const int INPUT_W = 416; // Width of input
static const int INPUT_H = 416; // Height of input
static constexpr int NUM_CLASSES = 8; // Number of classes
static constexpr int NUM_COLORS = 4; // Number of color
static constexpr float MERGE_CONF_ERROR = 0.15;
static constexpr float MERGE_MIN_IOU = 0.9;
namespace armor_cuda_infer {

GPUGridAndStride* init_grid_strides_on_gpu(
    int in_w,
    int in_h,
    const std::vector<int>& strides,
    size_t& device_grid_count)
{
    std::vector<GPUGridAndStride> host_grid_strides;
    for (auto stride : strides) {
        int num_grid_w = in_w / stride;
        int num_grid_h = in_h / stride;
        for (int y = 0; y < num_grid_h; ++y) {
            for (int x = 0; x < num_grid_w; ++x) {
                host_grid_strides.emplace_back(GPUGridAndStride{
                    static_cast<float>(x),
                    static_cast<float>(y),
                    static_cast<float>(stride)});
            }
        }
    }
    device_grid_count = host_grid_strides.size();
    if (device_grid_count == 0) return nullptr;

    GPUGridAndStride* device_grid_strides = nullptr;
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&device_grid_strides),
        device_grid_count * sizeof(GPUGridAndStride)));
    CUDA_CHECK(cudaMemcpy(device_grid_strides, host_grid_strides.data(),
        device_grid_count * sizeof(GPUGridAndStride),
        cudaMemcpyHostToDevice));

    return device_grid_strides;
}

__device__ float bilinear_interpolate(const unsigned char* img, int w, int h, float x, float y, int c) {
    x = fminf(fmaxf(x, 0.f), w - 1.f);
    y = fminf(fmaxf(y, 0.f), h - 1.f);
    int x0 = floorf(x), x1 = min(x0 + 1, w - 1);
    int y0 = floorf(y), y1 = min(y0 + 1, h - 1);
    float dx = x - x0, dy = y - y0;
    float v00 = img[(y0 * w + x0) * 3 + c];
    float v01 = img[(y0 * w + x1) * 3 + c];
    float v10 = img[(y1 * w + x0) * 3 + c];
    float v11 = img[(y1 * w + x1) * 3 + c];
    return (1 - dx) * (1 - dy) * v00 + dx * (1 - dy) * v01 + (1 - dx) * dy * v10 + dx * dy * v11;
}

__global__ void letterbox_kernel(const unsigned char* input_bgr, int img_w, int img_h, float* output_nchw,
    int out_w, int out_h, float scale, int pad_top, int pad_left) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= out_w || y >= out_h) return;
    float in_x = (x - pad_left) / scale;
    float in_y = (y - pad_top) / scale;
    bool pad = (in_x < 0 || in_y < 0 || in_x >= img_w || in_y >= img_h);
    for (int c = 0; c < 3; ++c) {
        int oc = 2 - c;
        float v = pad ? 114.f : bilinear_interpolate(input_bgr, img_w, img_h, in_x, in_y, c);
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

__global__ void decode_kernel(const float* output, const GPUGridAndStride* grid_strides,
    int N, const float* t, GPUArmorObject* results, float conf_thresh) {
    __shared__ float tf[9];
    if (threadIdx.x < 9) tf[threadIdx.x] = t[threadIdx.x];
    __syncthreads();
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    const float* det = output + i * 21;
    if (det[8] < conf_thresh) { results[i].valid = 0; return; }

    float gx = grid_strides[i].grid0;
    float gy = grid_strides[i].grid1;
    float s  = grid_strides[i].stride;
    GPUArmorObject obj;
    for (int j = 0; j < 4; ++j) {
        float px = (det[2 * j] + gx) * s;
        float py = (det[2 * j + 1] + gy) * s;
        float w  = tf[6] * px + tf[7] * py + tf[8];
        obj.x[j] = (tf[0] * px + tf[1] * py + tf[2]) / w;
        obj.y[j] = (tf[3] * px + tf[4] * py + tf[5]) / w;
    }
    obj.confidence = det[8];
    obj.color_id   = argmax(det + 9, NUM_COLORS);
    obj.number_id  = argmax(det + 9 + NUM_COLORS, NUM_CLASSES);
    obj.valid      = 1;
    obj.num_pts    = 4;
    results[i]     = obj;
}

__global__ void clear_invalid_topk(GPUArmorObject* objs, int N, int k) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    if (i >= k) objs[i].valid = 0;
}

__global__ void nms_kernel(GPUArmorObject* objs, int N, float thresh) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N || !objs[i].valid) return;
    for (int j = 0; j < i; ++j) {
        if (!objs[j].valid) continue;
        if (objs[i].color_id == objs[j].color_id && objs[i].number_id == objs[j].number_id) {
            float x1 = fmaxf(fminf(objs[i].x[0], objs[i].x[2]), fminf(objs[j].x[0], objs[j].x[2]));
            float y1 = fmaxf(fminf(objs[i].y[0], objs[i].y[2]), fminf(objs[j].y[0], objs[j].y[2]));
            float x2 = fminf(fmaxf(objs[i].x[0], objs[i].x[2]), fmaxf(objs[j].x[0], objs[j].x[2]));
            float y2 = fminf(fmaxf(objs[i].y[0], objs[i].y[2]), fmaxf(objs[j].y[0], objs[j].y[2]));
            float inter = fmaxf(0.f, x2 - x1) * fmaxf(0.f, y2 - y1);
            float area_i = fabsf((objs[i].x[2] - objs[i].x[0]) * (objs[i].y[2] - objs[i].y[0]));
            float area_j = fabsf((objs[j].x[2] - objs[j].x[0]) * (objs[j].y[2] - objs[j].y[0]));
            if (inter / (area_i + area_j - inter + 1e-6f) > thresh) {
                objs[i].valid = 0;
                break;
            }
        }
    }
}

CudaInfer::CudaInfer() = default;
CudaInfer::~CudaInfer() { release(); }

void CudaInfer::init(GPUGridAndStride* grid_strides, size_t img_bytes, int max_N) {
    d_grid_strides_ = grid_strides;
    buf_image_bytes_ = img_bytes;
    buf_max_N_ = max_N;
    CUDA_CHECK(cudaMalloc(&d_input_bgr_, buf_image_bytes_));
    CUDA_CHECK(cudaMalloc(&d_nchw_, INPUT_W * INPUT_H * 3 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_objs_, buf_max_N_ * sizeof(GPUArmorObject)));
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

std::vector<GPUArmorObject> CudaInfer::postprocess(const float* output,
    int N, const Eigen::Matrix3f& tfm, float conf_th,
    float nms_th, int top_k)
{
    float h_tf[9] = {
        tfm(0, 0), tfm(0, 1), tfm(0, 2),
        tfm(1, 0), tfm(1, 1), tfm(1, 2),
        tfm(2, 0), tfm(2, 1), tfm(2, 2)
    };
    CUDA_CHECK(cudaMemcpy(d_tf_, h_tf, sizeof(h_tf), cudaMemcpyHostToDevice));

    int threads = 256, blocks = (N + threads - 1) / threads;
    decode_kernel<<<blocks, threads>>>(output, d_grid_strides_, N, d_tf_, d_objs_, conf_th);
    CUDA_CHECK(cudaGetLastError());

    thrust::device_ptr<GPUArmorObject> dev_ptr(d_objs_);
    thrust::sort(thrust::device, dev_ptr, dev_ptr + N, ConfidenceComparator());

    clear_invalid_topk<<<blocks, threads>>>(d_objs_, N, top_k);
    nms_kernel<<<blocks, threads>>>(d_objs_, top_k, nms_th);

    std::vector<GPUArmorObject> results(top_k);
    CUDA_CHECK(cudaMemcpy(results.data(), d_objs_,
        top_k * sizeof(GPUArmorObject), cudaMemcpyDeviceToHost));
    return results;
}

} // namespace armor_cuda_infer