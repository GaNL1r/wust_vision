// armor_cuda_infer.cu
#include "armor_infer.hpp"
#include "letter_box.hpp"
#include <cmath>
#include <cstdio>
#include <cuda_fp16.h>
#include <opencv2/core/hal/interface.h>
#include <thrust/device_ptr.h>
#include <thrust/sort.h>
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf( \
                stderr, \
                "CUDA error at %s:%d: %s\n", \
                __FILE__, \
                __LINE__, \
                cudaGetErrorString(err) \
            ); \
            exit(EXIT_FAILURE); \
        } \
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
    size_t& device_grid_count
) {
    std::vector<GPUGridAndStride> host_grid_strides;
    for (auto stride: strides) {
        int num_grid_w = in_w / stride;
        int num_grid_h = in_h / stride;
        for (int y = 0; y < num_grid_h; ++y) {
            for (int x = 0; x < num_grid_w; ++x) {
                host_grid_strides.emplace_back(GPUGridAndStride { x, y, stride });
            }
        }
    }
    device_grid_count = host_grid_strides.size();
    if (device_grid_count == 0)
        return nullptr;

    GPUGridAndStride* device_grid_strides = nullptr;
    CUDA_CHECK(cudaMalloc(
        reinterpret_cast<void**>(&device_grid_strides),
        device_grid_count * sizeof(GPUGridAndStride)
    ));
    CUDA_CHECK(cudaMemcpy(
        device_grid_strides,
        host_grid_strides.data(),
        device_grid_count * sizeof(GPUGridAndStride),
        cudaMemcpyHostToDevice
    ));

    return device_grid_strides;
}
__global__ void check_invalid_objs(GPUArmorObject* objs, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N)
        return;

    if (objs[i].valid && !isfinite(objs[i].confidence)) {
        printf("[GPU] Invalid obj[%d]: valid=%d, conf=%f\n", i, objs[i].valid, objs[i].confidence);
    }
}

__device__ float3 bilinear_interpolate_rgb_fast(const uchar* img, int w, int h, float x, float y) {
    x = fminf(fmaxf(x, 0.f), w - 1.001f);
    y = fminf(fmaxf(y, 0.f), h - 1.001f);

    int x0 = floorf(x), x1 = x0 + 1;
    int y0 = floorf(y), y1 = y0 + 1;

    float dx = x - x0, dy = y - y0;
    float dx1 = 1.f - dx, dy1 = 1.f - dy;

    const int s0 = (y0 * w + x0) * 3;
    const int s1 = (y0 * w + x1) * 3;
    const int s2 = (y1 * w + x0) * 3;
    const int s3 = (y1 * w + x1) * 3;

    float3 out;
#pragma unroll
    for (int c = 0; c < 3; ++c) {
        float v00 = img[s0 + c];
        float v01 = img[s1 + c];
        float v10 = img[s2 + c];
        float v11 = img[s3 + c];
        float val = dx1 * dy1 * v00 + dx * dy1 * v01 + dx1 * dy * v10 + dx * dy * v11;
        (&out.x)[c] = val;
    }
    return out;
}

__device__ int argmax(const float* ptr, int len) {
    float m = ptr[0];
    int idx = 0;
    for (int i = 1; i < len; ++i) {
        if (ptr[i] > m) {
            m = ptr[i];
            idx = i;
        }
    }
    return idx;
}

__global__ void decode_kernel(
    const float* output,
    const GPUGridAndStride* grid_strides,
    int N,
    const float* t,
    GPUArmorObject* results,
    float conf_thresh
) {
    __shared__ float tf[9];
    if (threadIdx.x < 9)
        tf[threadIdx.x] = t[threadIdx.x];
    __syncthreads();

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N || output == nullptr || grid_strides == nullptr || results == nullptr)
        return;

    const float* det = output + i * 21;
    float conf = det[8];
    GPUArmorObject obj;
    // 初始化全部x,y防止未初始化
    for (int k = 0; k < 16; ++k) {
        obj.x[k] = 0.f;
        obj.y[k] = 0.f;
    }
    obj.valid = 0;
    obj.confidence = 0.f;
    obj.color_id = 0;
    obj.number_id = 0;
    obj.num_pts = 0;

    if (!isfinite(conf) || conf < conf_thresh) {
        results[i] = obj;
        return;
    }

    float gx = grid_strides[i].grid0;
    float gy = grid_strides[i].grid1;
    float s = grid_strides[i].stride;

    for (int j = 0; j < 4; ++j) {
        float px = (det[2 * j] + gx) * s;
        float py = (det[2 * j + 1] + gy) * s;
        float w = tf[6] * px + tf[7] * py + tf[8];
        if (fabsf(w) < 1e-6f)
            w = 1e-6f;
        obj.x[j] = (tf[0] * px + tf[1] * py + tf[2]) / w;
        obj.y[j] = (tf[3] * px + tf[4] * py + tf[5]) / w;
    }

    obj.confidence = conf;
    obj.color_id = argmax(det + 9, NUM_COLORS);
    obj.number_id = argmax(det + 9 + NUM_COLORS, NUM_CLASSES);
    obj.valid = 1;
    obj.num_pts = 4;

    results[i] = obj;
}

__global__ void clear_invalid_topk(GPUArmorObject* objs, int N, int k) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N)
        return;
    if (i >= k)
        objs[i].valid = 0;
}

__global__ void nms_kernel(GPUArmorObject* objs, int N, float thresh) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N || !objs[i].valid)
        return;

    GPUArmorObject& a = objs[i];

    for (int j = 0; j < i; ++j) {
        GPUArmorObject& b = objs[j];
        if (!b.valid)
            continue;

        if (a.color_id == b.color_id && a.number_id == b.number_id) {
            // Compute IoU
            float x1 = fmaxf(fminf(a.x[0], a.x[2]), fminf(b.x[0], b.x[2]));
            float y1 = fmaxf(fminf(a.y[0], a.y[2]), fminf(b.y[0], b.y[2]));
            float x2 = fminf(fmaxf(a.x[0], a.x[2]), fmaxf(b.x[0], b.x[2]));
            float y2 = fminf(fmaxf(a.y[0], a.y[2]), fmaxf(b.y[0], b.y[2]));
            float inter = fmaxf(0.f, x2 - x1) * fmaxf(0.f, y2 - y1);
            float area_a = fabsf((a.x[2] - a.x[0]) * (a.y[2] - a.y[0]));
            float area_b = fabsf((b.x[2] - b.x[0]) * (b.y[2] - b.y[0]));
            float union_area = area_a + area_b - inter + 1e-6f;
            float iou = inter / union_area;

            if (iou > thresh) {
                a.valid = 0;
                break;
            }

            // 合并点
            if (iou > MERGE_MIN_IOU && fabsf(a.confidence - b.confidence) < MERGE_CONF_ERROR) {
                int n = b.num_pts;
                if (n + 4 <= 16) {
                    for (int k = 0; k < 4; ++k) {
                        b.x[n + k] = a.x[k];
                        b.y[n + k] = a.y[k];
                    }
                    b.num_pts += 4;
                }
            }
        }
    }
}

CudaInfer::CudaInfer() = default;
CudaInfer::~CudaInfer() {
    release();
}

void CudaInfer::init(
    GPUGridAndStride* grid_strides,
    size_t img_bytes,
    int max_N,
    size_t grid_count
) {
    d_grid_strides_ = grid_strides;
    buf_image_bytes_ = img_bytes;
    buf_max_N_ = max_N;
    grid_count_ = grid_count;
    CUDA_CHECK(cudaMalloc(&d_input_bgr_, buf_image_bytes_));
    CUDA_CHECK(cudaMalloc(&d_nchw_, INPUT_W * INPUT_H * 3 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_objs_, buf_max_N_ * sizeof(GPUArmorObject)));
    CUDA_CHECK(cudaMalloc(&d_tf_, 9 * sizeof(float)));
}

void CudaInfer::release() {
    if (d_grid_strides_)
        cudaFree(d_grid_strides_), d_grid_strides_ = nullptr;
    if (d_input_bgr_)
        cudaFree(d_input_bgr_), d_input_bgr_ = nullptr;
    if (d_nchw_)
        cudaFree(d_nchw_), d_nchw_ = nullptr;
    if (d_objs_)
        cudaFree(d_objs_), d_objs_ = nullptr;
    if (d_tf_)
        cudaFree(d_tf_), d_tf_ = nullptr;
}

float* CudaInfer::preprocess(
    const unsigned char* input_bgr_host,
    int img_w,
    int img_h,
    Eigen::Matrix3f& tf_matrix,
    cudaStream_t stream
) {
    if (!isInitialized()) {
        throw std::runtime_error("CudaInfer not initialized properly.");
    }

    if (!input_bgr_host || !d_input_bgr_ || !d_nchw_) {
        fprintf(stderr, "[Error] Null pointer in preprocess input\n");
        return nullptr;
    }

    float scale = fminf(INPUT_W / (float)img_w, INPUT_H / (float)img_h);
    int rw = round(img_w * scale), rh = round(img_h * scale);
    int pad_l = (INPUT_W - rw) / 2, pad_t = (INPUT_H - rh) / 2;

    tf_matrix << 1.f / scale, 0, -pad_l / scale, 0, 1.f / scale, -pad_t / scale, 0, 0, 1;

    size_t img_size = img_w * img_h * 3;
    CUDA_CHECK(
        cudaMemcpyAsync(d_input_bgr_, input_bgr_host, img_size, cudaMemcpyHostToDevice, stream)
    );

    dim3 threads(TILE_W, TILE_H);
    dim3 blocks((INPUT_W + TILE_W - 1) / TILE_W, (INPUT_H + TILE_H - 1) / TILE_H);

    letterbox_kernel_shared<<<blocks, threads, 0, stream>>>(
        d_input_bgr_,
        img_w,
        img_h,
        d_nchw_,
        INPUT_W,
        INPUT_H,
        scale,
        pad_t,
        pad_l
    );

    CUDA_CHECK(cudaGetLastError());
    return d_nchw_;
}

std::vector<GPUArmorObject> CudaInfer::postprocess(
    const float* output,
    int N,
    const Eigen::Matrix3f& tfm,
    float conf_th,
    float nms_th,
    int top_k
) {
    if (!isInitialized()) {
        throw std::runtime_error("CudaInfer not initialized properly.");
    }
    if (!output || !d_grid_strides_ || !d_tf_ || !d_objs_) {
        throw std::invalid_argument("[Error] Null pointer in postprocess input");
    }

    if (N <= 0) {
        std::cerr << "[Warn] postprocess called with N <= 0" << std::endl;
        return {};
    }

    if (N > (int)buf_max_N_) {
        std::cerr << "[Warn] Requested N=" << N << " exceeds max buf size=" << buf_max_N_
                  << ", clamping." << std::endl;
        N = buf_max_N_; // truncate to prevent overflow
    }

    // Copy TF matrix
    float h_tf[9] = { tfm(0, 0), tfm(0, 1), tfm(0, 2), tfm(1, 0), tfm(1, 1),
                      tfm(1, 2), tfm(2, 0), tfm(2, 1), tfm(2, 2) };
    CUDA_CHECK(cudaMemcpy(d_tf_, h_tf, sizeof(h_tf), cudaMemcpyHostToDevice));

    // Kernel: decode
    int threads = 256;
    int blocks = (N + threads - 1) / threads;
    decode_kernel<<<blocks, threads>>>(output, d_grid_strides_, N, d_tf_, d_objs_, conf_th);
    CUDA_CHECK(cudaGetLastError());

#ifdef DEBUG_GPU_OBJS
    check_invalid_objs<<<blocks, threads>>>(d_objs_, N);
    CUDA_CHECK(cudaDeviceSynchronize());
#endif

    // Sort by confidence
    int sort_N = std::min(N, (int)buf_max_N_);

    if (top_k > sort_N) {
        top_k = sort_N;
    }

    thrust::device_ptr<GPUArmorObject> dev_ptr(d_objs_);
    thrust::sort(thrust::device, dev_ptr, dev_ptr + sort_N, ConfidenceComparator());

    clear_invalid_topk<<<blocks, threads>>>(d_objs_, sort_N, top_k);
    CUDA_CHECK(cudaGetLastError());

    nms_kernel<<<blocks, threads>>>(d_objs_, top_k, nms_th);
    CUDA_CHECK(cudaGetLastError());

    // Copy back result
    std::vector<GPUArmorObject> results(top_k);
    CUDA_CHECK(
        cudaMemcpy(results.data(), d_objs_, top_k * sizeof(GPUArmorObject), cudaMemcpyDeviceToHost)
    );

    return results;
}

std::vector<GPUArmorObject> CudaInfer::process_trt(
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
) {
    void* input_tensor_ptr = preprocess(input_bgr_host, img_w, img_h, tf_matrix, stream);

    context->setTensorAddress("images", input_tensor_ptr);
    context->setTensorAddress("output", device_buffers[output_idx_]);

    if (!context->enqueueV3(stream)) {
        std::cerr << "enqueueV3 failed!";
        return {};
    }

    std::vector<GPUArmorObject> results =
        postprocess((float*)device_buffers[output_idx_], N, tf_matrix, conf_th, nms_th, top_k);
    return results;
}

} // namespace armor_cuda_infer