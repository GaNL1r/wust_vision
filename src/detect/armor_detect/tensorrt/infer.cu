#include "detect/armor_detect/tensorrt/infer.hpp"
#include <cuda_runtime.h>
#include <Eigen/Dense>
#include <opencv2/opencv.hpp>
#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <thrust/execution_policy.h>
#include <cstdio>
#include <vector>
#include <cmath>

static const int INPUT_W = 416;
static const int INPUT_H = 416;
static constexpr int NUM_CLASSES = 8;
static constexpr int NUM_COLORS = 4;

#define CUDA_CHECK(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(EXIT_FAILURE); \
    } \
} while(0)

__device__ float bilinear_interpolate(
    const unsigned char* img, int w, int h,
    float x, float y, int c)
{
    x = fminf(fmaxf(x, 0.f), w - 1.f);
    y = fminf(fmaxf(y, 0.f), h - 1.f);

    int x0 = (int)floorf(x);
    int x1 = min(x0 + 1, w - 1);
    int y0 = (int)floorf(y);
    int y1 = min(y0 + 1, h - 1);

    float dx = x - x0;
    float dy = y - y0;

    float v00 = img[(y0 * w + x0) * 3 + c];
    float v01 = img[(y0 * w + x1) * 3 + c];
    float v10 = img[(y1 * w + x0) * 3 + c];
    float v11 = img[(y1 * w + x1) * 3 + c];

    return (1 - dx) * (1 - dy) * v00
         + dx * (1 - dy) * v01
         + (1 - dx) * dy * v10
         + dx * dy * v11;
}

__global__ void letterbox_kernel(
    const unsigned char* __restrict__ input_bgr, int img_w, int img_h,
    float* __restrict__ output_nchw, int out_w, int out_h,
    float scale, int pad_top, int pad_left)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= out_w || y >= out_h) return;

    float in_x = (x - pad_left) / scale;
    float in_y = (y - pad_top) / scale;

    bool is_padding = (in_x < 0.f || in_y < 0.f || in_x >= img_w || in_y >= img_h);

    for (int c = 0; c < 3; ++c) {
        int out_c = 2 - c; // BGR->RGB
        float val = is_padding ? 114.0f : bilinear_interpolate(input_bgr, img_w, img_h, in_x, in_y, c);
        output_nchw[out_c * out_h * out_w + y * out_w + x] = val;
    }
}

void cuda_letterbox_blob(
    const unsigned char* input_bgr_host, int img_w, int img_h,
    float* output_nchw_device, int out_w, int out_h,
    Eigen::Matrix3f& transform_matrix,
    cudaStream_t stream)
{
    float scale = std::min(out_w * 1.f / img_w, out_h * 1.f / img_h);
    int resize_w = static_cast<int>(round(img_w * scale));
    int resize_h = static_cast<int>(round(img_h * scale));
    int pad_w = out_w - resize_w;
    int pad_h = out_h - resize_h;
    int pad_left = static_cast<int>(round(pad_w / 2.f - 0.1f));
    int pad_top = static_cast<int>(round(pad_h / 2.f - 0.1f));

    transform_matrix << 1.f / scale, 0, -pad_left / scale,
                        0, 1.f / scale, -pad_top / scale,
                        0, 0, 1;

    size_t img_size = img_w * img_h * 3 * sizeof(unsigned char);
    unsigned char* input_gpu = nullptr;
    CUDA_CHECK(cudaMalloc(&input_gpu, img_size));
    CUDA_CHECK(cudaMemcpyAsync(input_gpu, input_bgr_host, img_size, cudaMemcpyHostToDevice, stream));

    dim3 threads(32, 32);
    dim3 blocks((out_w + threads.x - 1) / threads.x, (out_h + threads.y - 1) / threads.y);

    letterbox_kernel<<<blocks, threads, 0, stream>>>(
        input_gpu, img_w, img_h,
        output_nchw_device, out_w, out_h,
        scale, pad_top, pad_left);

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(stream));

    CUDA_CHECK(cudaFree(input_gpu));
}

__device__ int argmax(const float* ptr, int len) {
    int idx = 0;
    float max_val = ptr[0];
    for (int i = 1; i < len; ++i) {
        if (ptr[i] > max_val) {
            max_val = ptr[i];
            idx = i;
        }
    }
    return idx;
}

__global__ void decode_armor_objects_kernel(
    const float* __restrict__ output,
    const GPUGridAndStride* __restrict__ grid_strides,
    int num_detections,
    const float* __restrict__ transform_matrix,
    GPUArmorObject* __restrict__ results,
    float conf_thresh)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_detections) return;

    const float* det = output + idx * 21;
    float conf = det[8];
    if (conf < conf_thresh) {
        results[idx].valid = 0;
        return;
    }

    GPUGridAndStride gs = grid_strides[idx];
    float stride = (float)gs.stride;
    float gx = (float)gs.grid0;
    float gy = (float)gs.grid1;

    float x[4], y[4];
    for (int i = 0; i < 4; ++i) {
        x[i] = (det[i * 2] + gx) * stride;
        y[i] = (det[i * 2 + 1] + gy) * stride;
    }

    GPUArmorObject obj;
    for (int i = 0; i < 4; ++i) {
        float tx = transform_matrix[0] * x[i] + transform_matrix[1] * y[i] + transform_matrix[2];
        float ty = transform_matrix[3] * x[i] + transform_matrix[4] * y[i] + transform_matrix[5];
        float w  = transform_matrix[6] * x[i] + transform_matrix[7] * y[i] + transform_matrix[8];
        w = (fabsf(w) < 1e-6f) ? 1e-6f : w;
        obj.x[i] = tx / w;
        obj.y[i] = ty / w;
    }

    obj.confidence = conf;

    constexpr int color_offset = 9;
    constexpr int num_offset = 9 + NUM_COLORS;
    int best_color = 0, best_number = 0;
    float best_color_val = det[color_offset];
    float best_number_val = det[num_offset];

    for (int i = 1; i < NUM_COLORS; ++i) {
        if (det[color_offset + i] > best_color_val) {
            best_color_val = det[color_offset + i];
            best_color = i;
        }
    }

    for (int i = 1; i < NUM_CLASSES; ++i) {
        if (det[num_offset + i] > best_number_val) {
            best_number_val = det[num_offset + i];
            best_number = i;
        }
    }

    obj.color_id = best_color;
    obj.number_id = best_number;
    obj.valid = 1;

    results[idx] = obj;
}

struct ConfidenceComparator {
    __host__ __device__
    bool operator()(const GPUArmorObject& a, const GPUArmorObject& b) const {
        return a.confidence > b.confidence;
    }
};

__global__ void clear_valid_out_of_topk_kernel(GPUArmorObject* objs, int num, int k) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num) return;
    if (idx >= k) objs[idx].valid = 0;
}

void gpu_topk_sort(GPUArmorObject* device_results, int num_results, int top_k) {
    thrust::device_ptr<GPUArmorObject> dev_ptr(device_results);
    thrust::sort(thrust::device, dev_ptr, dev_ptr + num_results, ConfidenceComparator());

    int threads = 256;
    int blocks = (num_results + threads - 1) / threads;
    clear_valid_out_of_topk_kernel<<<blocks, threads>>>(device_results, num_results, top_k);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}

__device__ float iou_gpu(const GPUArmorObject& a, const GPUArmorObject& b) {
    float x1 = max(min(a.x[0], a.x[2]), min(b.x[0], b.x[2]));
    float y1 = max(min(a.y[0], a.y[2]), min(b.y[0], b.y[2]));
    float x2 = min(max(a.x[0], a.x[2]), max(b.x[0], b.x[2]));
    float y2 = min(max(a.y[0], a.y[2]), max(b.y[0], b.y[2]));

    float w = max(0.f, x2 - x1);
    float h = max(0.f, y2 - y1);
    float inter = w * h;

    float area_a = fabs((a.x[2] - a.x[0]) * (a.y[2] - a.y[0]));
    float area_b = fabs((b.x[2] - b.x[0]) * (b.y[2] - b.y[0]));

    return inter / (area_a + area_b - inter);
}

__global__ void nms_kernel(GPUArmorObject* objs, int num_objs, float nms_thresh) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_objs) return;
    if (objs[i].valid == 0) return;

    for (int j = 0; j < i; j++) {
        if (objs[j].valid == 0) continue;

        if (objs[i].color_id == objs[j].color_id && objs[i].number_id == objs[j].number_id) {
            float iou = iou_gpu(objs[i], objs[j]);
            if (iou > nms_thresh) {
                objs[i].valid = 0;
                break;
            }
        }
    }
}

std::vector<GPUArmorObject> postProcessGpu(
    const float* output,
    int num_detections,
    const Eigen::Matrix3f& transform_matrix,
    GPUGridAndStride* device_grid_strides,
    float conf_threshold,
    float nms_threshold,
    int top_k)
{
    GPUArmorObject* device_results = nullptr;
    CUDA_CHECK(cudaMalloc(&device_results, num_detections * sizeof(GPUArmorObject)));

    float h_transform[9] = {
        transform_matrix(0,0), transform_matrix(0,1), transform_matrix(0,2),
        transform_matrix(1,0), transform_matrix(1,1), transform_matrix(1,2),
        transform_matrix(2,0), transform_matrix(2,1), transform_matrix(2,2),
    };

    float* d_transform = nullptr;
    CUDA_CHECK(cudaMalloc(&d_transform, 9 * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_transform, h_transform, 9 * sizeof(float), cudaMemcpyHostToDevice));

    int threads = 256;
    int blocks = (num_detections + threads - 1) / threads;

    decode_armor_objects_kernel<<<blocks, threads>>>(
        output,
        device_grid_strides,
        num_detections,
        d_transform,
        device_results,
        conf_threshold);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());


    gpu_topk_sort(device_results, num_detections, top_k);

    int blocks_nms = (top_k + threads - 1) / threads;
    nms_kernel<<<blocks_nms, threads>>>(device_results, top_k, nms_threshold);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<GPUArmorObject> host_results(top_k);
    CUDA_CHECK(cudaMemcpy(host_results.data(), device_results, top_k * sizeof(GPUArmorObject), cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(device_results));
    CUDA_CHECK(cudaFree(d_transform));

    return host_results;
}
