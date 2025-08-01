#pragma once

#include <Eigen/Dense>
#include <NvInferRuntime.h>
#include <cuda_runtime.h>
#include <memory>
#include <vector>

namespace rune_cuda_infer {

struct GPUGridAndStride {
    int grid0, grid1, stride;
};

struct GPURuneObject {
    float x[10]; // [r_center, bl, tl, tr, br]
    float y[10];
    float confidence;
    int color_id;
    int type_id;
    int valid;
    int num_pts;

    __host__ __device__ GPURuneObject():
        confidence(0.f),
        color_id(-1),
        type_id(-1),
        valid(0),
        num_pts(0) {
#pragma unroll
        for (int i = 0; i < 5; ++i) {
            x[i] = 0.f;
            y[i] = 0.f;
        }
    }
};

template<typename T>
struct ConfidenceComparator {
    __host__ __device__ bool operator()(const T& a, const T& b) const {
        return a.confidence > b.confidence;
    }
};

// Allocate and upload grid+stride info to GPU
GPUGridAndStride* init_grid_strides_on_gpu(
    int input_w,
    int input_h,
    const std::vector<int>& strides,
    size_t& device_grid_count
);

class CudaInfer {
public:
    CudaInfer();
    ~CudaInfer();

    /// Initialize GPU buffers (grid, image buffer, max detections)
    void init(GPUGridAndStride* grid_strides, size_t img_bytes, int max_N);

    /// Release GPU resources
    void release();

    /// Preprocess: letterbox + convert BGR to NCHW on device
    /// @returns device pointer to NCHW buffer
    float* preprocess(
        const unsigned char* input_bgr_host,
        int img_w,
        int img_h,
        Eigen::Matrix3f& tf_matrix,
        cudaStream_t stream
    );

    /// Postprocess: decode, top-K, and NMS
    std::vector<GPURuneObject> postprocess(
        const float* output,
        int N,
        const Eigen::Matrix3f& tf_matrix,
        float conf_th,
        float nms_th,
        int top_k
    );

    /// Full pipeline: preprocess, inference enqueueV3, postprocess
    std::vector<GPURuneObject> process_trt(
        nvinfer1::IExecutionContext* context,
        void* device_buffers[2],
        int input_idx,
        int output_idx,
        const unsigned char* input_bgr_host,
        int img_w,
        int img_h,
        Eigen::Matrix3f& tf_matrix,
        cudaStream_t stream,
        int N,
        float conf_th,
        float nms_th,
        int top_k
    );

    // disable copy
    CudaInfer(const CudaInfer&) = delete;
    CudaInfer& operator=(const CudaInfer&) = delete;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace rune_cuda_infer
