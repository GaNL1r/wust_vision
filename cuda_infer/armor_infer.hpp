#pragma once

#include <Eigen/Dense>
#include <NvInferRuntime.h>
#include <cuda_runtime.h>
#include <memory>
#include <vector>

namespace armor_cuda_infer {

struct GPUGridAndStride {
    int grid0, grid1, stride;
};

struct GPUArmorObject {
    float x[16];
    float y[16];
    float confidence;
    int color_id;
    int number_id;
    int valid;
    int num_pts;

    __host__ __device__ GPUArmorObject() {
        for (int i = 0; i < 4; ++i) {
            x[i] = 0.0f;
            y[i] = 0.0f;
        }
        confidence = 0.0f;
        color_id = 0;
        number_id = 0;
        valid = 0;
        num_pts = 0;
    }
};

struct ConfidenceComparator {
    __host__ __device__ bool operator()(const GPUArmorObject& a, const GPUArmorObject& b) const {
        return a.confidence > b.confidence;
    }
};

// Allocate and upload grid+stride information to GPU
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

    // Initialize GPU buffers: grid strides, image buffer size, and maximum detections
    void init(GPUGridAndStride* grid_strides, size_t img_bytes, int max_N);

    // Release all GPU resources
    void release();

    // Preprocess: letterbox + convert BGR to NCHW on device
    // @param input_bgr_host Host-side BGR image pointer
    // @param img_w, img_h   Input image width and height
    // @param tf_matrix      Output inverse transform for postprocessing
    // @param stream         CUDA stream for async copy and kernel launch
    // @returns Device pointer to NCHW buffer
    float* preprocess(
        const unsigned char* input_bgr_host,
        int img_w,
        int img_h,
        Eigen::Matrix3f& tf_matrix,
        cudaStream_t stream
    );

    // Postprocess: decode detections, top-K filter, and NMS
    // @param output     Device pointer to raw network output
    // @param N          Number of grid points
    // @param tf_matrix  Inverse transform from preprocess
    // @param conf_th    Confidence threshold for decoding
    // @param nms_th     IoU threshold for NMS
    // @param top_k      Maximum detections to retain
    // @returns Vector of detected objects on host
    std::vector<GPUArmorObject> postprocess(
        const float* output,
        int N,
        const Eigen::Matrix3f& tf_matrix,
        float conf_th,
        float nms_th,
        int top_k
    );

    // Full pipeline: preprocess, infer (enqueueV3), then postprocess
    std::vector<GPUArmorObject> process_trt(
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

} // namespace armor_cuda_infer