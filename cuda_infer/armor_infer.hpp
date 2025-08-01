// armor_cuda_infer.hpp
#pragma once

#include <Eigen/Dense>
#include <NvInferRuntime.h>
#include <cuda_runtime.h>
#include <iostream>
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

// 用于 thrust::sort 的比较器
struct ConfidenceComparator {
    __host__ __device__ bool operator()(const GPUArmorObject& a, const GPUArmorObject& b) const {
        return a.confidence > b.confidence;
    }
};
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

    /// 一次性申请所有 GPU 资源
    void init(GPUGridAndStride* grid_strides, size_t img_bytes, int max_N);

    /// 释放所有 GPU 资源
    void release();

    /// 预处理：letterbox + NCHW 转存
    /// @param  input_bgr_host  host 侧 BGR 图数据
    /// @param  img_w, img_h    原图宽高
    /// @param  output_nchw     返回 device 侧 NCHW 缓冲指针
    /// @param  tf_matrix       输出逆变换矩阵，用于后处理映射
    /// @param  stream          CUDA stream
    float* preprocess(
        const unsigned char* input_bgr_host,
        int img_w,
        int img_h,
        Eigen::Matrix3f& tf_matrix,
        cudaStream_t stream
    );

    /// 后处理：decode + TopK + NMS
    /// @param  output          device 侧模型原始输出
    /// @param  N               网格总点数
    /// @param  tf_matrix       preprocess 返回的逆变换矩阵
    /// @param  grid_strides    device 侧 grid+stride 数组
    /// @param  conf_th         置信度阈值
    /// @param  nms_th          NMS 阈值
    /// @param  top_k           保留前 K
    /// @return                  Host vector of valid detections
    std::vector<GPUArmorObject> postprocess(
        const float* output,
        int N,
        const Eigen::Matrix3f& tf_matrix,
        float conf_th,
        float nms_th,
        int top_k
    );

    std::vector<GPUArmorObject> process_trt(
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
    );

private:
    // 禁用拷贝
    CudaInfer(const CudaInfer&) = delete;
    CudaInfer& operator=(const CudaInfer&) = delete;

    // 设备缓冲
    unsigned char* d_input_bgr_ = nullptr; // 原始 BGR
    float* d_nchw_ = nullptr; // letterbox 后的 NCHW
    GPUArmorObject* d_objs_ = nullptr; // decode & sort 输出
    float* d_tf_ = nullptr; // 3×3 逆变换矩阵
    GPUGridAndStride* d_grid_strides_;
    // 缓冲大小
    size_t buf_image_bytes_;
    int buf_max_N_;
};

} // namespace armor_cuda_infer
