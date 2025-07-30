#pragma once

#include "cuda_runtime_api.h"
#include <Eigen/Dense>
#include <opencv2/core/hal/interface.h>
#include <vector>

struct GPUArmorObject {
    float x[4];
    float y[4];
    float confidence;
    int color_id;
    int number_id;
    int valid; // 标志是否保留
    __host__ __device__ GPUArmorObject(): confidence(0.f), color_id(0), number_id(0), valid(0) {
        for (int i = 0; i < 4; ++i) {
            x[i] = 0.f;
            y[i] = 0.f;
        }
    }
};
struct GPUGridAndStride {
    int grid0;
    int grid1;
    int stride;
};

void cuda_letterbox_blob(
    const unsigned char* input_bgr_host,
    int img_w,
    int img_h,
    float* output_nchw_device,
    int out_w,
    int out_h,
    Eigen::Matrix3f& transform_matrix,
    cudaStream_t stream
);

// GPU端后处理主流程，输入模型原始输出和格点信息，返回检测结果
std::vector<GPUArmorObject> postProcessGpu(
    const float* output, // 模型输出 GPU内存指针
    int num_detections, // 检测框数
    const Eigen::Matrix3f& transform_matrix,
    GPUGridAndStride* device_grid_strides, // GPU内存，格点信息数组
    float conf_threshold,
    float nms_threshold,
    int top_k
);
