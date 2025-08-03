// Copyright 2025 Zikang Xie
// Copyright 2025 XiaoJian Wu
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#pragma once

#include <memory>
#include <string>
#include <vector>

#include "NvInfer.h"
#include "NvInferRuntime.h"
#include "armor_infer.hpp"
#include "common/ThreadPool.h"
#include "common/adaptive_resource_pool.hpp"
#include "common/logger.hpp"
#include "detect/armor_detect/armor_detect_common.hpp"
#include "detect/armor_detect/light_corner_corrector.hpp"
#include "detect/mono_measure_tool.hpp"
#include "eigen3/Eigen/Dense"
#include "fmt/color.h"
#include "fmt/core.h"
#include "fmt/printf.h"
#include "opencv2/opencv.hpp"

// 定义检测框结构体，与 OpenVINO 模型输出对齐
class ArmorDetectTrt {
public:
    // 初始化参数结构体
    using DetectorCallback =
        std::function<void(const std::vector<armor::ArmorObject>&, const CommonFrame&)>;
    struct Params {
        int input_w = 416; // 模型输入宽度
        int input_h = 416; // 模型输入高度
        int num_classes = 8; // 类别数 (0-7)
        int num_colors = 4; // 颜色数 (0-3)
        float conf_threshold = 0.3; // 置信度阈值
        float nms_threshold = 0.5; // NMS阈值
        int top_k = 128; // 最大检测框数
        int device_id = 0;
        int max_infer_running;
        double min_free_mem_ratio;
        bool use_cuda_pre = false;
        bool use_cuda_post = false;
        bool log_time = false;
    };
    struct Infer {
        std::unique_ptr<nvinfer1::IExecutionContext> context;
        std::unique_ptr<armor_cuda_infer::CudaInfer> cuda_infer;
    };

    // 构造函数：加载 ONNX 模型并构建 TensorRT 引擎
    explicit ArmorDetectTrt(
        const std::string& onnx_path,
        const Params& params,
        const ArmorDetectCommonParams& common_params,
        bool use_armor_detect_common = true
    );

    // 析构函数：释放资源
    ~ArmorDetectTrt();

    void pushInput(const CommonFrame& frame);

    bool processCallback(const CommonFrame& frame, Infer* infer);
    void setCallback(DetectorCallback callback);

private:
    // TensorRT 引擎初始化
    void buildEngine(const std::string& onnx_path);
    // 后处理：解析输出张量，生成检测框
    std::vector<armor::ArmorObject> postProcess(
        std::vector<armor::ArmorObject>& output_objs,
        std::vector<float>& scores,
        std::vector<cv::Rect>& rects,
        const float* output,
        int num_detections,
        const Eigen::Matrix<float, 3, 3>& transform_matrix
    );

    // 成员变量
    Params params_;
    nvinfer1::ICudaEngine* engine_;
    nvinfer1::IExecutionContext* context_;
    void* device_buffers_[2]; // 输入输出显存指针
    float* output_buffer_; // 输出数据主机内存
    cudaStream_t stream_; // CUDA流
    int input_idx_, output_idx_;
    size_t input_sz_, output_sz_;
    TRTLogger g_logger_;
    DetectorCallback infer_callback_;
    nvinfer1::IRuntime* runtime_ = nullptr;
    std::unique_ptr<ArmorDetectCommon> armor_detect_common_;
    bool use_armor_detect_common_ = true;
    std::shared_ptr<ThreadPool> thread_pool_;
    std::unique_ptr<AdaptiveResourcePool<Infer>> infer_pool_;
};
