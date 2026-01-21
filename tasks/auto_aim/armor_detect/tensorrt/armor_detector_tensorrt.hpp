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

#include "cuda_infer/armor_infer.hpp"
#include "tasks/auto_aim/armor_detect/armor_detect_common.hpp"
#include "tasks/auto_aim/armor_detect/armor_infer.hpp"
#include "wust_vl/common/concurrency/adaptive_resource_pool.hpp"
#include "wust_vl/ml_net/tensorrt/tensorrt_net.hpp"
class ArmorDetectTrt {
public:
    using DetectorCallback =
        std::function<void(const std::vector<armor::ArmorObject>&, const CommonFrame&)>;
    struct Params {
        float conf_threshold = 0.3; // 置信度阈值
        float nms_threshold = 0.5; // NMS阈值
        int top_k = 128; // 最大检测框数
        int device_id = 0;
        int max_infer_running;
        double min_free_mem_ratio;
        bool use_cuda_pre = false;
        bool log_time = false;
    };
    struct Infer {
        std::unique_ptr<nvinfer1::IExecutionContext> context;
        std::unique_ptr<armor_cuda_infer::CudaInfer> cuda_infer;
    };

    // 构造函数：加载 ONNX 模型并构建 TensorRT 引擎
    explicit ArmorDetectTrt(
        std::string model_type,
        const std::string& onnx_path,
        const Params& params,
        const ArmorDetectCommonParams& common_params,
        bool use_armor_detect_common = true
    );

    ~ArmorDetectTrt();

    void pushInput(CommonFrame& frame, const std::optional<armor::ArmorNumber>& target_number);

    bool processCallback(
        const CommonFrame& frame,
        Infer* infer,
        const std::optional<armor::ArmorNumber>& target_number
    );
    void setCallback(DetectorCallback callback);

private:
    // 成员变量
    Params params_;
    nvinfer1::Dims input_dims_;
    nvinfer1::Dims output_dims_;
    std::vector<int> strides_;
    std::vector<GridAndStride> grid_strides_;
    DetectorCallback infer_callback_;
    std::unique_ptr<ArmorDetectCommon> armor_detect_common_;
    bool use_armor_detect_common_ = true;
    std::unique_ptr<AdaptiveResourcePool<Infer>> infer_pool_;
    std::unique_ptr<armor_infer::ArmorInfer> armor_infer_;
    int current_id_ = 0;
    std::unique_ptr<ml_net::TensorRTNet> trt_net_;
};
