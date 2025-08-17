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
#include "detect/armor_detect/armor_detect_common.hpp"
#include "detect/armor_detect/armor_infer.hpp"
#include "detect/armor_detect/light_corner_corrector.hpp"
#include "detect/mono_measure_tool.hpp"
#include "eigen3/Eigen/Dense"
#include "fmt/color.h"
#include "fmt/core.h"
#include "fmt/printf.h"
#include "ncnn/net.h"
#include "opencv2/opencv.hpp"
#include "wust_vl/common/concurrency/ThreadPool.h"
#include "wust_vl/common/utils/logger.hpp"
#include "wust_vl/ml_net/ncnn/ncnn_net.hpp"
class ArmorDetectNCNN {
public:
    using DetectorCallback =
        std::function<void(const std::vector<armor::ArmorObject>&, const CommonFrame&)>;
    explicit ArmorDetectNCNN(
        std::string model_type,
        std::string input_name_,
        std::string output_name_,
        const std::string& model_path_param_,
        const std::string& model_path_bin_,
        const ArmorDetectCommonParams& armor_detect_common_params,
        float conf_threshold = 0.25,
        int top_k = 128,
        float nms_threshold = 0.3,
        bool use_gpu = false,
        int cpu_threads = 1,
        bool use_lightmode = true,
        bool use_armor_detect_common = true,
        int device_id = 0
    );
    ~ArmorDetectNCNN();
    void init(int device_id);
    bool processCallback(const CommonFrame& frame);

    void setCallback(DetectorCallback callback);
    void pushInput(CommonFrame& frame);

private:
    std::string model_path_param_;
    std::string model_path_bin_;
    DetectorCallback infer_callback_;
    std::vector<int> strides_;
    std::vector<GridAndStride> grid_strides_;
    std::string input_name_;
    std::string output_name_;
    float conf_threshold_;
    int top_k_;
    float nms_threshold_;
    bool use_gpu_;
    int cpu_threads_;
    bool use_lightmode_ = true;
    std::unique_ptr<ArmorDetectCommon> armor_detect_common_;
    bool use_armor_detect_common_ = true;
    std::unique_ptr<armor_infer::ArmorInfer> armor_infer_;
    int current_id_ = 0;
    std::unique_ptr<ml_net::NCNNNet> ncnn_net_;
};