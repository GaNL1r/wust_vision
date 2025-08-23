// Copyright 2025 Xiaojian Wu
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
#include "Eigen/Dense"
#include "fmt/color.h"
#include "fmt/core.h"
#include "fmt/printf.h"
#include "opencv2/opencv.hpp"
#include "tasks/auto_aim/armor_detect/armor_detect_common.hpp"
#include "tasks/auto_aim/armor_detect/armor_infer.hpp"
#include "tasks/type_common.hpp"
#include "wust_vl/common/concurrency/ThreadPool.h"
#include "wust_vl/common/utils/logger.hpp"
#include "wust_vl/ml_net/onnxruntime/onnxruntime_net.hpp"
#include <filesystem>
#include <onnxruntime_cxx_api.h>
class ArmorDetectOnnxRuntime {
public:
    using DetectorCallback =
        std::function<void(const std::vector<armor::ArmorObject>&, const CommonFrame&)>;

    explicit ArmorDetectOnnxRuntime(
        std::string provider,
        std::string model_type,
        const std::filesystem::path& model_path,
        const ArmorDetectCommonParams& armor_detect_common_params,
        float conf_threshold = 0.25,
        int top_k = 128,
        float nms_threshold = 0.3,
        bool use_armor_detect_common_ = true
    );

    ~ArmorDetectOnnxRuntime();

    void init();
    bool processCallback(const CommonFrame& frame);

    void drawResult(const cv::Mat& src_img, std::vector<armor::ArmorObject>& armor_objects);

    void pushInput(CommonFrame& frame);

    void setCallback(DetectorCallback callback);

private:
    std::string model_path_;
    float conf_threshold_;
    int top_k_;
    float nms_threshold_;
    OrtProvider provider_ = OrtProvider::CPU;
    std::vector<int> strides_;
    std::vector<GridAndStride> grid_strides_;
    DetectorCallback infer_callback_;
    std::unique_ptr<ArmorDetectCommon> armor_detect_common_;
    bool use_armor_detect_common_ = true;
    std::unique_ptr<armor_infer::ArmorInfer> armor_infer_;
    int current_id_ = 0;
    std::unique_ptr<ml_net::OnnxRuntimeNet> onnxruntime_net_;
};