// Copyright 2023 Yunlong Feng
// Copyright 2025 Lihan Chen
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

#include "common/ThreadPool.h"
#include "common/logger.hpp"
#include "detect/armor_detect/armor_detect_common.hpp"
#include "eigen3/Eigen/Dense"
#include "fmt/color.h"
#include "fmt/core.h"
#include "fmt/printf.h"
#include "opencv2/opencv.hpp"
#include "openvino/openvino.hpp"
#include <filesystem>

class ArmorDetectOpenVino {
public:
    using DetectorCallback =
        std::function<void(const std::vector<armor::ArmorObject>&, const CommonFrame&)>;

    explicit ArmorDetectOpenVino(
        const std::filesystem::path& model_path,
        const std::string& device_name,
        const ArmorDetectCommonParams& armor_detect_common_params,
        float conf_threshold = 0.25,
        int top_k = 128,
        float nms_threshold = 0.3,
        bool use_throughputmode_ = false,
        bool use_armor_detect_common = true
    );

    ~ArmorDetectOpenVino();

    void init();
    bool processCallback(
        const cv::Mat resized_img,
        Eigen::Matrix3f transform_matrix,
        const CommonFrame& frame
    );

    void drawResult(const cv::Mat& src_img, std::vector<armor::ArmorObject>& armor_objects);

    void pushInput(const CommonFrame& frame);

    void setCallback(DetectorCallback callback);

private:
    std::string model_path_;
    std::string device_name_;
    float conf_threshold_;
    int top_k_;
    float nms_threshold_;
    std::unique_ptr<ov::Core> ov_core_;
    std::unique_ptr<ov::CompiledModel> compiled_model_;
    std::shared_ptr<ov::Model> model_;
    ov::preprocess::PrePostProcessor* ppp_;
    std::vector<int> strides_;
    std::vector<GridAndStride> grid_strides_;
    DetectorCallback infer_callback_;
    std::unique_ptr<ArmorDetectCommon> armor_detect_common_;
    bool use_throughputmode_ = false;
    bool use_armor_detect_common_ = true;
};