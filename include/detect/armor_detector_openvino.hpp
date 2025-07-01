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
#include "detect/light_corner_corrector.hpp"
#include "detect/mono_measure_tool.hpp"
#include "eigen3/Eigen/Dense"
#include "fmt/color.h"
#include "fmt/core.h"
#include "fmt/printf.h"
#include "opencv2/opencv.hpp"
#include "openvino/openvino.hpp"
#include <filesystem>

class ArmorDetectOpenVino {
public:
    using DetectorCallback = std::function<void(
        const std::vector<ArmorObject>&,
        std::chrono::steady_clock::time_point,
        const cv::Mat&,
        Eigen::Matrix4d
    )>;

    explicit ArmorDetectOpenVino(
        const std::filesystem::path& model_path,
        const std::string& classify_model_path,
        const std::string& classify_label_path,
        const std::string& device_name,
        const LightParams& l,
        const ArmorParams& a,
        float conf_threshold = 0.25,
        int top_k = 128,
        float nms_threshold = 0.3,
        float expand_ratio_w = 1.1f,
        float expand_ratio_h = 1.1f,
        int binary_thres_ = 85,
        bool auto_init = true
    );

    ~ArmorDetectOpenVino();

    void init();
    bool processCallback(
        const cv::Mat resized_img,
        Eigen::Matrix3f transform_matrix,
        std::chrono::steady_clock::time_point timestamp,
        const cv::Mat& src_img,
        Eigen::Matrix4d T_camera_to_odom
    );

    void drawresult(const cv::Mat& src_img, std::vector<ArmorObject>& armor_objects);

    void pushInput(
        const cv::Mat& rgb_img,
        std::chrono::steady_clock::time_point timestamp,
        Eigen::Matrix4d T_camera_to_odom
    );
    void initNumberClassifier();
    void setCallback(DetectorCallback callback);
    bool classifyNumber(ArmorObject& armor);

    std::vector<Light>
    findLights(const cv::Mat& rbg_img, const cv::Mat& binary_img, ArmorObject& armor) noexcept;

    bool isLight(const Light& possible_light) noexcept;
    bool isArmor(const Light& light_1, const Light& light_2) noexcept;

    void detect(ArmorObject& armor);
    void extractNumberImage(const cv::Mat& src, ArmorObject& armor);
    bool refineLightsFromArmorPts(ArmorObject& armor) const;

    LightParams light_params_;
    ArmorParams armor_params_;
    std::unique_ptr<LightCornerCorrector> corner_corrector;
    std::thread image_thread_;
    std::string model_path_;
    std::string classify_model_path_;
    std::string classify_label_path_;
    std::string device_name_;
    float conf_threshold_;
    int top_k_;
    float nms_threshold_;
    std::unique_ptr<ov::Core> ov_core_;
    std::unique_ptr<ov::CompiledModel> compiled_model_;
    std::vector<int> strides_;
    std::vector<GridAndStride> grid_strides_;
    std::unique_ptr<ThreadPool> thread_pool_;
    DetectorCallback infer_callback_;
    cv::dnn::Net number_net_;
    std::vector<std::string> class_names_;
    float number_threshold_;
    int binary_thres_;

    float expand_ratio_w_;
    float expand_ratio_h_;

    bool isinited_ = false;
};