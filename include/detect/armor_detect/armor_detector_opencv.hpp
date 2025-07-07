// Copyright Chen Jun 2023. Licensed under the MIT License.
//
// Additional modifications and features by Chengfu Zou, Labor. Licensed under
// Apache License 2.0.
//
// Copyright (C) FYT Vision Group. All rights reserved.
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

// std
#include <cmath>
#include <string>
#include <vector>
// third party
#include <opencv2/core.hpp>
#include <opencv2/core/types.hpp>
// project
#include "detect/armor_detect/light_corner_corrector.hpp"
#include "detect/armor_detect/number_classifier.hpp"
#include "type/type.hpp"

class ArmorDetectOpenCV {
public:
    using DetectorCallback = std::function<void(
        const std::vector<ArmorObject>&,
        std::chrono::steady_clock::time_point,
        const cv::Mat&,
        Eigen::Matrix4d
    )>;

    ArmorDetectOpenCV(
        const std::string& classify_model_path,
        const std::string& classify_label_path,
        const int& bin_thres,
        const double& classifier_threshold,
        const LightParams& l,
        const ArmorParams& a
    );

    std::vector<ArmorObject> detect(const cv::Mat& input) noexcept;
    void pushInput(
        const cv::Mat& rgb_img,
        std::chrono::steady_clock::time_point timestamp,
        Eigen::Matrix4d T_camera_to_odom
    );

    cv::Mat preprocessImage(const cv::Mat& input, cv::Mat& gray_img_) noexcept;
    std::vector<Light> findLights(const cv::Mat& rbg_img, const cv::Mat& binary_img) noexcept;
    std::vector<ArmorObject> matchLights(const std::vector<Light>& lights) noexcept;

    cv::Mat extractNumber(const cv::Mat& src, const ArmorObject& armor) const noexcept;

    void setCallback(DetectorCallback callback);
    // Parameters
    int binary_thres;

    LightParams light_params;
    ArmorParams armor_params;

    std::unique_ptr<LightCornerCorrector> corner_corrector;

    double classifier_threshold_ = 0.5;

private:
    bool isLight(const Light& possible_light) noexcept;
    bool containLight(const int i, const int j, const std::vector<Light>& lights) noexcept;
    ArmorType isArmor(const Light& light_1, const Light& light_2) noexcept;
    void topts(ArmorObject& armor);

    std::unique_ptr<NumberClassifier> number_classifier_;

    DetectorCallback infer_callback_;
};
