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
#include "tasks/auto_aim/armor_detect/light_corner_corrector.hpp"
#include "tasks/auto_aim/armor_detect/number_classifier.hpp"
#include "tasks/type_common.hpp"
class ArmorDetectOpenCV {
public:
    using DetectorCallback =
        std::function<void(const std::vector<armor::ArmorObject>&, const CommonFrame&)>;

    ArmorDetectOpenCV(
        const std::string& classify_model_path,
        const std::string& classify_label_path,
        const int& bin_thres,
        const double& classifier_threshold,
        const armor::LightParams& l,
        const armor::ArmorParams& a
    );

    std::vector<armor::ArmorObject> detect(const cv::Mat& input, int detect_color) noexcept;
    void pushInput(CommonFrame& frame);

    std::tuple<cv::Mat, cv::Mat> preprocessImage(const cv::Mat& input) noexcept;
    std::vector<armor::Light>
    findLights(const cv::Mat& rbg_img, const cv::Mat& binary_img) noexcept;
    std::vector<armor::ArmorObject>
    matchLights(const std::vector<armor::Light>& lights, int detect_color) noexcept;

    cv::Mat extractNumber(const cv::Mat& src, const armor::ArmorObject& armor) const noexcept;

    void setCallback(DetectorCallback callback);
    // Parameters
    int binary_thres_;

    armor::LightParams light_params_;
    armor::ArmorParams armor_params_;

    std::unique_ptr<LightCornerCorrector> corner_corrector_;

    double classifier_threshold_ = 0.5;

private:
    bool isLight(const armor::Light& possible_light) noexcept;
    bool containLight(const int i, const int j, const std::vector<armor::Light>& lights) noexcept;
    armor::ArmorType isArmor(const armor::Light& light_1, const armor::Light& light_2) noexcept;
    void topts(armor::ArmorObject& armor);

    std::unique_ptr<NumberClassifier> number_classifier_;

    DetectorCallback infer_callback_;
    int current_id_ = 0;
};
