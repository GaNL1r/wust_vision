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

#include "tasks/auto_aim/armor_detect/light_corner_corrector.hpp"
#include "tasks/auto_aim/armor_detect/number_classifier.hpp"
#include "tasks/auto_aim/type.hpp"
struct ArmorDetectCommonParams {
    std::string classify_model_path;
    std::string classify_label_path;
    double classifier_threshold = 0.5;
    armor::LightParams light_params;
    armor::ArmorParams armor_params;
    float expand_ratio_w = 1.1f;
    float expand_ratio_h = 1.1f;
    int binary_thres = 85;
    double max_pts_error = 20.0;
};

class ArmorDetectCommon {
public:
    ArmorDetectCommon(const ArmorDetectCommonParams& params);

    std::vector<armor::Light> findLights(
        const cv::Mat& color_img,
        const cv::Mat& binary_img,
        armor::ArmorObject& armor
    ) noexcept;

    bool isLight(const armor::Light& possible_light) noexcept;
    bool isArmor(const armor::Light& light_1, const armor::Light& light_2) noexcept;

    std::vector<armor::ArmorObject> detectNet(
        const cv::Mat& src_img,
        std::vector<armor::ArmorObject>& objs_result,
        Eigen::Matrix3f transform_matrix,
        int detect_color
    );
    bool extractNetImage(const cv::Mat& src, armor::ArmorObject& armor);
    bool refineLightsFromArmorPts(armor::ArmorObject& armor) const;
    std::unique_ptr<LightCornerCorrector> corner_corrector_;
    std::unique_ptr<NumberClassifier> number_classifier_;
    ArmorDetectCommonParams params_;
};