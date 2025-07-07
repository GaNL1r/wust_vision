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

#include "detect/armor_detect/light_corner_corrector.hpp"
#include "detect/armor_detect/number_classifier.hpp"
#include "type/type.hpp"

class ArmorDetectCommon {
public:
    ArmorDetectCommon(
        const std::string& classify_model_path,
        const std::string& classify_label_path,
        const LightParams& l,
        const ArmorParams& a,
        double classifier_threshold = 0.5,
        float expand_ratio_w = 1.1f,
        float expand_ratio_h = 1.1f,
        int binary_thres_ = 85
    );

    std::vector<Light>
    findLights(const cv::Mat& rbg_img, const cv::Mat& binary_img, ArmorObject& armor) noexcept;

    bool isLight(const Light& possible_light) noexcept;
    bool isArmor(const Light& light_1, const Light& light_2) noexcept;

    std::vector<ArmorObject>
    detectNet(const cv::Mat& src_img, std::vector<ArmorObject>& objs_result);
    void extractNetImage(const cv::Mat& src, ArmorObject& armor);
    bool refineLightsFromArmorPts(ArmorObject& armor) const;
    LightParams light_params_;
    ArmorParams armor_params_;
    std::unique_ptr<LightCornerCorrector> corner_corrector;
    std::unique_ptr<NumberClassifier> number_classifier_;
    float expand_ratio_w_;
    float expand_ratio_h_;
    int binary_thres_;
    double classifier_threshold;
};