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
#include "tasks/auto_aim/armor_detect/opencv/armor_detector_opencv_wrapper.hpp"
#include "tasks/utils.hpp"
#include "wust_vl/common/utils/logger.hpp"
ArmorDetectorOpencvWrapper::ArmorDetectorOpencvWrapper(const YAML::Node& config) {
    auto classify_model_path =
        utils::expandEnv(config["armor_detect_opencv"]["classify"]["model_path"].as<std::string>());
    auto classify_label_path =
        utils::expandEnv(config["armor_detect_opencv"]["classify"]["label_path"].as<std::string>());
    double classify_threshold = config["armor_detect_opencv"]["classify"]["threshold"].as<double>();

    int binary_thres = config["armor_detect_opencv"]["light"]["binary_thres"].as<int>();

    armor::LightParams l_params = {
        .min_ratio = config["armor_detect_opencv"]["light"]["min_ratio"].as<double>(),
        .max_ratio = config["armor_detect_opencv"]["light"]["max_ratio"].as<double>(),
        .max_angle = config["armor_detect_opencv"]["light"]["max_angle"].as<double>(),
        .max_angle_diff = config["armor_detect_opencv"]["light"]["max_angle_diff"].as<double>(),
        .color_diff_thresh = config["armor_detect_opencv"]["light"]["color_diff_thresh"].as<int>()
    };
    armor::ArmorParams a_params = {
        .min_light_ratio = config["armor_detect_opencv"]["armor"]["min_light_ratio"].as<double>(),
        .min_small_center_distance =
            config["armor_detect_opencv"]["armor"]["min_small_center_distance"].as<double>(),
        .max_small_center_distance =
            config["armor_detect_opencv"]["armor"]["max_small_center_distance"].as<double>(),
        .min_large_center_distance =
            config["armor_detect_opencv"]["armor"]["min_large_center_distance"].as<double>(),
        .max_large_center_distance =
            config["armor_detect_opencv"]["armor"]["max_large_center_distance"].as<double>(),
        .max_angle = config["armor_detect_opencv"]["armor"]["max_angle"].as<double>()
    };

    detector_ = std::make_unique<ArmorDetectOpenCV>(
        classify_model_path,
        classify_label_path,
        binary_thres,
        classify_threshold,
        l_params,
        a_params
    );
    detector_->target_height_ = config["armor_detect_opencv"]["target_height"].as<int>();
    detector_->target_width_ = config["armor_detect_opencv"]["target_width"].as<int>();
}

ArmorDetectorOpencvWrapper::~ArmorDetectorOpencvWrapper() = default;

void ArmorDetectorOpencvWrapper::pushInput(
    CommonFrame& frame,
    const std::optional<armor::ArmorNumber>& target_number
) {
    detector_->pushInput(frame, target_number);
}

void ArmorDetectorOpencvWrapper::setCallback(DetectorCallback cb) {
    detector_->setCallback(std::move(cb));
}