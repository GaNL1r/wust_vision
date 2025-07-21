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
#include "detect/armor_detect/tensorrt/armor_detector_trt_wrapper.hpp"

ArmorDetectorTrtWrapper::ArmorDetectorTrtWrapper(
    const YAML::Node& config,
    bool use_armor_detect_common
) {
    auto classify_model_path = config["armor_detect"]["classify"]["model_path"].as<std::string>();
    auto classify_label_path = config["armor_detect"]["classify"]["label_path"].as<std::string>();
    double classify_threshold = config["armor_detect"]["classify"]["threshold"].as<double>();
    const std::string model_path = config["armor_detect"]["model"]["model_path"].as<std::string>();
    ArmorDetectTrt::Params params;
    params.input_w = config["armor_detect"]["model"]["input_w"].as<int>();
    params.input_h = config["armor_detect"]["model"]["input_h"].as<int>();
    params.num_classes = config["armor_detect"]["model"]["num_classes"].as<int>();
    params.num_colors = config["armor_detect"]["model"]["num_colors"].as<int>();
    params.conf_threshold = config["armor_detect"]["model"]["conf_threshold"].as<float>();
    params.nms_threshold = config["armor_detect"]["model"]["nms_threshold"].as<float>();
    params.top_k = config["armor_detect"]["model"]["top_k"].as<int>();
    params.device_id = config["armor_detect"]["model"]["device_id"].as<int>();
    WUST_INFO("armor_detector") << "model_path: " << model_path;
    int binary_thres;
    float expand_ratio_w, expand_ratio_h;
    LightParams l_params;
    ArmorParams a_params;
    if (use_armor_detect_common) {
        binary_thres = config["armor_detect"]["light"]["binary_thres"].as<int>(85);
        expand_ratio_w = config["armor_detect"]["light"]["expand_ratio_w"].as<float>(1.1);
        expand_ratio_h = config["armor_detect"]["light"]["expand_ratio_h"].as<float>(1.1);
        l_params = { .min_ratio = config["armor_detect"]["light"]["min_ratio"].as<double>(0.1),
                     .max_ratio = config["armor_detect"]["light"]["max_ratio"].as<double>(3.0),
                     .max_angle = config["armor_detect"]["light"]["max_angle"].as<double>(40) };
        a_params = { .min_light_ratio =
                         config["armor_detect"]["armor"]["min_light_ratio"].as<double>(1),
                     .min_small_center_distance =
                         config["armor_detect"]["armor"]["min_small_center_distance"].as<double>(1),
                     .max_small_center_distance =
                         config["armor_detect"]["armor"]["max_small_center_distance"].as<double>(1),
                     .min_large_center_distance =
                         config["armor_detect"]["armor"]["min_large_center_distance"].as<double>(1),
                     .max_large_center_distance =
                         config["armor_detect"]["armor"]["max_large_center_distance"].as<double>(1),
                     .max_angle = config["armor_detect"]["armor"]["max_angle"].as<double>(1) };
    }

    detector_ = std::make_unique<ArmorDetectTrt>(
        model_path,
        params,
        expand_ratio_w,
        expand_ratio_h,
        binary_thres,
        l_params,
        a_params,
        classify_model_path,
        classify_label_path,
        classify_threshold,
        use_armor_detect_common
    );
}

ArmorDetectorTrtWrapper::~ArmorDetectorTrtWrapper() = default;

void ArmorDetectorTrtWrapper::pushInput(const CommonFrame& frame) {
    detector_->pushInput(frame);
}

void ArmorDetectorTrtWrapper::setCallback(DetectorCallback cb) {
    detector_->setCallback(std::move(cb));
}
