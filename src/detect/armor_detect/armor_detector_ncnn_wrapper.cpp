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
#include "detect/armor_detect/armor_detector_ncnn_wrapper.hpp"

ArmorDetectorNCNNWrapper::ArmorDetectorNCNNWrapper(const YAML::Node& config) {
    auto classify_model_path = config["armor_detect"]["classify_model_path"].as<std::string>();
    auto classify_label_path = config["armor_detect"]["classify_label_path"].as<std::string>();
    const std::string model_path_param =
        config["armor_detect"]["model"]["model_path_param"].as<std::string>();
    const std::string model_path_bin =
        config["armor_detect"]["model"]["model_path_bin"].as<std::string>();
    bool use_gpu = config["armor_detect"]["model"]["use_gpu"].as<bool>();
    int cpu_threads = config["armor_detect"]["model"]["cpu_threads"].as<int>();
    float conf_threshold = config["armor_detect"]["model"]["conf_threshold"].as<float>();
    int top_k = config["armor_detect"]["model"]["top_k"].as<int>();
    float nms_threshold = config["armor_detect"]["model"]["nms_threshold"].as<float>();
    float expand_ratio_w = config["armor_detect"]["light"]["expand_ratio_w"].as<float>();
    float expand_ratio_h = config["armor_detect"]["light"]["expand_ratio_h"].as<float>();
    int binary_thres = config["armor_detect"]["light"]["binary_thres"].as<int>();
    WUST_INFO("armor_detector") << "model_path_param: " << model_path_param
                                << "model_path_bin: " << model_path_param;
    LightParams l_params = { .min_ratio = config["armor_detect"]["light"]["min_ratio"].as<double>(),
                             .max_ratio = config["armor_detect"]["light"]["max_ratio"].as<double>(),
                             .max_angle =
                                 config["armor_detect"]["light"]["max_angle"].as<double>() };
    ArmorParams a_params = {
        .min_light_ratio = config["armor_detect"]["armor"]["min_light_ratio"].as<double>(),
        .min_small_center_distance =
            config["armor_detect"]["armor"]["min_small_center_distance"].as<double>(),
        .max_small_center_distance =
            config["armor_detect"]["armor"]["max_small_center_distance"].as<double>(),
        .min_large_center_distance =
            config["armor_detect"]["armor"]["min_large_center_distance"].as<double>(),
        .max_large_center_distance =
            config["armor_detect"]["armor"]["max_large_center_distance"].as<double>(),
        .max_angle = config["armor_detect"]["armor"]["max_angle"].as<double>()
    };

    detector_ = std::make_unique<ArmorDetectNCNN>(
        model_path_param,
        model_path_bin,
        classify_model_path,
        classify_label_path,
        l_params,
        a_params,
        conf_threshold,
        top_k,
        nms_threshold,
        expand_ratio_w,
        expand_ratio_h,
        binary_thres,
        use_gpu,
        cpu_threads
    );
}

ArmorDetectorNCNNWrapper::~ArmorDetectorNCNNWrapper() = default;

void ArmorDetectorNCNNWrapper::pushInput(
    const cv::Mat& rgb_img,
    std::chrono::steady_clock::time_point timestamp,
    Eigen::Matrix4d T_camera_to_odom
) {
    detector_->pushInput(rgb_img, timestamp, T_camera_to_odom);
}

void ArmorDetectorNCNNWrapper::setCallback(DetectorCallback cb) {
    detector_->setCallback(std::move(cb));
}