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
#include "tasks/auto_aim/armor_detect/tensorrt/armor_detector_tensorrt_wrapper.hpp"
#include "tasks/utils.hpp"

ArmorDetectorTrtWrapper::ArmorDetectorTrtWrapper(
    const YAML::Node& config,
    bool use_armor_detect_common
) {
    auto classify_model_path =
        utils::expandEnv(config["armor_detect"]["classify"]["model_path"].as<std::string>());
    auto classify_label_path =
        utils::expandEnv(config["armor_detect"]["classify"]["label_path"].as<std::string>());
    double classify_threshold = config["armor_detect"]["classify"]["threshold"].as<double>();
    std::string model_path =
        utils::expandEnv(config["armor_detect"]["model"]["model_path"].as<std::string>());
    ArmorDetectTrt::Params params;
    params.conf_threshold = config["armor_detect"]["model"]["conf_threshold"].as<float>();
    params.nms_threshold = config["armor_detect"]["model"]["nms_threshold"].as<float>();
    params.top_k = config["armor_detect"]["model"]["top_k"].as<int>();
    params.device_id = config["armor_detect"]["model"]["device_id"].as<int>();
    params.max_infer_running = config["armor_detect"]["model"]["max_infer_running"].as<int>();
    params.min_free_mem_ratio = config["armor_detect"]["model"]["min_free_mem_ratio"].as<double>();
    params.use_cuda_pre = config["armor_detect"]["model"]["use_cuda_pre"].as<bool>();
    params.use_cuda_post = config["armor_detect"]["model"]["use_cuda_post"].as<bool>();
    params.log_time = config["armor_detect"]["model"]["log_time"].as<bool>();
    std::string model_type = config["armor_detect"]["model"]["model_type"].as<std::string>();
    WUST_INFO("armor_detector") << "model_path: " << model_path;
    int binary_thres;
    float expand_ratio_w, expand_ratio_h;
    double max_pts_error;
    armor::LightParams l_params;
    armor::ArmorParams a_params;
    if (use_armor_detect_common) {
        binary_thres = config["armor_detect"]["light"]["binary_thres"].as<int>(85);
        expand_ratio_w = config["armor_detect"]["light"]["expand_ratio_w"].as<float>(1.1);
        expand_ratio_h = config["armor_detect"]["light"]["expand_ratio_h"].as<float>(1.1);
        max_pts_error = config["armor_detect"]["light"]["max_pts_error"].as<double>(20.0);
        l_params = { .min_ratio = config["armor_detect"]["light"]["min_ratio"].as<double>(0.1),
                     .max_ratio = config["armor_detect"]["light"]["max_ratio"].as<double>(3.0),
                     .max_angle = config["armor_detect"]["light"]["max_angle"].as<double>(40),
                     .max_angle_diff =
                         config["armor_detect"]["light"]["max_angle_diff"].as<double>(20),
                     .color_diff_thresh =
                         config["armor_detect"]["light"]["color_diff_thresh"].as<int>() };
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
    ArmorDetectCommonParams armor_detect_common_params = {
        .binary_thres = binary_thres,
        .classifier_threshold = classify_threshold,
        .classify_label_path = classify_label_path,
        .classify_model_path = classify_model_path,
        .expand_ratio_h = expand_ratio_h,
        .expand_ratio_w = expand_ratio_w,
        .armor_params = a_params,
        .light_params = l_params,
        .max_pts_error = max_pts_error
    };

    detector_ = std::make_unique<ArmorDetectTrt>(
        model_type,
        model_path,
        params,
        armor_detect_common_params,
        use_armor_detect_common
    );
}

ArmorDetectorTrtWrapper::~ArmorDetectorTrtWrapper() = default;

void ArmorDetectorTrtWrapper::pushInput(CommonFrame& frame) {
    detector_->pushInput(frame);
}

void ArmorDetectorTrtWrapper::setCallback(DetectorCallback cb) {
    detector_->setCallback(std::move(cb));
}
