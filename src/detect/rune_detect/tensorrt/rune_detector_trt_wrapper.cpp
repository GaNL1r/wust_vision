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
#include "detect/rune_detect/tensorrt/rune_detector_trt_wrapper.hpp"

RuneDetectorTrtWrapper::RuneDetectorTrtWrapper(const YAML::Node& config) {
    std::string model_path = config["rune_detector"]["model"].as<std::string>();

    WUST_INFO("rune_detector") << "model : " << model_path;

    RuneDetectorTrt::Params params;
    params.conf_threshold = config["rune_detector"]["confidence_threshold"].as<float>(0.50);
    params.nms_threshold = config["rune_detector"]["nms_threshold"].as<float>(0.3);
    params.top_k = config["rune_detector"]["top_k"].as<int>(128);
    params.device_id = config["rune_detector"]["device_id"].as<int>();
    params.max_infer_running = config["rune_detector"]["max_infer_running"].as<int>();
    params.min_free_mem_ratio = config["rune_detector"]["min_free_mem_ratio"].as<double>();
    params.use_cuda_pre = config["rune_detector"]["use_cuda_pre"].as<bool>();
    params.use_cuda_post = config["rune_detector"]["use_cuda_post"].as<bool>();
    params.log_time = config["rune_detector"]["log_time"].as<bool>();
    std::string model_type = config["rune_detector"]["model_type"].as<std::string>();
    WUST_INFO("rune_detector") << "model_path : " << model_path;
    rune_detector_ = std::make_unique<RuneDetectorTrt>(model_type, model_path, params);
}

RuneDetectorTrtWrapper::~RuneDetectorTrtWrapper() = default;

void RuneDetectorTrtWrapper::pushInput(const CommonFrame& frame) {
    rune_detector_->pushInput(frame);
}

void RuneDetectorTrtWrapper::setCallback(CallbackType cb) {
    rune_detector_->setCallback(std::move(cb));
}

std::tuple<cv::Point2f, cv::Mat> RuneDetectorTrtWrapper::detectRTag(
    const cv::Mat& img,
    int binary_thresh,
    const cv::Point2f& prior,
    bool precise
) {
    return rune_detector_->detectRTag(img, binary_thresh, prior, precise);
}
