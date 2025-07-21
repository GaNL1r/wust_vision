// Copyright 2023 Yunlong Feng
//
// Additional modifications and features by Chengfu Zou, 2024.
//
// Copyright (C) FYT Vision Group. All rights reserved.
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

#pragma once

// std
#include <filesystem>
#include <functional>
#include <future>
#include <memory>
#include <string>
#include <tuple>
#include <vector>
// third party
#include <Eigen/Dense>
#include <opencv2/opencv.hpp>
#include <openvino/openvino.hpp>
// project
#include "common/ThreadPool.h"
#include "type/type.hpp"
class RuneDetectorOpenvino {
public:
    using CallbackType = std::function<void(std::vector<RuneObject>&, const CommonFrame&)>;

public:
    // Construct a new OpenVINO Detector object
    explicit RuneDetectorOpenvino(
        const std::filesystem::path& model_path,
        const std::string& device_name,
        float conf_threshold = 0.25,
        int top_k = 128,
        float nms_threshold = 0.3,
        bool use_throughputmode_ = false
    );

    void init();

    // Push an inference request to the detector
    void pushInput(const CommonFrame& frame);

    void setCallback(CallbackType callback);

    // Detect R tag using traditional method
    // Return the center of the R tag and binary roi image (for debug)
    std::tuple<cv::Point2f, cv::Mat>
    detectRTag(const cv::Mat& img, int binary_thresh, const cv::Point2f& prior, bool precise);

private:
    // Do inference and call the infer_callback_ after inference
    bool processCallback(
        const cv::Mat resized_img,
        Eigen::Matrix3f transform_matrix,
        const CommonFrame& frame
    );

private:
    std::string model_path_;
    std::string device_name_;
    float conf_threshold_;
    int top_k_;
    float nms_threshold_;
    std::vector<int> strides_;
    std::vector<GridAndStride> grid_strides_;
    CallbackType infer_callback_;
    std::unique_ptr<ov::Core> ov_core_;
    std::unique_ptr<ov::CompiledModel> compiled_model_;
    std::shared_ptr<ov::Model> model_;
    ov::preprocess::PrePostProcessor* ppp_;
    bool use_throughputmode_ = false;
};
