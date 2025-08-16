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
#include "detect/rune_detect/rune_infer.hpp"
#include "onnxruntime_cxx_api.h"
#include "type/type.hpp"
#include "wust_vl/common/ThreadPool.h"
#include "wust_vl/ml_net/onnxruntime/onnxruntime_net.hpp"
class RuneDetectorOnnxRuntime {
public:
    using CallbackType = std::function<void(std::vector<rune::RuneObject>&, const CommonFrame&)>;

public:
    // Construct a new OpenVINO Detector object
    explicit RuneDetectorOnnxRuntime(
        std::string provider,
        std::string model_type,
        const std::filesystem::path& model_path,
        float conf_threshold = 0.25,
        int top_k = 128,
        float nms_threshold = 0.3
    );

    void init();

    // Push an inference request to the detector
    void pushInput(CommonFrame& frame);

    void setCallback(CallbackType callback);

    // Detect R tag using traditional method
    // Return the center of the R tag and binary roi image (for debug)
    std::tuple<cv::Point2f, cv::Mat>
    detectRTag(const cv::Mat& img, int binary_thresh, const cv::Point2f& prior, bool precise);

private:
    // Do inference and call the infer_callback_ after inference
    bool processCallback(const CommonFrame& frame);

private:
    std::string model_path_;
    float conf_threshold_;
    int top_k_;
    float nms_threshold_;
    OrtProvider provider_ = OrtProvider::CPU;
    std::vector<int> strides_;
    std::vector<GridAndStride> grid_strides_;
    CallbackType infer_callback_;
    std::unique_ptr<rune_infer::RuneInfer> rune_infer_;
    int current_id_ = 0;
    std::unique_ptr<ml_net::OnnxRuntimeNet> onnxruntime_net_;
};
