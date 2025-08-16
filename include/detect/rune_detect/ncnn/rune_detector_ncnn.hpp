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
#include <ncnn/net.h>
#include <opencv2/opencv.hpp>
// project
#include "common/ThreadPool.h"
#include "detect/rune_detect/rune_infer.hpp"
#include "ml_net/ncnn/ncnn_net.hpp"
#include "type/type.hpp"
class RuneDetectorNCNN {
public:
    using CallbackType = std::function<void(std::vector<rune::RuneObject>&, const CommonFrame&)>;

public:
    // Construct a new OpenNCNN Detector object
    explicit RuneDetectorNCNN(
        std::string model_type,
        std::string input_name_,
        std::string output_name_,
        const std::string& model_path_param_,
        const std::string& model_path_bin_,
        float conf_threshold = 0.25,
        int top_k = 128,
        float nms_threshold = 0.3,
        bool use_gpu = false,
        int cpu_threads = 1,
        bool use_lightmode = true,
        int device_id = 0
    );
    ~RuneDetectorNCNN();

    void init(int device_id);

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
    std::string model_path_param_;
    std::string model_path_bin_;
    float conf_threshold_;
    int top_k_;
    float nms_threshold_;
    std::vector<int> strides_;
    std::vector<GridAndStride> grid_strides_;
    std::string input_name_;
    std::string output_name_;
    CallbackType infer_callback_;
    int cpu_threads_;
    bool use_gpu_;
    bool use_lightmode_ = true;
    std::unique_ptr<rune_infer::RuneInfer> rune_infer_;
    int current_id_ = 0;
    std::unique_ptr<ml_net::NCNNNet> ncnn_net_;
};