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
#include "type/type.hpp"

class RuneDetectorNCNN {
public:
    using CallbackType = std::function<void(
        std::vector<RuneObject>&,
        std::chrono::steady_clock::time_point,
        const cv::Mat&,
        Eigen::Matrix4d
    )>;

public:
    // Construct a new OpenNCNN Detector object
    explicit RuneDetectorNCNN(
        const std::string& model_path_param_,
        const std::string& model_path_bin_,
        float conf_threshold = 0.25,
        int top_k = 128,
        float nms_threshold = 0.3,
        bool use_gpu = false,
        int cpu_threads = 1
    );
    ~RuneDetectorNCNN();

    void init();

    // Push an inference request to the detector
    void pushInput(
        const cv::Mat& rgb_img,
        std::chrono::steady_clock::time_point timestamp,
        Eigen::Matrix4d T_camera_to_odom
    );

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
        std::chrono::steady_clock::time_point timestamp,
        const cv::Mat& src_img,
        Eigen::Matrix4d T_camera_to_odom
    );

private:
    ncnn::Net net_;
    ncnn::Option opt_;
    std::string model_path_param_;
    std::string model_path_bin_;
    float conf_threshold_;
    int top_k_;
    float nms_threshold_;
    std::mutex mtx_;
    std::vector<int> strides_;
    std::vector<GridAndStride> grid_strides_;
    std::string input_name_;
    std::string output_name_;

    CallbackType infer_callback_;
    std::unique_ptr<ThreadPool> thread_pool_;

    int cpu_threads;
    bool use_gpu;
};