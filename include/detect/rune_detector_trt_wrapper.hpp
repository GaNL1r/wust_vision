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
#include "detect/rune_detector_base.hpp"
#include "detect/rune_detector_trt.hpp"
#include <yaml-cpp/yaml.h>

class RuneDetectorTrtWrapper: public RuneDetectorBase {
public:
    RuneDetectorTrtWrapper(const YAML::Node& config);
    ~RuneDetectorTrtWrapper() override;

    void pushInput(
        const cv::Mat& rgb_img,
        std::chrono::steady_clock::time_point timestamp,
        Eigen::Matrix4d T_camera_to_odom
    ) override;

    void setCallback(CallbackType cb) override;

    std::tuple<cv::Point2f, cv::Mat>
    detectRTag(const cv::Mat& img, int binary_thresh, const cv::Point2f& prior, bool precise)
        override;

private:
    std::unique_ptr<RuneDetectorTrt> rune_detector_;
};
