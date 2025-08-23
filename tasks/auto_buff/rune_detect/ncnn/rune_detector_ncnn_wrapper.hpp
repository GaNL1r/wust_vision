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
#include "tasks/auto_buff/rune_detect/ncnn/rune_detector_ncnn.hpp"
#include "tasks/auto_buff/rune_detect/rune_detector_base.hpp"
#include <yaml-cpp/yaml.h>

class RuneDetectorNCNNWrapper: public RuneDetectorBase {
public:
    RuneDetectorNCNNWrapper(const YAML::Node& config);
    ~RuneDetectorNCNNWrapper() override;

    void pushInput(CommonFrame& frame) override;

    void setCallback(CallbackType cb) override;

    std::tuple<cv::Point2f, cv::Mat>
    detectRTag(const cv::Mat& img, int binary_thresh, const cv::Point2f& prior, bool precise)
        override;

private:
    std::unique_ptr<RuneDetectorNCNN> rune_detector_;
};
