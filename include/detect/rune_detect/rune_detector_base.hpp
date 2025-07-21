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
#include "type/type.hpp"
#include <opencv2/core/mat.hpp>

class RuneDetectorBase {
public:
    virtual ~RuneDetectorBase() = default;

    using CallbackType = std::function<void(std::vector<rune::RuneObject>&, const CommonFrame&)>;

    virtual void pushInput(const CommonFrame& frame) = 0;

    virtual void setCallback(CallbackType cb) = 0;
    virtual std::tuple<cv::Point2f, cv::Mat>
    detectRTag(const cv::Mat& img, int binary_thresh, const cv::Point2f& prior, bool precise) = 0;
};
