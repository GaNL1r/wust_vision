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
#pragma once
#include "type/type.hpp"
#include <opencv2/core/mat.hpp>

class ArmorDetectorBase {
public:
    virtual ~ArmorDetectorBase() = default;

    virtual void pushInput(
        const cv::Mat& rgb_img,
        std::chrono::steady_clock::time_point timestamp,
        Eigen::Matrix4d T_camera_to_odom
    ) = 0;

    using DetectorCallback = std::function<void(
        const std::vector<ArmorObject>&,
        std::chrono::steady_clock::time_point,
        const cv::Mat&,
        Eigen::Matrix4d
    )>;

    virtual void setCallback(DetectorCallback cb) = 0;
};
