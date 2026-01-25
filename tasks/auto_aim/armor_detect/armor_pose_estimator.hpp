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
#include "tasks/auto_aim/type.hpp"
namespace wust_vision {
namespace auto_aim {
    class ArmorPoseEstimator {
    public:
        using Ptr = std::unique_ptr<ArmorPoseEstimator>;
        explicit ArmorPoseEstimator(
            const YAML::Node& config,
            std::pair<cv::Mat, cv::Mat> camera_info
        );
        static Ptr create(const YAML::Node& config, std::pair<cv::Mat, cv::Mat> camera_info) {
            return std::make_unique<ArmorPoseEstimator>(config, camera_info);
        }
        ~ArmorPoseEstimator();
        std::vector<Armor> extractArmorPoses(
            const std::vector<ArmorObject>& armors,
            Eigen::Matrix4d T_camera_to_odom,
            const cv::Mat& camera_intrinsic,
            const cv::Mat& camera_distortion
        ) const noexcept;
        struct Impl;
        std::unique_ptr<Impl> _impl;
    };
} // namespace auto_aim
} // namespace wust_vision