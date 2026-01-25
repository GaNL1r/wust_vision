// Created by Labor 2023.8.25
// Maintained by Labor, Chengfu Zou
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
#include "tasks/auto_buff/type.hpp"
namespace auto_buff {
class BaSolver {
public:
    using Ptr = std::unique_ptr<BaSolver>;
    BaSolver(const YAML::Node& config, const cv::Mat& camera_matrix);
    static Ptr create(const YAML::Node& config, const cv::Mat& camera_matrix) {
        return std::make_unique<BaSolver>(config, camera_matrix);
    }
    ~BaSolver();
    Eigen::Matrix3d solveBa_R(
        const auto_buff::RuneFan::Simple& rune_fan,
        const Eigen::Vector3d& t_camera_armor,
        const Eigen::Matrix3d& R_camera_armor,
        const Eigen::Matrix3d& R_imu_camera
    ) const noexcept;

private:
    struct Impl;
    std::unique_ptr<Impl> _impl;
};

} // namespace auto_buff
