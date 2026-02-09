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
namespace wust_vision {
namespace auto_buff {
    class RuneWhere {
    public:
        using Ptr = std::unique_ptr<RuneWhere>;
        RuneWhere(const YAML::Node& config, const std::pair<cv::Mat, cv::Mat>& camera_info);
        static Ptr
        create(const YAML::Node& config, const std::pair<cv::Mat, cv::Mat>& camera_info) {
            return std::make_unique<RuneWhere>(config, camera_info);
        }
        ~RuneWhere();
        auto_buff::RuneFan where(auto_buff::RuneFan f, Eigen::Matrix4d T_camera_to_odom);

    private:
        struct Impl;
        std::unique_ptr<Impl> _impl;
    };

} // namespace auto_buff
} // namespace wust_vision
