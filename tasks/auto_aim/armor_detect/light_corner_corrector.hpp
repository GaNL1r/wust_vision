// Maintained by Shenglin Qin, Chengfu Zou
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

// opencv
#include "tasks/auto_aim/type.hpp"
namespace auto_aim {

// This class is used to improve the precision of the corner points of the light
// bar. First, the PCA algorithm is used to find the symmetry axis of the light
// bar, and then along the symmetry axis to find the corner points of the light
// bar based on the gradient of brightness.
class LightCornerCorrector {
public:
    explicit LightCornerCorrector() noexcept;
    ~LightCornerCorrector();
    void correctCorners(ArmorObject& armor, const cv::Mat& gray_img) const noexcept;
    struct Impl;
    std::unique_ptr<Impl> _impl;
};
} // namespace auto_aim