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

#include <ceres/ceres.h>

#include "tracker/extended_kalman_filter.hpp"

namespace ypdrune_motion_model {

constexpr int X_N = 4, Z_N = 4;

struct Predict {
    template<typename T>
    void operator()(const T x0[X_N], T x1[X_N]) {
        for (int i = 0; i < X_N; ++i) {
            x1[i] = x0[i];
        }
    }
};

struct Measure {
    template<typename T>
    void operator()(const T x[Z_N], T z[Z_N]) {
        T xy_dist = ceres::sqrt(x[0] * x[0] + x[1] * x[1]);
        T dist = ceres::sqrt(xy_dist * xy_dist + x[2] * x[2]);

        // Observation model
        z[0] = ceres::atan2(x[1], x[0]); // yaw
        z[1] = ceres::atan2(x[3], xy_dist); // pitch
        z[2] = dist; // distance
        z[3] = x[3]; // orientation_yaw
    }
};

using RuneCenterEKF = ExtendedKalmanFilter<X_N, Z_N, Predict, Measure>;
} // namespace ypdrune_motion_model
