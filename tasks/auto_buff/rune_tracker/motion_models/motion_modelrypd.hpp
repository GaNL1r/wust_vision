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

#include "KalmanHyLib/kalman_hybird_lib.hpp"

namespace ypdrune_motion_model {

constexpr int X_N = 6, Z_N = 5;

struct Predict {
    Predict() = default;
    explicit Predict(double dt): dt(dt) {}
    template<typename T>
    void operator()(const T x0[X_N], T x1[X_N]) const {
        for (int i = 0; i < X_N; ++i) {
            x1[i] = x0[i];
        }
        x1[4] += x0[5] * dt;
    }
    double dt;
};
template<typename T>
T normalize_angle_t(T angle) {
    T two_pi = T(2.0 * M_PI);
    return angle - two_pi * floor((angle + T(M_PI)) / two_pi);
}
struct Measure {
    Measure() = default;
    explicit Measure(int id): id(id) {}
    template<typename T>
    void operator()(const T x[X_N], T z[Z_N]) const {
        T xy_dist = ceres::sqrt(x[0] * x[0] + x[1] * x[1]);
        T dist = ceres::sqrt(xy_dist * xy_dist + x[2] * x[2]);

        // Observation model
        z[0] = ceres::atan2(x[1], x[0]); // yaw
        z[1] = ceres::atan2(x[2], xy_dist); // pitch
        z[2] = dist; // distance
        z[3] = x[3]; // orientation_yaw
        z[4] = normalize_angle_t(x[4] + id * 2 * M_PI / 5); // roll
    }
    int id = 0;
};

using RuneESKF = kalman_hybird_lib::ErrorStateEKF<X_N, Z_N, Predict, Measure>;
} // namespace ypdrune_motion_model