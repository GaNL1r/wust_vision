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
using VecZ = Eigen::Matrix<double, Z_N, 1>;
using VecX = Eigen::Matrix<double, X_N, 1>;
enum class Meas : uint8_t { YPD_Y = 0, YPD_P = 1, YPD_D = 2, ORI_YAW = 3, ORI_ROLL = 4, Z_N = 5 };
enum class State : uint8_t { CX = 0, CY = 1, CZ = 2, YAW = 3, ROLL = 4, VROLL = 5, X_N = 6 };
struct Predict {
    Predict() = default;
    explicit Predict(double dt): dt(dt) {}
    template<typename T>
    void operator()(const T x0[X_N], T x1[X_N]) const {
        for (int i = 0; i < X_N; ++i) {
            x1[i] = x0[i];
        }
        x1[(int)State::ROLL] += x0[(int)State::VROLL] * dt;
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
        T xy_dist = ceres::sqrt(
            x[(int)State::CX] * x[(int)State::CX] + x[(int)State::CY] * x[(int)State::CY]
        );
        T dist = ceres::sqrt(xy_dist * xy_dist + x[(int)State::CZ] * x[(int)State::CZ]);

        // Observation model
        z[(int)Meas::YPD_Y] = ceres::atan2(x[1], x[0]); // yaw
        z[(int)Meas::YPD_P] = ceres::atan2(x[2], xy_dist); // pitch
        z[(int)Meas::YPD_D] = dist; // distance
        z[(int)Meas::ORI_YAW] = x[(int)State::YAW]; // orientation_yaw
        z[(int)Meas::ORI_ROLL] = normalize_angle_t(x[(int)State::ROLL] + id * 2 * M_PI / 5); // roll
    }
    void h(const VecX& x, VecZ& z) const {
        assert(x.size() == X_N);
        assert(z.size() == Z_N);
        operator()(x.data(), z.data());
    }
    int id = 0;
};

using RuneESKF = kalman_hybird_lib::ErrorStateEKF<X_N, Z_N, Predict, Measure>;
} // namespace ypdrune_motion_model