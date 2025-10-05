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

namespace bigrune_motion_model {

constexpr int X_N = 8, Z_N = 5;

// struct Predict {
//     Predict() = default;
//     explicit Predict(double dt): dt(dt) {}

//     template<typename T>
//     void operator()(const T x0[X_N], T x1[X_N]) const {
//         // 拷贝
//         for (int i = 0; i < X_N; ++i) {
//             x1[i] = x0[i];
//         }

//         // roll = roll + v_roll * dt
//         x1[4] += x0[5] * dt;

//         // 从状态量中取 a, w
//         const T& a = x0[6];
//         const T& w = x0[7];
//         T b = T(2.090) - a;

//         // v_roll 由拟合函数驱动
//         x1[5] = a * ceres::sin(w * T(current_time)) + b;

//         // 让 a, w 允许自适应变化：保持常值，但噪声由协方差注入
//         x1[6] = x0[6];  // + 过程噪声
//         x1[7] = x0[7];  // + 过程噪声
//     }

//     double dt;
//     double current_time = 0.0; // 由外部 predict() 调用时更新
// };
struct Predict {
    Predict() = default;
    explicit Predict(double dt): dt(dt) {}

    template<typename T>
    void operator()(const T x0[X_N], T x1[X_N]) const {
        for (int i = 0; i < X_N; ++i)
            x1[i] = x0[i];

        // 拷贝 a, w
        const T& a = x0[6];
        const T& w = x0[7];
        T b = T(2.090) - a;

        // 用 roll 累积代替 t
        T roll0 = T(x0[4] - x0[5] * dt); // 上一次 roll（近似）
        T phi = x0[4] - roll0; // roll 累积
        x1[5] = a * ceres::sin(w * phi / b) + b; // v_roll

        // roll 积分
        x1[4] += x1[5] * dt;

        // a, w 自适应，由过程噪声驱动
        x1[6] = x0[6];
        x1[7] = x0[7];
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

        z[0] = ceres::atan2(x[1], x[0]); // yaw
        z[1] = ceres::atan2(x[2], xy_dist); // pitch
        z[2] = dist; // distance
        z[3] = x[3]; // orientation_yaw
        z[4] = normalize_angle_t(x[4] + id * 2 * M_PI / 5); // roll
    }
    int id = 0;
};

using RuneESKF = kalman_hybird_lib::ErrorStateEKF<X_N, Z_N, Predict, Measure>;
} // namespace bigrune_motion_model
