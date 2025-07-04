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
#include "tracker/math/extended_kalman_filter.hpp"
#include <ceres/ceres.h>

namespace onecaarmor_motion_model {

/// @brief 支持的运动模型类型
enum class MotionModel {
    CONSTANT_ACCEL_ROT = 0 // 匀加速 + 匀角速度（默认）
};

/// 状态维度 [x, vx, ax, y, vy, ay, z, vz, az, yaw, yaw_rate]
constexpr int X_N = 11;
/// 观测维度 [x, y, z, yaw]
constexpr int Z_N = 4;

struct Predict {
    explicit Predict(double dt, MotionModel model = MotionModel::CONSTANT_ACCEL_ROT):
        dt(dt),
        model(model) {}

    template<typename T>
    void operator()(const T x0[X_N], T x1[X_N]) {
        for (int i = 0; i < X_N; ++i) {
            x1[i] = x0[i];
        }

        // 匀加速线性运动（x, y 方向）
        x1[0] += x0[1] * dt + T(0.5) * x0[2] * dt * dt; // x
        x1[1] += x0[2] * dt; // vx

        x1[3] += x0[4] * dt + T(0.5) * x0[5] * dt * dt; // y
        x1[4] += x0[5] * dt; // vy

        // z 匀速运动
        x1[6] += x0[7] * dt; // z
        // x1[7] unchanged (vz)

        // 匀角速度
        x1[8] += x0[9] * dt; // yaw
    }

    double dt;
    MotionModel model;
};

/// 观测模型
struct Measure {
    template<typename T>
    void operator()(const T x[X_N], T z[Z_N]) {
        z[0] = x[0]; // x
        z[1] = x[3]; // y
        z[2] = x[6]; // z
        z[3] = x[9]; // yaw
    }
};

/// EKF 类型定义
using RobotStateEKF = ExtendedKalmanFilter<X_N, Z_N, Predict, Measure>;

} // namespace onecaarmor_motion_model
