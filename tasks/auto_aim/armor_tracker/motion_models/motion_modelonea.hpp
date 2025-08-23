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
#include "KalmanHyLib/kalman_hybird_lib.hpp"
#include <ceres/ceres.h>

namespace onearmor_motion_model {

/// @brief 支持的运动模型类型
enum class MotionModel {
    CONSTANT_VELOCITY = 0, // 仅匀速运动（无角速度）
    CONSTANT_ROTATION = 1, // 仅匀角速度（无线速度）
    CONSTANT_VEL_ROT = 2 // 匀速 + 匀角速度（默认）
};

/// 状态维度 [x, vx, y, vy, z, vz, yaw, yaw_rate]
constexpr int X_N = 8;
/// 观测维度 [x, y, z, yaw]
constexpr int Z_N = 4;

/// 状态预测模型
struct Predict {
    explicit Predict(double dt, MotionModel model = MotionModel::CONSTANT_VEL_ROT):
        dt(dt),
        model(model) {}

    template<typename T>
    void operator()(const T x0[X_N], T x1[X_N]) {
        // 默认拷贝状态
        for (int i = 0; i < X_N; ++i) {
            x1[i] = x0[i];
        }

        if (model == MotionModel::CONSTANT_VELOCITY || model == MotionModel::CONSTANT_VEL_ROT) {
            // 匀速线性预测
            x1[0] += x0[1] * dt; // x += vx * dt
            x1[2] += x0[3] * dt; // y += vy * dt
            x1[4] += x0[5] * dt; // z += vz * dt
        } else {
            // 否则清除速度
            x1[1] = T(0);
            x1[3] = T(0);
            x1[5] = T(0);
        }

        if (model == MotionModel::CONSTANT_ROTATION || model == MotionModel::CONSTANT_VEL_ROT) {
            // 匀角速度预测
            x1[6] += x0[7] * dt; // yaw += yaw_rate * dt
        } else {
            x1[7] = T(0);
        }
    }

    double dt;
    MotionModel model;
};

/// 观测模型
struct Measure {
    template<typename T>
    void operator()(const T x[X_N], T z[Z_N]) {
        z[0] = x[0]; // x
        z[1] = x[2]; // y
        z[2] = x[4]; // z
        z[3] = x[6]; // yaw
    }
};

/// EKF 类型定义
using RobotStateEKF = kalman_hybird_lib::ExtendedKalmanFilter<X_N, Z_N, Predict, Measure>;

} // namespace onearmor_motion_model
