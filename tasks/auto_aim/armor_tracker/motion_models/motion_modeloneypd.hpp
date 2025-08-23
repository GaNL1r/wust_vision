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

namespace oneypdarmor_motion_model {

/// 运动模型类型
enum class MotionModel {
    CONSTANT_VELOCITY = 0,
    CONSTANT_ROTATION = 1,
    CONSTANT_VEL_ROT = 2 // 默认
};

// 状态维度 [x, vx, y, vy, z, vz, yaw, yaw_rate]
constexpr int X_N = 8;
// 观测维度 [yaw, pitch, distance, orientation_yaw]
constexpr int Z_N = 4;

/// 状态预测器
struct Predict {
    explicit Predict(
        double dt,
        MotionModel model = MotionModel::CONSTANT_VEL_ROT,
        double vrx = 0.0,
        double vry = 0.0,
        double vrz = 0.0
    ):
        dt(dt),
        model(model),
        vrx(vrx),
        vry(vry),
        vrz(vrz) {}

    template<typename T>
    void operator()(const T x0[X_N], T x1[X_N]) {
        for (int i = 0; i < X_N; ++i)
            x1[i] = x0[i];

        if (model == MotionModel::CONSTANT_VELOCITY || model == MotionModel::CONSTANT_VEL_ROT) {
            x1[0] += x0[1] * dt; // x += vx * dt
            x1[2] += x0[3] * dt; // y += vy * dt
            x1[4] += x0[5] * dt; // z += vz * dt
        } else {
            x1[1] = T(0);
            x1[3] = T(0);
            x1[5] = T(0);
        }
        x1[0] -= T(vrx) * T(dt);
        x1[2] -= T(vry) * T(dt);
        x1[4] -= T(vrz) * T(dt);

        if (model == MotionModel::CONSTANT_ROTATION || model == MotionModel::CONSTANT_VEL_ROT) {
            x1[6] += x0[7] * dt; // yaw += yaw_rate * dt
        } else {
            x1[7] = T(0);
        }
    }

    double dt;
    MotionModel model;
    double vrx, vry, vrz;
};

/// 观测器
struct Measure {
    template<typename T>
    void operator()(const T x[X_N], T z[Z_N]) {
        // 位置
        T armor_x = x[0];
        T armor_y = x[2];
        T armor_z = x[4];

        T xy_dist = ceres::sqrt(armor_x * armor_x + armor_y * armor_y);
        T dist = ceres::sqrt(xy_dist * xy_dist + armor_z * armor_z);

        z[0] = ceres::atan2(armor_y, armor_x); // yaw
        z[1] = ceres::atan2(armor_z, xy_dist); // pitch
        z[2] = dist; // distance
        z[3] = x[6]; // orientation_yaw
    }
};

/// 扩展卡尔曼滤波器类型
using RobotStateEKF = kalman_hybird_lib::ExtendedKalmanFilter<X_N, Z_N, Predict, Measure>;

} // namespace oneypdarmor_motion_model
