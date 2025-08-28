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

// ceres
#include <ceres/ceres.h>
// project
#include "KalmanHyLib/kalman_hybird_lib.hpp"
namespace ypdv2armor_motion_model {

enum class MotionModel {
    CONSTANT_VELOCITY = 0, // Constant velocity
    CONSTANT_ROTATION = 1, // Constant rotation velocity
    CONSTANT_VEL_ROT = 2 // Constant velocity and rotation velocity
};

// X_N: state dimension, Z_N: measurement dimension
constexpr int X_N = 11, Z_N = 4;

struct Predict {
    Predict() = default;
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
        for (int i = 0; i < X_N; i++) {
            x1[i] = x0[i];
        }
        x1[0] -= T(vrx) * T(dt);
        x1[2] -= T(vry) * T(dt);
        x1[4] -= T(vrz) * T(dt);

        // v_xyz
        if (model == MotionModel::CONSTANT_VEL_ROT || model == MotionModel::CONSTANT_VELOCITY) {
            // linear velocity
            x1[0] += x0[1] * dt;
            x1[2] += x0[3] * dt;
            x1[4] += x0[5] * dt;
        } else {
            // no velocity
            x1[1] *= T(0.);
            x1[3] *= T(0.);
            x1[5] *= T(0.);
        }

        // v_yaw
        if (model == MotionModel::CONSTANT_VEL_ROT || model == MotionModel::CONSTANT_ROTATION) {
            // angular velocity
            x1[6] += x0[7] * dt;
        } else {
            // no rotation
            x1[7] *= T(0.);
        }
    }

    double dt;
    MotionModel model;
    double vrx, vry, vrz;
};
template<typename T>
T normalize_angle_t(T angle) {
    T two_pi = T(2.0 * M_PI);
    return angle - two_pi * floor((angle + T(M_PI)) / two_pi);
}

struct Measure {
    Measure() = default;
    explicit Measure(int id, int armor_num): id(id), armor_num(armor_num) {}
    template<typename T>
    void operator()(const T x[Z_N], T z[Z_N]) {
        // Compute armor position
        T angle = normalize_angle_t(x[6] + id * 2 * M_PI / armor_num);
        auto use_l_h = (armor_num == 4) && (id == 1 || id == 3);
        T r = (use_l_h) ? x[8] + x[9] : x[8];
        T armor_x = x[0] - ceres::cos(angle) * r;
        T armor_y = x[2] - ceres::sin(angle) * r;
        T armor_z = (use_l_h) ? x[4] + x[10] : x[4];

        T xy_dist = ceres::sqrt(armor_x * armor_x + armor_y * armor_y);
        T dist = ceres::sqrt(xy_dist * xy_dist + armor_z * armor_z);

        // Observation model
        z[0] = ceres::atan2(armor_y, armor_x); // yaw
        z[1] = ceres::atan2(armor_z, xy_dist); // pitch
        z[2] = dist; // distance
        z[3] = normalize_angle_t(x[6] + id * 2 * M_PI / armor_num); // orientation_yaw
    }
    int armor_num = 4;
    int id = 0;
};

using RobotStateEKF = kalman_hybird_lib::ExtendedKalmanFilter<X_N, Z_N, Predict, Measure>;
using RobotStateESEKF = kalman_hybird_lib::ErrorStateEKF<X_N, Z_N, Predict, Measure>;

} // namespace ypdv2armor_motion_model
