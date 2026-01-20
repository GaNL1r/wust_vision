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
using VecZ = Eigen::Matrix<double, Z_N, 1>;
using VecX = Eigen::Matrix<double, X_N, 1>;
enum class Meas : uint8_t { YPD_Y = 0, YPD_P = 1, YPD_D = 2, ORI_YAW = 3, Z_N = 4 };
enum class State : uint8_t {
    CX = 0,
    VCX = 1,
    CY = 2,
    VCY = 3,
    CZ = 4,
    VCZ = 5,
    YAW = 6,
    VYAW = 7,
    R = 8,
    L = 9,
    H = 10,
    outpost01DZ = 9,
    outpost02DZ = 10,
    X_N = 11
};
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
    void operator()(const T x0[X_N], T x1[X_N]) const {
        for (int i = 0; i < X_N; i++) {
            x1[i] = x0[i];
        }

        // v_xyz
        if (model == MotionModel::CONSTANT_VEL_ROT || model == MotionModel::CONSTANT_VELOCITY) {
            // linear velocity
            x1[(int)State::CX] += x0[(int)State::VCX] * T(dt);
            x1[(int)State::CY] += x0[(int)State::VCY] * T(dt);
            x1[(int)State::CZ] += x0[(int)State::VCZ] * T(dt);
        } else {
            // no velocity
            x1[(int)State::VCX] *= T(0.);
            x1[(int)State::VCY] *= T(0.);
            x1[(int)State::VCZ] *= T(0.);
        }

        x1[(int)State::CX] -= T(vrx) * T(dt);
        x1[(int)State::CY] -= T(vry) * T(dt);
        x1[(int)State::CZ] -= T(vrz) * T(dt);

        // v_yaw
        if (model == MotionModel::CONSTANT_VEL_ROT || model == MotionModel::CONSTANT_ROTATION) {
            // angular velocity
            x1[(int)State::YAW] += x0[(int)State::VYAW] * T(dt);
        } else {
            // no rotation
            x1[(int)State::VYAW] *= T(0.);
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
    struct MeasureCtx {
        MeasureCtx() = default;
        MeasureCtx(int id, int armor_num): armor_num(armor_num), id(id) {}
        int armor_num = 4;
        int id = 0;
    } ctx;
    Measure() = default;
    explicit Measure(MeasureCtx c): ctx(c) {}
    template<typename T>
    void operator()(const T x[X_N], T z[Z_N]) const {
        // Compute armor position
        T angle = normalize_angle_t(x[(int)State::YAW] + ctx.id * 2 * M_PI / ctx.armor_num);
        auto outpost = ctx.armor_num == 3;
        auto use_l_h = (ctx.armor_num == 4) && (ctx.id == 1 || ctx.id == 3);
        T r = (use_l_h) ? x[(int)State::R] + x[(int)State::L] : x[(int)State::R];
        T armor_x = x[(int)State::CX] - ceres::cos(angle) * r;
        T armor_y = x[(int)State::CY] - ceres::sin(angle) * r;
        T armor_z = (outpost) ? getoutpost_armor_z(x)
            : (use_l_h)       ? x[(int)State::CZ] + x[(int)State::H]
                              : x[(int)State::CZ];

        T xy_dist = ceres::sqrt(armor_x * armor_x + armor_y * armor_y);
        T dist = ceres::sqrt(xy_dist * xy_dist + armor_z * armor_z);

        // Observation model
        z[(int)Meas::YPD_Y] = ceres::atan2(armor_y, armor_x); // yaw
        z[(int)Meas::YPD_P] = ceres::atan2(armor_z, xy_dist); // pitch
        z[(int)Meas::YPD_D] = dist; // distance
        z[(int)Meas::ORI_YAW] = angle; // orientation_yaw
    }
    template<typename T>
    T getoutpost_armor_z(const T x[X_N]) const {
        return (ctx.id == 0) ? x[(int)State::CZ]
            : (ctx.id == 1)  ? x[(int)State::CZ] + x[(int)State::outpost01DZ]
            : (ctx.id == 2)  ? x[(int)State::CZ] + x[(int)State::outpost02DZ]
                             : x[(int)State::CZ];
    }
    void h(const VecX& x, VecZ& z) const {
        assert(x.size() == X_N);
        assert(z.size() == Z_N);
        operator()(x.data(), z.data());
    }
};

using RobotStateEKF = kalman_hybird_lib::ExtendedKalmanFilter<X_N, Z_N, Predict, Measure>;
using RobotStateESEKF = kalman_hybird_lib::ErrorStateEKF<X_N, Z_N, Predict, Measure>;

} // namespace ypdv2armor_motion_model
