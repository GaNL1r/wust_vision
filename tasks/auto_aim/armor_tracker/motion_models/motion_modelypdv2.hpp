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
enum class MeasureID : uint8_t { YPD_Y = 0, YPD_P = 1, YPD_D = 2, ORI_YAW = 3, Z_N = 4 };
enum class StateID : uint8_t {
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
struct State {
    VecX x;
    [[nodiscard]] inline double cx() const noexcept {
        return x((int)StateID::CX);
    }
    [[nodiscard]] inline double cy() const noexcept {
        return x((int)StateID::CY);
    }
    [[nodiscard]] inline double cz() const noexcept {
        return x((int)StateID::CZ);
    }
    [[nodiscard]] inline Eigen::Vector3d pos() const noexcept {
        return Eigen::Vector3d(cx(), cy(), cz());
    }
    [[nodiscard]] inline double vcx() const noexcept {
        return x((int)StateID::VCX);
    }
    [[nodiscard]] inline double vcy() const noexcept {
        return x((int)StateID::VCY);
    }
    [[nodiscard]] inline double vcz() const noexcept {
        return x((int)StateID::VCZ);
    }
    [[nodiscard]] inline Eigen::Vector3d vel() const noexcept {
        return Eigen::Vector3d(vcx(), vcy(), vcz());
    }
    [[nodiscard]] inline double vyaw() const noexcept {
        return x((int)StateID::VYAW);
    }
    [[nodiscard]] inline double yaw() const noexcept {
        return x((int)StateID::YAW);
    }
    [[nodiscard]] inline double r() const noexcept {
        return x((int)StateID::R);
    }
    [[nodiscard]] inline double l() const noexcept {
        return x((int)StateID::L);
    }
    [[nodiscard]] inline double h() const noexcept {
        return x((int)StateID::H);
    }
    [[nodiscard]] inline double outpost01DZ() const noexcept {
        return x((int)StateID::outpost01DZ);
    }
    [[nodiscard]] inline double outpost02DZ() const noexcept {
        return x((int)StateID::outpost02DZ);
    }
};
struct Predict {
    Predict() = default;
    explicit Predict(double dt, MotionModel model = MotionModel::CONSTANT_VEL_ROT):
        dt(dt),
        model(model) {}

    template<typename T>
    void operator()(const T x0[X_N], T x1[X_N]) const {
        for (int i = 0; i < X_N; i++) {
            x1[i] = x0[i];
        }

        // v_xyz
        if (model == MotionModel::CONSTANT_VEL_ROT || model == MotionModel::CONSTANT_VELOCITY) {
            // linear velocity
            x1[(int)StateID::CX] += x0[(int)StateID::VCX] * T(dt);
            x1[(int)StateID::CY] += x0[(int)StateID::VCY] * T(dt);
            x1[(int)StateID::CZ] += x0[(int)StateID::VCZ] * T(dt);
        } else {
            // no velocity
            x1[(int)StateID::VCX] *= T(0.);
            x1[(int)StateID::VCY] *= T(0.);
            x1[(int)StateID::VCZ] *= T(0.);
        }

        // v_yaw
        if (model == MotionModel::CONSTANT_VEL_ROT || model == MotionModel::CONSTANT_ROTATION) {
            // angular velocity
            x1[(int)StateID::YAW] += x0[(int)StateID::VYAW] * T(dt);
        } else {
            // no rotation
            x1[(int)StateID::VYAW] *= T(0.);
        }
    }
    void f(const VecX& x0, VecX& x1) const {
        assert(x0.size() == X_N);
        assert(x1.size() == X_N);
        operator()(x0.data(), x1.data());
    }

    double dt;
    MotionModel model;
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
        auto [armor_x, armor_y, armor_z, angle] = h_armor_xyza(x);
        T xy_dist = ceres::sqrt(armor_x * armor_x + armor_y * armor_y);
        T dist = ceres::sqrt(xy_dist * xy_dist + armor_z * armor_z);
        // Observation model
        z[(int)MeasureID::YPD_Y] = ceres::atan2(armor_y, armor_x); // yaw
        z[(int)MeasureID::YPD_P] = ceres::atan2(armor_z, xy_dist); // pitch
        z[(int)MeasureID::YPD_D] = dist; // distance
        z[(int)MeasureID::ORI_YAW] = angle; // orientation_yaw
    }
    template<typename T>
    T get_angle(const T x[X_N]) const {
        return normalize_angle_t(x[(int)StateID::YAW] + ctx.id * 2 * M_PI / ctx.armor_num);
    }
    template<typename T>
    std::tuple<T, T, T, T> h_armor_xyza(const T x[X_N]) const {
        T angle = get_angle(x);
        auto outpost = ctx.armor_num == 3;
        auto use_l_h = (ctx.armor_num == 4) && (ctx.id == 1 || ctx.id == 3);
        T r = (use_l_h) ? x[(int)StateID::R] + x[(int)StateID::L] : x[(int)StateID::R];
        T armor_x = x[(int)StateID::CX] - ceres::cos(angle) * r;
        T armor_y = x[(int)StateID::CY] - ceres::sin(angle) * r;
        T armor_z = (outpost) ? getoutpost_armor_z(x)
            : (use_l_h)       ? x[(int)StateID::CZ] + x[(int)StateID::H]
                              : x[(int)StateID::CZ];
        return { armor_x, armor_y, armor_z, angle };
    }
    Eigen::Vector4d h_armor_xyza(const VecX& x) const {
        assert(x.size() == X_N);
        auto [armor_x, armor_y, armor_z, angle] = h_armor_xyza(x.data());

        return { armor_x, armor_y, armor_z, angle };
    }
    template<typename T>
    T getoutpost_armor_z(const T x[X_N]) const {
        return (ctx.id == 0) ? x[(int)StateID::CZ]
            : (ctx.id == 1)  ? x[(int)StateID::CZ] + x[(int)StateID::outpost01DZ]
            : (ctx.id == 2)  ? x[(int)StateID::CZ] + x[(int)StateID::outpost02DZ]
                             : x[(int)StateID::CZ];
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
