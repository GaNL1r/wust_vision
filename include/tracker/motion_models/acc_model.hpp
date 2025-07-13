#pragma once

// ceres
#include <ceres/ceres.h>
// project
#include "tracker/math/error_state_extended_kalman_filter.hpp"
#include "tracker/math/extended_kalman_filter.hpp"

namespace acc_model {

// 状态维度，6 维：速度与加速度
constexpr int X_N = 6;
// 观测维度，3 维：速度观测
constexpr int Z_N = 3;

struct Predict {
    explicit Predict(double dt): dt(dt) {}

    template<typename T>
    void operator()(const T x0[X_N], T x1[X_N]) {
        // 复制状态
        for (int i = 0; i < X_N; i++) {
            x1[i] = x0[i];
        }

        // v += a * dt
        x1[0] += x0[1] * dt; // vx += ax * dt
        x1[2] += x0[3] * dt; // vy += ay * dt
        x1[4] += x0[5] * dt; // vz += az * dt
        // 加速度保持不变
    }

    double dt;
};

struct Measure {
    template<typename T>
    void operator()(const T x[X_N], T z[Z_N]) {
        // 观测速度部分
        z[0] = x[0]; // vx
        z[1] = x[2]; // vy
        z[2] = x[4]; // vz
    }
};

using VelocityAccelEKF = ExtendedKalmanFilter<X_N, Z_N, Predict, Measure>;

} // namespace acc_model
