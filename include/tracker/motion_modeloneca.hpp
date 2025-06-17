#pragma once
#include "tracker/extended_kalman_filter.hpp"
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

/// 状态预测模型
struct Predict {
  explicit Predict(double dt, MotionModel model = MotionModel::CONSTANT_ACCEL_ROT)
      : dt(dt), model(model) {}

  template <typename T>
  void operator()(const T x0[X_N], T x1[X_N]) {
    for (int i = 0; i < X_N; ++i) {
      x1[i] = x0[i];
    }

    // 匀加速线性运动
    x1[0] += x0[1] * dt + T(0.5) * x0[2] * dt * dt; // x += vx * dt + 0.5 * ax * dt^2
    x1[1] += x0[2] * dt;                            // vx += ax * dt

    x1[3] += x0[4] * dt + T(0.5) * x0[5] * dt * dt; // y += vy * dt + 0.5 * ay * dt^2
    x1[4] += x0[5] * dt;                            // vy += ay * dt

    x1[6] += x0[7] * dt + T(0.5) * x0[8] * dt * dt; // z += vz * dt + 0.5 * az * dt^2
    x1[7] += x0[8] * dt;                            // vz += az * dt

    // 匀角速度旋转
    x1[9] += x0[10] * dt;                           // yaw += yaw_rate * dt
  }

  double dt;
  MotionModel model;
};

/// 观测模型
struct Measure {
  template <typename T> void operator()(const T x[X_N], T z[Z_N]) {
    z[0] = x[0];  // x
    z[1] = x[3];  // y
    z[2] = x[6];  // z
    z[3] = x[9];  // yaw
  }
};

/// EKF 类型定义
using RobotStateEKF = ExtendedKalmanFilter<X_N, Z_N, Predict, Measure>;

} // namespace onecaarmor_motion_model
