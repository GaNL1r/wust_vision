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
#include <opencv2/calib3d.hpp>
#include <opencv2/core/mat.hpp>
// project
#include "KalmanHyLib/kalman_hybird_lib.hpp"

namespace crazy {
enum class MotionModel {
    CONSTANT_VELOCITY = 0, // Constant velocity
    CONSTANT_ROTATION = 1, // Constant rotation velocity
    CONSTANT_VEL_ROT = 2 // Constant velocity and rotation velocity
};

constexpr int X_N = 11, Z_N = 8;
using VecZ = Eigen::Matrix<double, Z_N, 1>;
using VecX = Eigen::Matrix<double, X_N, 1>;
enum class Mean : uint8_t {
    PLBX = 0,
    PLBY = 1,
    PLTX = 2,
    PLTY = 3,
    PRTX = 4,
    PRTY = 5,
    PRBX = 6,
    PRBY = 7,
    // IMUY = 4,
    // IMUP = 5,
    // IMUR = 6,
    Z_N = 8
};
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

        x1[(int)State::VCZ] *= T(0.);
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
constexpr double SMALL_ARMOR_WIDTH = 133.0 / 1000.0; // 135
constexpr double SMALL_ARMOR_HEIGHT = 50.0 / 1000.0; // 55
constexpr double LARGE_ARMOR_WIDTH = 225.0 / 1000.0;
constexpr double LARGE_ARMOR_HEIGHT = 50.0 / 1000.0; // 55

constexpr double FIFTTEN_DEGREE_RAD = 15 * CV_PI / 180;

template<typename T>
T normalize_angle_t(T angle) {
    T two_pi = T(2.0 * M_PI);
    return angle - two_pi * floor((angle + T(M_PI)) / two_pi);
}

template<typename T>
Eigen::Quaternion<T>
eulerToQuat(const Eigen::Vector<T, 3>& euler, int axis0, int axis1, int axis2, bool extrinsic) {
    T rz = euler[0];
    T ry = euler[1];
    T rx = euler[2];

    Eigen::Quaternion<T> qx(Eigen::AngleAxis<T>(rx, Eigen::Vector3<T>::UnitX()));
    Eigen::Quaternion<T> qy(Eigen::AngleAxis<T>(ry, Eigen::Vector3<T>::UnitY()));
    Eigen::Quaternion<T> qz(Eigen::AngleAxis<T>(rz, Eigen::Vector3<T>::UnitZ()));

    if (!extrinsic)
        std::swap(axis0, axis2);

    Eigen::Quaternion<T> q;

    if (axis0 == 0 && axis1 == 1 && axis2 == 2)
        q = qx * qy * qz;
    else if (axis0 == 0 && axis1 == 2 && axis2 == 1)
        q = qx * qz * qy;
    else if (axis0 == 1 && axis1 == 0 && axis2 == 2)
        q = qy * qx * qz;
    else if (axis0 == 1 && axis1 == 2 && axis2 == 0)
        q = qy * qz * qx;
    else if (axis0 == 2 && axis1 == 0 && axis2 == 1)
        q = qz * qx * qy;
    else if (axis0 == 2 && axis1 == 1 && axis2 == 0)
        q = qz * qy * qx;
    else
        throw std::invalid_argument("Unsupported axis order");

    return q;
}

template<typename PointType, typename U>
std::vector<PointType> buildObjectPoints(const U& w, const U& h) noexcept {
    using T = U;
    T half2 = T(2.0);
    return { PointType(T(0), w / half2, -h / half2),
             PointType(T(0), w / half2, h / half2),
             PointType(T(0), -w / half2, h / half2),
             PointType(T(0), -w / half2, -h / half2) };
}

struct Measure {
    struct MeasureCtx {
        int armor_num = 4;
        int id = 0;
        Eigen::Matrix4d T_odom_to_camera_d;
        cv::Mat camera_intrinsic;
        cv::Mat camera_distortion;
        bool is_big;
    } ctx;

    Measure() = default;
    explicit Measure(const MeasureCtx& c): ctx(c) {}

    template<typename T>
    void operator()(const T x[X_N], T z[Z_N]) const {
        T id_t = T(ctx.id);
        T num_t = T(ctx.armor_num);
        T two = T(2.0);
        T angle = normalize_angle_t(x[(int)State::YAW] + id_t * two * T(M_PI) / num_t);

        bool outpost = (ctx.armor_num == 3);
        bool use_l_h = (ctx.armor_num == 4) && (ctx.id == 1 || ctx.id == 3);

        T r = use_l_h ? x[(int)State::R] + x[(int)State::L] : x[(int)State::R];

        T armor_x = x[(int)State::CX] - ceres::cos(angle) * r;
        T armor_y = x[(int)State::CY] - ceres::sin(angle) * r;
        T armor_z = outpost ? getoutpost_armor_z(x)
            : use_l_h       ? x[(int)State::CZ] + x[(int)State::H]
                            : x[(int)State::CZ];

        Eigen::Vector3<T> euler_odom;
        euler_odom[0] = angle; //yaw 
        euler_odom[1] = outpost ? T(-FIFTTEN_DEGREE_RAD) : T(FIFTTEN_DEGREE_RAD); //pitch 
        euler_odom[2] = T(M_PI / 2.0); //roll

        Eigen::Quaternion<T> q_odom = eulerToQuat(euler_odom, 2, 1, 0, true);

        Eigen::Matrix4<T> T_odom_to_camera = ctx.T_odom_to_camera_d.cast<T>();

        Eigen::Vector4<T> pos_odom4(armor_x, armor_y, armor_z, T(1.0));
        Eigen::Vector4<T> pos_camera4 = T_odom_to_camera * pos_odom4;
        Eigen::Vector3<T> pos_camera = pos_camera4.template head<3>();

        Eigen::Matrix3<T> R_odom_to_camera = T_odom_to_camera.block(0, 0, 3, 3).template cast<T>();
        Eigen::Matrix3<T> R_ori_odom = q_odom.normalized().toRotationMatrix();
        Eigen::Matrix3<T> R_camera = R_odom_to_camera * R_ori_odom;
        Eigen::Quaternion<T> q_camera(R_camera);
        q_camera.normalize();

        T w3 = ctx.is_big ? T(LARGE_ARMOR_WIDTH) : T(SMALL_ARMOR_WIDTH);
        T h3 = ctx.is_big ? T(LARGE_ARMOR_HEIGHT) : T(SMALL_ARMOR_HEIGHT);

        auto objPts = buildObjectPoints<Eigen::Matrix<T, 3, 1>>(w3, h3);

        Eigen::Matrix3<T> R = q_camera.toRotationMatrix();
        Eigen::Matrix<T, 3, 1> t = pos_camera;

        std::vector<Eigen::Matrix<T, 3, 1>> Pc;
        Pc.reserve(objPts.size());
        for (const auto& p: objPts) {
            Eigen::Matrix<T, 3, 1> v = p;
            Pc.push_back(R  * v + t);
        }

        const cv::Mat& K = ctx.camera_intrinsic;
        T fx = T(K.at<double>(0, 0));
        T fy = T(K.at<double>(1, 1));
        T cx = T(K.at<double>(0, 2));
        T cy = T(K.at<double>(1, 2));

        std::array<T, 4> u, v;
        for (int i = 0; i < 4; i++) {
            T Xc = Pc[i][0];
            T Yc = Pc[i][1];
            T Zc = Pc[i][2];

            u[i] = fx * (Xc / Zc) + cx;
            v[i] = fy * (Yc / Zc) + cy;
        }

        z[0] = u[0];
        z[1] = v[0];
        z[2] = u[1];
        z[3] = v[1];
        z[4] = u[2];
        z[5] = v[2];
        z[6] = u[3];
        z[7] = v[3];
    }

    template<typename T>
    T getoutpost_armor_z(const T x[X_N]) const {
        if (ctx.id == 0)
            return x[(int)State::CZ];
        if (ctx.id == 1)
            return x[(int)State::CZ] + x[(int)State::outpost01DZ];
        if (ctx.id == 2)
            return x[(int)State::CZ] + x[(int)State::outpost02DZ];
        return x[(int)State::CZ];
    }

    using VecX = Eigen::Matrix<double, X_N, 1>;
    using VecZ = Eigen::Matrix<double, Z_N, 1>;

    void h(const VecX& x, VecZ& z) const {
        operator()(x.data(), z.data());
    }
};

using RobotStateEKF = kalman_hybird_lib::ExtendedKalmanFilter<X_N, Z_N, Predict, Measure>;
using RobotStateESEKF = kalman_hybird_lib::ErrorStateEKF<X_N, Z_N, Predict, Measure>;

} // namespace crazy