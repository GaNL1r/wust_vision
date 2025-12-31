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
struct Measure {
    Measure() = default;
    struct MeasureCtx {
        int armor_num = 4;
        int id = 0;
        Eigen::Matrix4d T_camera_to_odom;
        cv::Mat camera_intrinsic;
        cv::Mat camera_distortion;
        bool is_big;
    } ctx;
    explicit Measure(MeasureCtx c):ctx(c) {}
    template<typename T>
    void operator()(const T x[X_N], T z[Z_N]) const {
        T angle = normalize_angle_t(x[(int)State::YAW] + ctx.id * 2 * M_PI / ctx.armor_num);
        auto outpost = ctx.armor_num == 3;
        auto use_l_h = (ctx.armor_num == 4) && (ctx.id == 1 || ctx.id == 3);
        T r = (use_l_h) ? x[(int)State::R] + x[(int)State::L] : x[(int)State::R];
        T armor_x = x[(int)State::CX] - ceres::cos(angle) * r;
        T armor_y = x[(int)State::CY] - ceres::sin(angle) * r;
        T armor_z = (outpost) ? getoutpost_armor_z(x)
            : (use_l_h)       ? x[(int)State::CZ] + x[(int)State::H]
                              : x[(int)State::CZ];
        Eigen::Vector3<T> euler_odom;
        euler_odom[0] = angle; //yaw
        euler_odom[1] = outpost ? -FIFTTEN_DEGREE_RAD : FIFTTEN_DEGREE_RAD; //pitch
        euler_odom[2] = M_PI / 2.0; //roll
        Eigen::Quaternion<T> ori_odom = eulerToQuat(euler_odom, 2, 1, 0, true);
        Eigen::Matrix4<T> T_odom_to_camera = ctx.T_camera_to_odom.inverse();
        Eigen::Vector3<T> pos_odom(armor_x, armor_y, armor_z);
        Eigen::Vector4<T> pos_odom4(armor_x, armor_y, armor_z, 1.0);
        Eigen::Vector4<T> pos_camera4 = T_odom_to_camera * pos_odom4;
        Eigen::Vector3<T> pos_camera = pos_camera4.template head<3>();
        Eigen::Matrix3<T> R_odom_to_camera = T_odom_to_camera.block(0, 0, 3, 3);
        Eigen::Matrix3<T> R_ori_odom = ori_odom.normalized().toRotationMatrix();
        Eigen::Matrix3<T> R_ori_camera = R_odom_to_camera * R_ori_odom;
        Eigen::Quaternion<T> ori_camera = Eigen::Quaternion<T>(R_ori_camera).normalized();
        Eigen::Vector2<T> armor_size = ctx.is_big
            ? Eigen::Vector2<T> { LARGE_ARMOR_WIDTH, LARGE_ARMOR_HEIGHT }
            : Eigen::Vector2<T> { SMALL_ARMOR_WIDTH, SMALL_ARMOR_HEIGHT };
        auto objPts = buildObjectPoints<Eigen::Vector3<T>>(armor_size.x(), armor_size.y());
        Eigen::Matrix3<T> R = ori_camera.toRotationMatrix();
        Eigen::Matrix<T, 3, 1> t = pos_camera;

        std::vector<Eigen::Matrix<T, 3, 1>> Pc;
        Pc.reserve(objPts.size());
        for (const auto& Pobj: objPts) {
            Eigen::Matrix<T, 3, 1> p(Pobj.x(), Pobj.y(), Pobj.z());
            Pc.push_back(R * p + t);
        }

        T fx = T(ctx.camera_intrinsic.at<double>(0, 0));
        T fy = T(ctx.camera_intrinsic.at<double>(1, 1));
        T cx = T(ctx.camera_intrinsic.at<double>(0, 2));
        T cy = T(ctx.camera_intrinsic.at<double>(1, 2));

        std::vector<Eigen::Vector2<T>> projPts;
        projPts.reserve(Pc.size());
        for (const auto& p: Pc) {
            if (p[2] <= T(1e-6)) {
                projPts.emplace_back(T(-1), T(-1));
                continue;
            }
            T u = fx * (p[0] / p[2]) + cx;
            T v = fy * (p[1] / p[2]) + cy;
            projPts.emplace_back(u, v);
        }

        std::vector<cv::Point2f> finalPts;
        finalPts.reserve(projPts.size());

        std::vector<cv::Point3f> cvObj;
        cvObj.reserve(Pc.size());
        for (const auto& p: Pc) {
            cvObj.emplace_back((float)(p[0].a), (float)(p[1].a), (float)(p[2].a));
        }
        std::vector<cv::Point2f> cvImgPts;
        cv::projectPoints(
            cvObj,
            cv::Mat::zeros(3, 1, CV_64F), // rvec = 0 (因为我们已经旋转过了)
            cv::Mat::zeros(3, 1, CV_64F), // tvec = 0
            ctx.camera_intrinsic,
            ctx.camera_distortion,
            cvImgPts
        );
        for (const auto& pt: cvImgPts) {
            finalPts.push_back(pt);
        }
        z[(int)Mean::PLBX] = finalPts[0].x;
        z[(int)Mean::PLBY] = finalPts[0].y;
        z[(int)Mean::PLTX] = finalPts[1].x;
        z[(int)Mean::PLTY] = finalPts[1].y;
        z[(int)Mean::PRTX] = finalPts[2].x;
        z[(int)Mean::PRTY] = finalPts[2].y;
        z[(int)Mean::PRBX] = finalPts[3].x;
        z[(int)Mean::PRBY] = finalPts[3].y;
    }
    template<typename T>
    T getoutpost_armor_z(const T x[X_N]) const {
        return (ctx.id == 0) ? x[(int)State::CZ]
            : (ctx.id == 1)  ? x[(int)State::CZ] + x[(int)State::outpost01DZ]
            : (ctx.id == 2)  ? x[(int)State::CZ] + x[(int)State::outpost02DZ]
                         : x[(int)State::CZ];
    }
    template<typename T>
    Eigen::Quaternion<T>
    eulerToQuat(const Eigen::Vector<T, 3>& euler, int axis0, int axis1, int axis2, bool extrinsic) {
        double rz = euler[0], ry = euler[1], rx = euler[2];
        Eigen::Quaternion<T> qx(Eigen::AngleAxis<T>(rx, Eigen::Vector3<T>::UnitX()));
        Eigen::Quaternion<T> qy(Eigen::AngleAxis<T>(ry, Eigen::Vector3<T>::UnitY()));
        Eigen::Quaternion<T> qz(Eigen::AngleAxis<T>(rz, Eigen::Vector3<T>::UnitZ()));

        if (!extrinsic)
            std::swap(axis0, axis2);
        Eigen::Quaternion<T> q;

        // 生成四元数
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
    template<typename PointType>
    std::vector<PointType> buildObjectPoints(const double& w, const double& h) noexcept {
        return { PointType(0, w / 2, -h / 2),
                 PointType(0, w / 2, h / 2),
                 PointType(0, -w / 2, h / 2),
                 PointType(0, -w / 2, -h / 2) };
    }
    
};

} // namespace crazy