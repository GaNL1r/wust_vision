// Created by Labor 2023.8.25
// Maintained by Labor, Chengfu Zou
// Copyright (C) FYT Vision Group. All rights reserved.
// Copyright 2025 XiaoJian Wu
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
#include "tasks/auto_aim/type.hpp"
#include <ceres/autodiff_cost_function.h>
#include <ceres/local_parameterization.h>
#include <ceres/loss_function.h>
#include <ceres/problem.h>
#include <ceres/solver.h>
// BA algorithm based Optimizer for the armor pose estimation (Particularly for
// the Yaw angle)
class BaSolver {
public:
    BaSolver(const YAML::Node& config, const std::pair<cv::Mat, cv::Mat>& camera_info);
    ~BaSolver() = default;
    struct Params {
        enum class OptMode : int {
            GOLDEN = 0,
            CERES = 1

        } mode;
        OptMode fromString(const std::string& mode) {
            if (mode == "golden" || mode == "GOLDEN") {
                return OptMode::GOLDEN;
            } else if (mode == "ceres" || mode == "CERES") {
                return OptMode::CERES;
            } else {
                return OptMode::GOLDEN;
            }
        }
        int ceres_max_iter = 40;
        int golden_search_side_deg = 60;
        void load(const YAML::Node& node) {
            mode = fromString(node["mode"].as<std::string>());
            ceres_max_iter = node["ceres_max_iter"].as<int>();
            golden_search_side_deg = node["golden_search_side_deg"].as<int>();
        }
    } params_;
    // Solve the armor pose using the BA algorithm, return the optimized rotation
    Eigen::Matrix3d solveBa_R(
        const armor::ArmorObject& armor,
        const Eigen::Vector3d& t_camera_armor,
        const Eigen::Matrix3d& R_camera_armor,
        const Eigen::Matrix3d& R_imu_camera,
        const std::string& type
    ) noexcept;
    double goldenYaw(
        double init,
        const std::vector<Eigen::Vector3d>& obj,
        const std::vector<cv::Point2f>& lm,
        const Eigen::Matrix3d& Rci,
        double pitch,
        double roll,
        const Eigen::Vector3d& t,
        const Eigen::Matrix3d& K
    );
    double ceresYaw(
        double init,
        const std::vector<Eigen::Vector3d>& obj,
        const std::vector<cv::Point2f>& lm,
        const Eigen::Matrix3d& Rci,
        double pitch,
        double roll,
        const Eigen::Vector3d& t,
        const Eigen::Matrix3d& K
    );

private:
    Eigen::Matrix3d K_;
    std::pair<cv::Mat, cv::Mat> camera_info_;
    Eigen::Vector<double, 5> dist_eigen_;
};
class YawLocalParameterization: public ceres::LocalParameterization {
public:
    bool Plus(const double* x, const double* delta, double* x_plus_delta) const override {
        x_plus_delta[0] = x[0] + delta[0];
        return true;
    }
    bool ComputeJacobian(const double* x, double* jacobian) const override {
        jacobian[0] = 1.0;
        return true;
    }
    int GlobalSize() const override {
        return 1;
    }
    int LocalSize() const override {
        return 1;
    }
};

struct CameraProjector {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    CameraProjector(
        const Eigen::Matrix3d& Rci,
        double pitch,
        double roll,
        const Eigen::Vector3d& t,
        const Eigen::Matrix3d& K,
        const Eigen::Vector<double, 5>& dist
    ):
        Rci_(Rci),
        pitch_(pitch),
        roll_(roll),
        t_(t),
        K_(K),
        dist_(dist) {}

    template<typename T, typename Derived>
    inline Eigen::Matrix<T, 2, 1>
    project(const T& yaw, const Eigen::MatrixBase<Derived>& Pw) const {
        T cy = ceres::cos(yaw);
        T sy = ceres::sin(yaw);
        Eigen::Matrix<T, 3, 3> R_yaw;
        R_yaw << cy, -sy, T(0), sy, cy, T(0), T(0), T(0), T(1);

        T cp = ceres::cos(T(pitch_));
        T sp = ceres::sin(T(pitch_));
        Eigen::Matrix<T, 3, 3> R_pitch;
        R_pitch << cp, T(0), sp, T(0), T(1), T(0), -sp, T(0), cp;

        T cr = ceres::cos(T(roll_));
        T sr = ceres::sin(T(roll_));
        Eigen::Matrix<T, 3, 3> R_roll;
        R_roll << cr, -sr, T(0), sr, cr, T(0), T(0), T(0), T(1);

        Eigen::Matrix<T, 3, 3> R = Rci_.cast<T>() * R_yaw * R_pitch * R_roll;

        Eigen::Matrix<T, 3, 1> Pc = R * Pw + t_.cast<T>();

        if (Pc.z() < T(0.3)) {
            return Eigen::Matrix<T, 2, 1>(T(0), T(0));
        }

        T x = Pc.x() / Pc.z();
        T y = Pc.y() / Pc.z();

        if (!ceres::isfinite(x) || !ceres::isfinite(y)) {
            return Eigen::Matrix<T, 2, 1>(T(0), T(0));
        }

        T r2 = x * x + y * y;
        const T k1 = T(dist_(0));
        const T k2 = T(dist_(1));
        const T p1 = T(dist_(2));
        const T p2 = T(dist_(3));
        const T k3 = T(dist_(4));

        T x_dist = x * (T(1) + k1 * r2 + k2 * r2 * r2 + k3 * r2 * r2 * r2) + T(2) * p1 * x * y
            + p2 * (r2 + T(2) * x * x);
        T y_dist = y * (T(1) + k1 * r2 + k2 * r2 * r2 + k3 * r2 * r2 * r2)
            + p1 * (r2 + T(2) * y * y) + T(2) * p2 * x * y;

        if (!ceres::isfinite(x_dist) || !ceres::isfinite(y_dist)) {
            return Eigen::Matrix<T, 2, 1>(T(0), T(0));
        }

        T xu = x_dist;
        T yu = y_dist;

        for (int i = 0; i < 8; i++) {
            T r2_u = xu * xu + yu * yu;
            T radial = T(1) + k1 * r2_u + k2 * r2_u * r2_u + k3 * r2_u * r2_u * r2_u;
            T dx = T(2) * p1 * xu * yu + p2 * (r2_u + T(2) * xu * xu);
            T dy = p1 * (r2_u + T(2) * yu * yu) + T(2) * p2 * xu * yu;

            xu = (x_dist - dx) / radial;
            yu = (y_dist - dy) / radial;

            if (!ceres::isfinite(xu) || !ceres::isfinite(yu)) {
                return Eigen::Matrix<T, 2, 1>(T(0), T(0));
            }
        }
        Eigen::Matrix<T, 2, 1> uv;
        uv(0) = T(K_(0, 0)) * xu + T(K_(0, 2));
        uv(1) = T(K_(1, 1)) * yu + T(K_(1, 2));
        return uv;
    }

    Eigen::Matrix3d Rci_;
    double pitch_;
    double roll_;
    Eigen::Vector3d t_;
    Eigen::Matrix3d K_;
    Eigen::Vector<double, 5> dist_;
};

struct ReprojectionError {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    ReprojectionError(
        const Eigen::Vector2d& uv,
        const Eigen::Vector3d& pt3d,
        const CameraProjector& cam
    ):
        uv_(uv),
        pt3_(pt3d),
        cam_(cam) {}

    template<typename T>
    bool operator()(const T* const yaw, T* residuals) const {
        Eigen::Matrix<T, 2, 1> uv_proj = cam_.project(yaw[0], pt3_.cast<T>());

        residuals[0] = uv_proj(0) - T(uv_(0));
        residuals[1] = uv_proj(1) - T(uv_(1));
        return true;
    }

    Eigen::Vector2d uv_;
    Eigen::Vector3d pt3_;
    CameraProjector cam_;
};
struct SymmetryError {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    SymmetryError(
        const Eigen::Vector3d& p1,
        const Eigen::Vector3d& p2,
        const Eigen::Vector2d& measCenter,
        const CameraProjector& cam
    ):
        p1_(p1),
        p2_(p2),
        meas_(measCenter),
        cam_(cam) {}

    template<typename T>
    bool operator()(const T* const yaw, T* residuals) const {
        Eigen::Matrix<T, 2, 1> uv1 = cam_.project(yaw[0], p1_.cast<T>());
        Eigen::Matrix<T, 2, 1> uv2 = cam_.project(yaw[0], p2_.cast<T>());

        residuals[0] = (uv1(0) + uv2(0)) * T(0.5) - T(meas_(0));
        residuals[1] = (uv1(1) + uv2(1)) * T(0.5) - T(meas_(1));
        return true;
    }

    Eigen::Vector3d p1_;
    Eigen::Vector3d p2_;
    Eigen::Vector2d meas_;
    CameraProjector cam_;
};
