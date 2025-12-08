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
// std
#include <array>
#include <cstddef>
#include <tuple>
#include <vector>
// 3rd party
#include <Eigen/Dense>
#include <ceres/autodiff_cost_function.h>
#include <ceres/local_parameterization.h>
#include <ceres/loss_function.h>
#include <ceres/problem.h>
#include <ceres/solver.h>
#include <opencv2/core.hpp>
// project
#include "tasks/auto_aim/type.hpp"
// BA algorithm based Optimizer for the armor pose estimation (Particularly for
// the Yaw angle)
class BaSolver {
public:
    BaSolver(const YAML::Node& config, const cv::Mat& camera_matrix);
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
        const Eigen::Vector3d& t,
        const Eigen::Matrix3d& K
    );
    double ceresYaw(
        double init,
        const std::vector<Eigen::Vector3d>& obj,
        const std::vector<cv::Point2f>& lm,
        const Eigen::Matrix3d& Rci,
        double pitch,
        const Eigen::Vector3d& t,
        const Eigen::Matrix3d& K
    );

private:
    Eigen::Matrix3d K_;
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

struct ReprojectionError {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    ReprojectionError(
        const Eigen::Vector2d& uv,
        const Eigen::Vector3d& pt_3d,
        const Eigen::Matrix3d& Rci,
        double pitch,
        const Eigen::Vector3d& t,
        const Eigen::Matrix3d& K
    ):
        uv_(uv),
        pt3_(pt_3d),
        Rci_(Rci),
        pitch_(pitch),
        t_(t),
        K_(K) {}

    template<typename T>
    bool operator()(const T* const yaw, T* residuals) const {
        // build yaw rotation matrix
        T cy = ceres::cos(yaw[0]);
        T sy = ceres::sin(yaw[0]);
        Eigen::Matrix<T, 3, 3> R_yaw;
        R_yaw << cy, -sy, T(0), sy, cy, T(0), T(0), T(0), T(1);

        // build pitch rotation matrix (pitch is double, cast to T)
        T cp = ceres::cos(T(pitch_));
        T sp = ceres::sin(T(pitch_));
        Eigen::Matrix<T, 3, 3> R_pitch;
        // Note: consistent with your previous Rpitch = exp([0, pitch, 0])
        R_pitch << cp, T(0), sp, T(0), T(1), T(0), -sp, T(0), cp;

        // R = Rci * R_yaw * R_pitch
        Eigen::Matrix<T, 3, 3> R = Rci_.cast<T>() * R_yaw * R_pitch;

        Eigen::Matrix<T, 3, 1> Pc = R * pt3_.cast<T>() + t_.cast<T>();

        // project (assumes fx == K(0,0), fy == K(1,1), cx, cy)
        T u = T(K_(0, 0)) * Pc.x() / Pc.z() + T(K_(0, 2));
        T v = T(K_(1, 1)) * Pc.y() / Pc.z() + T(K_(1, 2));

        residuals[0] = u - T(uv_(0));
        residuals[1] = v - T(uv_(1));
        return true;
    }

    const Eigen::Vector2d uv_;
    const Eigen::Vector3d pt3_;
    const Eigen::Matrix3d Rci_;
    const double pitch_;
    const Eigen::Vector3d t_;
    const Eigen::Matrix3d K_;
};

struct SymmetryError {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    SymmetryError(
        const Eigen::Vector3d& p1,
        const Eigen::Vector3d& p2,
        const Eigen::Vector2d& measCenter,
        const Eigen::Matrix3d& Rci,
        double pitch,
        const Eigen::Vector3d& t,
        const Eigen::Matrix3d& K
    ):
        p1_(p1),
        p2_(p2),
        meas_(measCenter),
        Rci_(Rci),
        pitch_(pitch),
        t_(t),
        K_(K) {}

    template<typename T>
    bool operator()(const T* const yaw, T* residuals) const {
        T cy = ceres::cos(yaw[0]);
        T sy = ceres::sin(yaw[0]);
        Eigen::Matrix<T, 3, 3> R_yaw;
        R_yaw << cy, -sy, T(0), sy, cy, T(0), T(0), T(0), T(1);

        T cp = ceres::cos(T(pitch_));
        T sp = ceres::sin(T(pitch_));
        Eigen::Matrix<T, 3, 3> R_pitch;
        R_pitch << cp, T(0), sp, T(0), T(1), T(0), -sp, T(0), cp;

        Eigen::Matrix<T, 3, 3> R = Rci_.cast<T>() * R_yaw * R_pitch;

        Eigen::Matrix<T, 3, 1> Pc1 = R * p1_.cast<T>() + t_.cast<T>();
        Eigen::Matrix<T, 3, 1> Pc2 = R * p2_.cast<T>() + t_.cast<T>();

        T u1 = T(K_(0, 0)) * Pc1.x() / Pc1.z() + T(K_(0, 2));
        T v1 = T(K_(1, 1)) * Pc1.y() / Pc1.z() + T(K_(1, 2));

        T u2 = T(K_(0, 0)) * Pc2.x() / Pc2.z() + T(K_(0, 2));
        T v2 = T(K_(1, 1)) * Pc2.y() / Pc2.z() + T(K_(1, 2));

        residuals[0] = (u1 + u2) * T(0.5) - T(meas_(0));
        residuals[1] = (v1 + v2) * T(0.5) - T(meas_(1));
        return true;
    }

    const Eigen::Vector3d p1_, p2_;
    const Eigen::Vector2d meas_;
    const Eigen::Matrix3d Rci_;
    const double pitch_;
    const Eigen::Vector3d t_;
    const Eigen::Matrix3d K_;
};
