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
#include <ceres/jet.h>
#include <ceres/loss_function.h>
#include <ceres/problem.h>
#include <opencv2/core.hpp>
// project
#include "tasks/auto_buff/type.hpp"
namespace rune {
inline double normalizeAngleaa(double angle) {
    while (angle > 180.0)
        angle -= 360.0;
    while (angle < -180.0)
        angle += 360.0;
    return angle;
}

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
        const rune::RuneFan::Simple& rune_fan,
        const Eigen::Vector3d& t_camera_armor,
        const Eigen::Matrix3d& R_camera_armor,
        const Eigen::Matrix3d& R_imu_camera
    ) noexcept;

    double ceresRoll(
        double init,
        const std::vector<Eigen::Vector3d>& obj,
        const std::vector<cv::Point2f>& lm,
        const Eigen::Matrix3d& Rci,
        double pitch,
        double yaw,
        const Eigen::Vector3d& t,
        const Eigen::Matrix3d& K
    );
    double goldenRoll(
        double init,
        const std::vector<Eigen::Vector3d>& obj,
        const std::vector<cv::Point2f>& lm,
        const Eigen::Matrix3d& Rci,
        double pitch,
        double yaw,
        const Eigen::Vector3d& t,
        const Eigen::Matrix3d& K
    );

private:
    Eigen::Matrix3d K_;
};
struct RollProjectionError {
    RollProjectionError(
        const Eigen::Vector2d& uv,
        const Eigen::Vector3d& pt_3d,
        const Eigen::Matrix3d& Rci,
        double pitch,
        double yaw,
        const Eigen::Vector3d& t,
        const Eigen::Matrix3d& K
    ):
        uv_(uv),
        pt3_(pt_3d),
        Rci_(Rci),
        pitch_(pitch),
        yaw_(yaw),
        t_(t),
        K_(K) {}

    template<typename T>
    bool operator()(const T* const roll, T* residuals) const {
        Eigen::Matrix<T, 3, 3> R_roll;
        R_roll << T(1), T(0), T(0), T(0), ceres::cos(*roll), -ceres::sin(*roll), T(0),
            ceres::sin(*roll), ceres::cos(*roll);

        // yaw 和 pitch 常量
        T cy = ceres::cos(T(yaw_)), sy = ceres::sin(T(yaw_));
        Eigen::Matrix<T, 3, 3> R_yaw;
        R_yaw << cy, T(0), sy, T(0), T(1), T(0), -sy, T(0), cy;

        T cp = ceres::cos(T(pitch_)), sp = ceres::sin(T(pitch_));
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
    const double pitch_, yaw_;
    const Eigen::Vector3d t_;
    const Eigen::Matrix3d K_;
};

} // namespace rune
