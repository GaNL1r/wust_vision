// Created by Labor 2023.8.25
// Maintained by Chengfu Zou, Labor
// Copyright (C) FYT Vision Group. All rights reserved.
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

#include "ba_solver.hpp"
// std
#include <iostream>
#include <memory>

// 3rd party
#include <Eigen/Core>
#include <opencv2/core/eigen.hpp>
// project
#include "tasks/utils.hpp"
#include "wust_vl/common/utils/logger.hpp"

BaSolver::BaSolver(const YAML::Node& config, const cv::Mat& camera_matrix) {
    cv::cv2eigen(camera_matrix, K_);
    params_.load(config);
}

double reprojectionErrorYaw(
    double yaw,
    const std::vector<Eigen::Vector3d>& object_points,
    const std::vector<cv::Point2f>& landmarks,
    const Eigen::Matrix3d& Rci,
    double pitch,
    const Eigen::Vector3d& t,
    const Eigen::Matrix3d& K
) {
    double cy = std::cos(yaw), sy = std::sin(yaw);
    Eigen::Matrix3d R_yaw;
    R_yaw << cy, -sy, 0, sy, cy, 0, 0, 0, 1;

    double cp = std::cos(pitch), sp = std::sin(pitch);
    Eigen::Matrix3d R_pitch;
    R_pitch << cp, 0, sp, 0, 1, 0, -sp, 0, cp;

    Eigen::Matrix3d R = Rci * R_yaw * R_pitch;

    double total = 0;
    for (int i = 0; i < object_points.size(); ++i) {
        Eigen::Vector3d Pc = R * object_points[i] + t;

        if (Pc.z() < 1e-6)
            return 1e9; // invalid projection

        Eigen::Vector2d proj(
            K(0, 0) * Pc.x() / Pc.z() + K(0, 2),
            K(1, 1) * Pc.y() / Pc.z() + K(1, 2)
        );
        Eigen::Vector2d obs(landmarks[i].x, landmarks[i].y);

        total += (proj - obs).squaredNorm();
    }
    return total / object_points.size();
}

double BaSolver::goldenYaw(
    double init,
    const std::vector<Eigen::Vector3d>& obj,
    const std::vector<cv::Point2f>& lm,
    const Eigen::Matrix3d& Rci,
    double pitch,
    const Eigen::Vector3d& t,
    const Eigen::Matrix3d& K
) {
    constexpr double phi = 1.618033988749894848; //(1.0 + std::sqrt(5.0)) * 0.5;
    double l = init - params_.golden_search_side_deg * M_PI / 180.0;
    double r = init + params_.golden_search_side_deg * M_PI / 180.0;

    double y1 = r - (r - l) / phi;
    double y2 = l + (r - l) / phi;

    double f1 = reprojectionErrorYaw(y1, obj, lm, Rci, pitch, t, K);
    double f2 = reprojectionErrorYaw(y2, obj, lm, Rci, pitch, t, K);

    while (r - l > 0.0001) { // stop threshold ~0.005 degree
        if (f1 < f2) {
            r = y2;
            y2 = y1;
            f2 = f1;
            y1 = r - (r - l) / phi;
            f1 = reprojectionErrorYaw(y1, obj, lm, Rci, pitch, t, K);
        } else {
            l = y1;
            y1 = y2;
            f1 = f2;
            y2 = l + (r - l) / phi;
            f2 = reprojectionErrorYaw(y2, obj, lm, Rci, pitch, t, K);
        }
    }

    return 0.5 * (l + r);
}
double BaSolver::ceresYaw(
    double initial_yaw,
    const std::vector<Eigen::Vector3d>& object_points,
    const std::vector<cv::Point2f>& landmarks,
    const Eigen::Matrix3d& R_camera_imu,
    double armor_pitch,
    const Eigen::Vector3d& t_camera_armor,
    const Eigen::Matrix3d& K
) {
    double yaw = initial_yaw;

    ceres::Problem problem;
    problem.AddParameterBlock(&yaw, 1, new YawLocalParameterization());

    for (size_t i = 0; i < object_points.size(); ++i) {
        ceres::CostFunction* cost =
            new ceres::AutoDiffCostFunction<ReprojectionError, 2, 1>(new ReprojectionError(
                Eigen::Vector2d(landmarks[i].x, landmarks[i].y),
                object_points[i],
                R_camera_imu,
                armor_pitch,
                t_camera_armor,
                K
            ));
        problem.AddResidualBlock(cost, new ceres::HuberLoss(1.0), &yaw);
    }

    auto buildSymPairs = [&](size_t n) {
        std::vector<std::pair<int, int>> pairs;
        for (int i = 0; i < n / 2; ++i) {
            pairs.emplace_back(i, n - 1 - i);
        }
        return pairs;
    };

    auto symPairs = buildSymPairs(object_points.size());

    for (auto& p: symPairs) {
        Eigen::Vector2d meas = (Eigen::Vector2d(landmarks[p.first].x, landmarks[p.first].y)
                                + Eigen::Vector2d(landmarks[p.second].x, landmarks[p.second].y))
            * 0.5;

        ceres::CostFunction* cost =
            new ceres::AutoDiffCostFunction<SymmetryError, 2, 1>(new SymmetryError(
                object_points[p.first],
                object_points[p.second],
                meas,
                R_camera_imu,
                armor_pitch,
                t_camera_armor,
                K
            ));
        problem.AddResidualBlock(cost, new ceres::HuberLoss(1.0), &yaw);
    }

    // ---- solver options ----
    ceres::Solver::Options options;
    options.max_num_iterations = params_.ceres_max_iter;
    options.linear_solver_type = ceres::DENSE_QR;
    options.minimizer_progress_to_stdout = false;

    // ---- solve ----
    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);

    return yaw;
}

Eigen::Matrix3d BaSolver::solveBa_R(
    const armor::ArmorObject& armor,
    const Eigen::Vector3d& t_camera_armor,
    const Eigen::Matrix3d& R_camera_armor,
    const Eigen::Matrix3d& R_imu_camera,
    const std::string& type
) noexcept {
    Eigen::Matrix3d K = K_;

    Eigen::Matrix3d R_imu_armor = R_imu_camera * R_camera_armor;
    Eigen::Matrix3d R_camera_imu = R_imu_camera.transpose();

    // initial yaw
    double yaw_init = std::atan2(-R_imu_armor(0, 1), R_imu_armor(1, 1));

    double armor_pitch =
        (armor.number == armor::ArmorNumber::OUTPOST) ? -FIFTTEN_DEGREE_RAD : FIFTTEN_DEGREE_RAD;

    Eigen::Vector2d armor_size = (type == "large")
        ? Eigen::Vector2d { LARGE_ARMOR_WIDTH, LARGE_ARMOR_HEIGHT }
        : Eigen::Vector2d { SMALL_ARMOR_WIDTH, SMALL_ARMOR_HEIGHT };

    auto objPts =
        armor::ArmorObject::buildObjectPoints<Eigen::Vector3d>(armor_size.x(), armor_size.y());
    const auto& lm = armor.landmarks();

    double yaw = yaw_init;
    if (params_.mode == Params::OptMode::CERES) {
        yaw = ceresYaw(yaw_init, objPts, lm, R_camera_imu, armor_pitch, t_camera_armor, K);
    } else if (params_.mode == Params::OptMode::GOLDEN) {
        yaw = goldenYaw(yaw_init, objPts, lm, R_camera_imu, armor_pitch, t_camera_armor, K);
    }

    // build yaw + pitch rotation
    double cy = std::cos(yaw), sy = std::sin(yaw);
    Eigen::Matrix3d R_yaw;
    R_yaw << cy, -sy, 0, sy, cy, 0, 0, 0, 1;

    double cp = std::cos(armor_pitch), sp = std::sin(armor_pitch);
    Eigen::Matrix3d R_pitch;
    R_pitch << cp, 0, sp, 0, 1, 0, -sp, 0, cp;

    Eigen::Matrix3d R_result = R_camera_imu * R_yaw * R_pitch;
    return R_result;
}
