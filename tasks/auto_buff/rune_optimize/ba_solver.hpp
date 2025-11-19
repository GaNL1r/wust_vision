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
#include <Eigen/Core>
#include <Eigen/Dense>
#include <opencv2/core.hpp>
#include <sophus/so3.hpp>

// g2o
#include <g2o/core/base_multi_edge.h>
#include <g2o/core/base_vertex.h>
#include <g2o/core/optimization_algorithm.h>
#include <g2o/core/optimization_algorithm_factory.h>
#include <g2o/core/optimization_algorithm_levenberg.h>
#include <g2o/core/robust_kernel.h>
#include <g2o/core/sparse_optimizer.h>
// project
#include "graph_optimizer.hpp"
#include "tasks/auto_aim/type.hpp"
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
    BaSolver(
        std::array<double, 9>& camera_matrix,
        int max_iter_R,
        int max_iter_t,
        int step_R,
        int step_t,
        double min_error_R,
        double min_error_t
    );
    ~BaSolver() = default;

    // Solve the armor pose using the BA algorithm, return the optimized rotation
    Eigen::Matrix3d solveBa_R(
        const rune::RuneFan::Simple& rune_fan,
        const Eigen::Vector3d& t_camera_armor,
        const Eigen::Matrix3d& R_camera_armor,
        const Eigen::Matrix3d& R_imu_camera,
        const cv::Mat& camera_matrix,
        const cv::Mat& distort_coeffs
    ) noexcept;
    int max_iter_R_;
    int max_iter_t_;
    int step_R_;
    int step_t_;
    double min_error_R_;
    double min_error_t_;

private:
    Eigen::Matrix3d K_;
    g2o::SparseOptimizer optimizer_;
    g2o::OptimizationAlgorithmProperty solver_property_;
    g2o::OptimizationAlgorithmLevenberg* lm_algorithm_;
};

} // namespace rune
