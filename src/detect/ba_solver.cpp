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

#include "detect/ba_solver.hpp"
// std
#include <iostream>
#include <memory>
// g2o
#include <g2o/core/robust_kernel.h>
#include <g2o/core/robust_kernel_factory.h>
#include <g2o/core/robust_kernel_impl.h>
#include <g2o/types/slam3d/types_slam3d.h>
// 3rd party
#include <Eigen/Core>
#include <opencv2/core/eigen.hpp>
#include <sophus/se3.hpp>
#include <sophus/so3.hpp>
// project
#include "common/logger.hpp"
#include "detect/graph_optimizer.hpp"
#include "type/type.hpp"

#include "common/utils.hpp"

G2O_USE_OPTIMIZATION_LIBRARY(dense)

BaSolver::BaSolver(std::array<double, 9>& camera_matrix) {
    K_ = Eigen::Matrix3d::Identity();
    K_(0, 0) = camera_matrix[0];
    K_(1, 1) = camera_matrix[4];
    K_(0, 2) = camera_matrix[2];
    K_(1, 2) = camera_matrix[5];

    // Optimization information
    optimizer_.setVerbose(false);
    // Optimization method
    optimizer_.setAlgorithm(
        g2o::OptimizationAlgorithmFactory::instance()->construct("lm_dense", solver_property_)
    );
    // Initial step size
    lm_algorithm_ = dynamic_cast<g2o::OptimizationAlgorithmLevenberg*>(
        const_cast<g2o::OptimizationAlgorithm*>(optimizer_.algorithm())
    );
    lm_algorithm_->setUserLambdaInit(0.1);
}

Eigen::Matrix3d BaSolver::solveBa(
    const ArmorObject& armor,
    const Eigen::Vector3d& t_camera_armor,
    const Eigen::Matrix3d& R_camera_armor,
    const Eigen::Matrix3d& R_imu_camera,
    int type_number
) noexcept {
    optimizer_.clear();

    // 坐标系变换
    Eigen::Matrix3d R_imu_armor = R_imu_camera * R_camera_armor;
    Sophus::SO3d R_camera_imu(R_imu_camera.transpose());
    auto theta_by_sin = std::asin(-R_imu_armor(0, 1));
    auto theta_by_cos = std::acos(R_imu_armor(1, 1));

    double initial_armor_yaw;
    if (std::abs(theta_by_sin) > 1e-5) {
        initial_armor_yaw = theta_by_sin > 0 ? theta_by_cos : -theta_by_cos;
    } else {
        initial_armor_yaw = R_imu_armor(1, 1) > 0 ? 0 : CV_PI;
    }

    double armor_roll = 0.0; // 你可以设置为一个估计值，比如 0.0 或 ±10°
    Sophus::SO3d R_roll = Sophus::SO3d::exp(Eigen::Vector3d(armor_roll, 0, 0));

    // 固定 pitch
    double armor_pitch =
        (armor.number == ArmorNumber::OUTPOST) ? -FIFTTEN_DEGREE_RAD : FIFTTEN_DEGREE_RAD;
    Sophus::SO3d R_pitch = Sophus::SO3d::exp(Eigen::Vector3d(0, armor_pitch, 0));

    // 构建 3D 角点
    Eigen::Vector2d armor_size;
    switch (type_number) {
        case 0:
            armor_size = Eigen::Vector2d(LARGE_ARMOR_WIDTH, LARGE_ARMOR_HEIGHT);
            break;
        case 1:
            armor_size = Eigen::Vector2d(SMALL_ARMOR_WIDTH, SMALL_ARMOR_HEIGHT);
            break;
        case 2:
            armor_size = Eigen::Vector2d(LARGE_ARMOR_WIDTH_NET, LARGE_ARMOR_HEIGHT_NET);
            break;
        case 3:
            armor_size = Eigen::Vector2d(SMALL_ARMOR_WIDTH_NET, SMALL_ARMOR_HEIGHT_NET);
            break;
    }
    auto object_points =
        ArmorObject::buildObjectPoints<Eigen::Vector3d>(armor_size.x(), armor_size.y());
    const auto& landmarks = armor.landmarks();

    size_t id_counter = 0;
    // 添加 yaw 顶点
    auto* v_yaw = new VertexYaw();
    v_yaw->setId(id_counter++);
    v_yaw->setEstimate(initial_armor_yaw);
    optimizer_.addVertex(v_yaw);

    for (size_t i = 0; i < object_points.size(); ++i) {
        auto* v_pt = new g2o::VertexPointXYZ();
        v_pt->setId(id_counter++);
        v_pt->setEstimate(object_points[i]);
        v_pt->setFixed(true);
        optimizer_.addVertex(v_pt);

        auto* edge = new EdgeProjection(R_camera_imu, R_pitch, t_camera_armor, K_);
        edge->setId(id_counter++);
        edge->setVertex(0, v_yaw);
        edge->setVertex(1, v_pt);
        edge->setMeasurement(Eigen::Vector2d(landmarks[i].x, landmarks[i].y));
        edge->setInformation(Eigen::Matrix2d::Identity());
        edge->setRobustKernel(new g2o::RobustKernelHuber);
        optimizer_.addEdge(edge);
    }

    std::vector<std::pair<int, int>> symPairs = { { 0, 5 }, { 1, 4 }, { 2, 3 } };
    const double symWeight = 1.0;

    for (auto& p: symPairs) {
        int i = p.first, j = p.second;
        Eigen::Vector2d measCenter = (Eigen::Vector2d(landmarks[i].x, landmarks[i].y)
                                      + Eigen::Vector2d(landmarks[j].x, landmarks[j].y))
            * 0.5;

        auto* edgeSym = new EdgeSymmetry(
            R_camera_imu,
            R_pitch,
            t_camera_armor,
            K_,
            object_points[i],
            object_points[j],
            measCenter,
            symWeight
        );
        edgeSym->setId(id_counter++);
        edgeSym->setVertex(0, v_yaw);
        optimizer_.addEdge(edgeSym);
    }

    // 优化
    optimizer_.initializeOptimization();
    optimizer_.optimize(1000);

    double yaw_opt = v_yaw->estimate();
    if (std::isnan(yaw_opt)) {
        std::cerr << "Yaw is nan after g2o optimization\n";
        return R_camera_armor;
    }

    Sophus::SO3d R_yaw = Sophus::SO3d::exp(Eigen::Vector3d(0, 0, yaw_opt));
    return (R_camera_imu * R_yaw * R_pitch).matrix();
}
Eigen::Vector3d BaSolver::solveBa_t(
    const ArmorObject& armor,
    const Eigen::Vector3d& t_camera_armor_init,
    const Eigen::Matrix3d& R_camera_armor, // 已优化的旋转
    const Eigen::Matrix3d& /*unused R_imu_camera*/,
    int type_number
) noexcept {
    optimizer_.clear();

    // 1) 直接使用传入的 R_camera_armor，视为 world->camera 的完整旋转
    Sophus::SO3d R_fixed = Sophus::SO3d(R_camera_armor);

    // 2) 构建装甲板 3D 点
    Eigen::Vector2d armor_size;
    switch (type_number) {
        case 0:
            armor_size = { LARGE_ARMOR_WIDTH, LARGE_ARMOR_HEIGHT };
            break;
        case 1:
            armor_size = { SMALL_ARMOR_WIDTH, SMALL_ARMOR_HEIGHT };
            break;
        case 2:
            armor_size = { LARGE_ARMOR_WIDTH_NET, LARGE_ARMOR_HEIGHT_NET };
            break;
        case 3:
            armor_size = { SMALL_ARMOR_WIDTH_NET, SMALL_ARMOR_HEIGHT_NET };
            break;
        default:
            armor_size = { LARGE_ARMOR_WIDTH, LARGE_ARMOR_HEIGHT };
    }
    const auto object_points =
        ArmorObject::buildObjectPoints<Eigen::Vector3d>(armor_size.x(), armor_size.y());
    const auto& landmarks = armor.landmarks();

    // 3) 添加优化器顶点：tvec
    size_t id = 0;
    auto* v_tvec = new VertexTranslation();
    v_tvec->setId(id++);
    v_tvec->setEstimate(t_camera_armor_init);
    optimizer_.addVertex(v_tvec);

    // 4) 添加投影边（仅依赖 tvec）
    for (size_t i = 0; i < object_points.size(); ++i) {
        auto* edge = new EdgeProjectionTvecOnly(R_fixed, object_points[i], K_);
        edge->setId(id++);
        edge->setVertex(0, v_tvec);
        edge->setMeasurement({ landmarks[i].x, landmarks[i].y });
        edge->setInformation(Eigen::Matrix2d::Identity());
        edge->setRobustKernel(new g2o::RobustKernelHuber);
        optimizer_.addEdge(edge);
    }

    // 5) 添加对称误差边（如果需要）
    std::vector<std::pair<int, int>> sym { { 0, 5 }, { 1, 4 }, { 2, 3 } };
    const double symWeight = 1.0;
    for (auto& [i, j]: sym) {
        Eigen::Vector2d meas_center = 0.5
            * (Eigen::Vector2d(landmarks[i].x, landmarks[i].y)
               + Eigen::Vector2d(landmarks[j].x, landmarks[j].y));
        auto* es = new EdgeSymmetryTvecOnly(
            R_fixed,
            object_points[i],
            object_points[j],
            meas_center,
            K_,
            symWeight
        );
        es->setId(id++);
        es->setVertex(0, v_tvec);
        optimizer_.addEdge(es);
    }

    // 6) 执行优化
    optimizer_.initializeOptimization();
    optimizer_.optimize(100);

    // 7) 获取并返回优化后的 tvec
    Eigen::Vector3d t_opt = v_tvec->estimate();
    if (t_opt.hasNaN()) {
        std::cerr << "ERROR: optimized tvec is NaN, return init\n";
        return t_camera_armor_init;
    }
    return t_opt;
}
