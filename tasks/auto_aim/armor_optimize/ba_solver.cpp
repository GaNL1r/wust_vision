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
#include "graph_optimizer.hpp"
#include "tasks/utils.hpp"
#include "wust_vl/common/utils/logger.hpp"

G2O_USE_OPTIMIZATION_LIBRARY(dense)

BaSolver::BaSolver(
    std::array<double, 9>& camera_matrix,
    int max_iter_R,
    int max_iter_t,
    int step_R,
    int step_t,
    double min_error_R,
    double min_error_t
):
    max_iter_R_(max_iter_R),
    max_iter_t_(max_iter_t),
    step_R_(step_R),
    step_t_(step_t),
    min_error_R_(min_error_R),
    min_error_t_(min_error_t) {
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
constexpr double SEARCH_RANGE_DEG = 140.0;

std::vector<cv::Point2d> reproject_armor(
    const Eigen::Matrix3d& R_camera_armor, // camera <- armor
    const Eigen::Vector3d& t_camera_armor, // translation camera <- armor
    const std::string& type,
    const cv::Mat& camera_matrix_in,
    const cv::Mat& distort_coeffs_in
) {
    // Convert rotation (Eigen -> cv::Mat) and get Rodrigues vector
    cv::Mat R_cv;
    cv::eigen2cv(R_camera_armor, R_cv);
    cv::Mat rvec;
    cv::Rodrigues(R_cv, rvec); // rvec will be 3x1 CV_64F

    // translation as 3x1 CV_64F
    cv::Mat tvec =
        (cv::Mat_<double>(3, 1) << t_camera_armor.x(), t_camera_armor.y(), t_camera_armor.z());

    // Force camera params to CV_64F (avoid implicit mixed-depth issues)
    cv::Mat cam_mat, dist;
    camera_matrix_in.convertTo(cam_mat, CV_64F);
    distort_coeffs_in.convertTo(dist, CV_64F);

    // Armor size
    Eigen::Vector2d armor_size = (type == "large")
        ? Eigen::Vector2d(LARGE_ARMOR_WIDTH, LARGE_ARMOR_HEIGHT)
        : Eigen::Vector2d(SMALL_ARMOR_WIDTH, SMALL_ARMOR_HEIGHT);

    // Build object points (Eigen -> vector<cv::Point3d>)
    auto object_points =
        armor::ArmorObject::buildObjectPoints<Eigen::Vector3d>(armor_size.x(), armor_size.y());
    std::vector<cv::Point3d> cv_object_points;
    cv_object_points.reserve(object_points.size());
    for (const auto& p: object_points) {
        cv_object_points.emplace_back(p.x(), p.y(), p.z());
    }

    // Basic sanity checks
    if (cam_mat.rows != 3 || cam_mat.cols != 3) {
        std::cerr << "[reproject_armor] camera_matrix not 3x3\n";
        return {};
    }
    if (rvec.empty() || tvec.empty()) {
        std::cerr << "[reproject_armor] empty rvec/tvec\n";
        return {};
    }

    // Reproject
    std::vector<cv::Point2d> image_points;
    try {
        cv::projectPoints(cv_object_points, rvec, tvec, cam_mat, dist, image_points);
    } catch (const cv::Exception& e) {
        std::cerr << "[reproject_armor] OpenCV exception: " << e.what() << std::endl;
        return {};
    }

    // sanity: check finite
    for (size_t i = 0; i < image_points.size(); ++i) {
        if (!std::isfinite(image_points[i].x) || !std::isfinite(image_points[i].y)) {
            std::cerr << "[reproject_armor] got non-finite proj pt idx=" << i << " ("
                      << image_points[i].x << "," << image_points[i].y << ")\n";
            return {};
        }
    }

    return image_points;
}

// Eigen::Matrix3d BaSolver::solveBa_R(
//     const armor::ArmorObject& armor,
//     const Eigen::Vector3d& t_camera_armor,
//     const Eigen::Matrix3d& R_camera_armor, // camera <- armor
//     const Eigen::Matrix3d& R_imu_camera, // imu -> camera
//     const std::string& type,
//     const cv::Mat& camera_matrix,
//     const cv::Mat& distort_coeffs
// ) noexcept {
//     // ---- NOTE ON CHAINING ----
//     // You must ensure R_camera_armor and R_imu_camera have the meanings above.
//     // If R_imu_camera is (imu -> camera) and R_camera_armor is (camera -> armor),
//     // then armor -> imu can be computed as:
//     //   R_armor_imu = (R_camera_armor * R_imu_camera).transpose();
//     // or equivalently:
//     //   R_armor_imu = R_imu_camera.transpose() * R_camera_armor.transpose();
//     // If your conventions differ, adjust accordingly.
//     Eigen::Matrix3d R_armor_imu = (R_camera_armor * R_imu_camera).transpose();

//     // Convert to Euler (2,1,0) as before (matrixToEuler must return [yaw, pitch, roll] under your util)
//     auto eulers = utils::matrixToEuler(R_armor_imu, 2, 1, 0);
//     double raw_yaw = eulers[0];

//     // choose nominal pitch based on armor type
//     double pitch = (armor.number == armor::ArmorNumber::OUTPOST) ? -15.0 * CV_PI / 180.0
//                                                                  : 15.0 * CV_PI / 180.0;
//     double sin_pitch = std::sin(pitch);
//     double cos_pitch = std::cos(pitch);

//     double min_error = 1e10;
//     double best_yaw = raw_yaw;

//     // compute yaw search start (normalize_angle: keep in [-pi,pi] or your convention)
//     double yaw_start = angles::normalize_angle(raw_yaw - (SEARCH_RANGE_DEG / 2.0) * CV_PI / 180.0);

//     const auto& landmarks = armor.landmarks();

//     // Pre-convert camera matrices once to CV_64F to avoid repeated conversions
//     cv::Mat cam_mat, dist;
//     camera_matrix.convertTo(cam_mat, CV_64F);
//     distort_coeffs.convertTo(dist, CV_64F);

//     for (int i = 0; i < static_cast<int>(SEARCH_RANGE_DEG); ++i) {
//         double yaw = angles::normalize_angle(yaw_start + i * CV_PI / 180.0);
//         double sin_yaw = std::sin(yaw);
//         double cos_yaw = std::cos(yaw);

//         // build rotation of armor in *world* (as you had)
//         const Eigen::Matrix3d R_armor2imu { { cos_yaw * cos_pitch, -sin_yaw, cos_yaw * sin_pitch },
//                                             { sin_yaw * cos_pitch, cos_yaw, sin_yaw * sin_pitch },
//                                             { -sin_pitch, 0.0, cos_pitch } };

//         // armor -> camera
//         Eigen::Matrix3d R_armor2camera = R_imu_camera * R_armor2imu;

//         // pass camera <- armor into reproject_armor
//         Eigen::Matrix3d R_camera2armor = R_armor2camera.transpose();

//         // reproject (returns double points)
//         auto image_points = reproject_armor(R_camera2armor, t_camera_armor, type, cam_mat, dist);

//         // if reproject failed (empty), skip iteration
//         if (image_points.empty()) {
//             std::cerr << "[solveBa_R] iter " << i << " reproject returned empty -> skip\n";
//             continue;
//         }

//         if (static_cast<int>(image_points.size()) != static_cast<int>(landmarks.size())) {
//             std::cerr << "[solveBa_R] iter " << i
//                       << " size mismatch: image_pts=" << image_points.size()
//                       << " landmarks=" << landmarks.size() << " -> skip\n";
//             continue;
//         }

//         // compute reprojection error (L2 sum)
//         double error = 0.0;
//         for (size_t j = 0; j < landmarks.size(); ++j) {
//             cv::Point2d lm_pt(landmarks[j].x, landmarks[j].y);
//             error += cv::norm(lm_pt - image_points[j]);
//         }

//         if (error < min_error) {
//             min_error = error;
//             best_yaw = yaw;
//         }
//         std::cout << "iter=" << i << " raw_yaw(deg)=" << (raw_yaw * 180.0 / CV_PI)
//                   << " best_yaw(deg)=" << (best_yaw * 180.0 / CV_PI) << " min_error=" << min_error
//                   << " img_pts=" << image_points.size() << std::endl;
//     }

//     // Build final rotation to return:
//     // R_camera_imu here we take as (camera <- imu) i.e., transpose of (imu -> camera)
//     Sophus::SO3d R_yaw = Sophus::SO3d::exp(Eigen::Vector3d(0.0, 0.0, best_yaw));
//     Sophus::SO3d R_pitch = Sophus::SO3d::exp(Eigen::Vector3d(0.0, pitch, 0.0));
//     Sophus::SO3d R_camera_imu(R_imu_camera.transpose());
//     Eigen::Matrix3d R_result = (R_camera_imu * R_yaw * R_pitch).matrix();

//     return R_result;
// }

Eigen::Matrix3d BaSolver::solveBa_R(
    const armor::ArmorObject& armor,
    const Eigen::Vector3d& t_camera_armor,
    const Eigen::Matrix3d& R_camera_armor, // camera <- armor
    const Eigen::Matrix3d& R_imu_camera, // imu -> camera
    const std::string& type,
    const cv::Mat& camera_matrix,
    const cv::Mat& distort_coeffs
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

    // 固定 pitch
    double armor_pitch =
        (armor.number == armor::ArmorNumber::OUTPOST) ? -FIFTTEN_DEGREE_RAD : FIFTTEN_DEGREE_RAD;
    Sophus::SO3d R_pitch = Sophus::SO3d::exp(Eigen::Vector3d(0, armor_pitch, 0));

    // 构建 3D 角点
    Eigen::Vector2d armor_size;
    if (type == "large") {
        armor_size = Eigen::Vector2d { LARGE_ARMOR_WIDTH, LARGE_ARMOR_HEIGHT };
    } else {
        armor_size = Eigen::Vector2d(SMALL_ARMOR_WIDTH, SMALL_ARMOR_HEIGHT);
    }

    auto object_points =
        armor::ArmorObject::buildObjectPoints<Eigen::Vector3d>(armor_size.x(), armor_size.y());
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
    double finalChi2 = std::numeric_limits<double>::infinity();
    int numEdges = optimizer_.edges().size();

    auto computeRMS = [&]() {
        // 用总 chi2 / 边数 开方得到 RMS reprojection error
        return std::sqrt(optimizer_.chi2() / numEdges);
    };
    for (int k = 0; k < max_iter_R_ / step_R_; ++k) {
        optimizer_.optimize(step_R_);
        finalChi2 = optimizer_.chi2();
        double rms = computeRMS();

        if (rms < min_error_R_) {
            break;
        }
    }
    double yaw_opt = v_yaw->estimate();
    if (std::isnan(yaw_opt)) {
        std::cerr << "Yaw is nan after g2o optimization\n";
        return R_camera_armor;
    }

    Sophus::SO3d R_yaw = Sophus::SO3d::exp(Eigen::Vector3d(0, 0, yaw_opt));
    return (R_camera_imu * R_yaw * R_pitch).matrix();
}
Eigen::Vector3d BaSolver::solveBa_t(
    const armor::ArmorObject& armor,
    const Eigen::Vector3d& t_camera_armor_init,
    const Eigen::Matrix3d& R_camera_armor,
    const Eigen::Matrix3d& /*unused R_imu_camera*/,
    const std::string type
) noexcept {
    optimizer_.clear();

    // 1) 直接使用传入的 R_camera_armor，视为 world->camera 的完整旋转
    Sophus::SO3d R_fixed = Sophus::SO3d(R_camera_armor);

    // 2) 构建装甲板 3D 点
    Eigen::Vector2d armor_size;
    if (type == "large") {
        armor_size = { LARGE_ARMOR_WIDTH, LARGE_ARMOR_HEIGHT };
    } else {
        armor_size = Eigen::Vector2d(SMALL_ARMOR_WIDTH, SMALL_ARMOR_HEIGHT);
    }
    const auto object_points =
        armor::ArmorObject::buildObjectPoints<Eigen::Vector3d>(armor_size.x(), armor_size.y());
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
    double finalChi2 = std::numeric_limits<double>::infinity();
    int numEdges = optimizer_.edges().size();

    auto computeRMS = [&]() {
        // 用总 chi2 / 边数 开方得到 RMS reprojection error
        return std::sqrt(optimizer_.chi2() / numEdges);
    };

    for (int k = 0; k < max_iter_t_ / step_t_; ++k) {
        optimizer_.optimize(step_t_);
        finalChi2 = optimizer_.chi2();
        double rms = computeRMS();
        if (rms < min_error_t_) {
            break;
        }
    }

    // 7) 获取并返回优化后的 tvec
    Eigen::Vector3d t_opt = v_tvec->estimate();
    if (t_opt.hasNaN()) {
        std::cerr << "ERROR: optimized tvec is NaN, return init\n";
        return t_camera_armor_init;
    }
    return t_opt;
}
Eigen::Vector3d BaSolver::solveBa_distanceOnly(
    const armor::ArmorObject& armor,
    const Eigen::Vector3d& t_camera_armor_init,
    const Eigen::Matrix3d& R_camera_armor, // 已优化的旋转
    const Eigen::Matrix3d& /*unused R_imu_camera*/,
    const std::string& type
) noexcept {
    optimizer_.clear();

    // 1) 固定旋转
    Sophus::SO3d R_fixed(R_camera_armor);

    // 2) 构建装甲板模型点
    Eigen::Vector2d armor_size;
    if (type == "large") {
        armor_size = { LARGE_ARMOR_WIDTH, LARGE_ARMOR_HEIGHT };
    } else {
        armor_size = { SMALL_ARMOR_WIDTH, SMALL_ARMOR_HEIGHT };
    }

    const auto object_points =
        armor::ArmorObject::buildObjectPoints<Eigen::Vector3d>(armor_size.x(), armor_size.y());
    const auto& landmarks = armor.landmarks();

    // 3) 构建 distance 优化顶点（t = distance * dir）
    Eigen::Vector3d dir = t_camera_armor_init.normalized();
    double d_init = t_camera_armor_init.norm();

    size_t id = 0;
    auto* v_dist = new VertexDistance(dir);
    v_dist->setId(id++);
    v_dist->setEstimate(d_init);
    optimizer_.addVertex(v_dist);

    // 4) 添加重投影边
    for (size_t i = 0; i < object_points.size(); ++i) {
        auto* edge = new EdgeProjectionDistanceOnly(R_fixed, object_points[i], K_);
        edge->setId(id++);
        edge->setVertex(0, v_dist);
        edge->setMeasurement({ landmarks[i].x, landmarks[i].y });
        edge->setInformation(Eigen::Matrix2d::Identity());
        edge->setRobustKernel(new g2o::RobustKernelHuber);
        optimizer_.addEdge(edge);
    }

    // 5) 添加对称误差边
    std::vector<std::pair<int, int>> sym { { 0, 5 }, { 1, 4 }, { 2, 3 } };
    const double symWeight = 1.0;

    for (auto& [i, j]: sym) {
        Eigen::Vector2d meas_center = 0.5
            * (Eigen::Vector2d(landmarks[i].x, landmarks[i].y)
               + Eigen::Vector2d(landmarks[j].x, landmarks[j].y));

        auto* es = new EdgeSymmetryDistanceOnly(
            R_fixed,
            object_points[i],
            object_points[j],
            meas_center,
            K_,
            symWeight
        );
        es->setId(id++);
        es->setVertex(0, v_dist);
        optimizer_.addEdge(es);
    }

    // 6) 优化流程
    optimizer_.initializeOptimization();
    double finalChi2 = std::numeric_limits<double>::infinity();
    int numEdges = optimizer_.edges().size();

    auto computeRMS = [&]() { return std::sqrt(optimizer_.chi2() / numEdges); };

    for (int k = 0; k < max_iter_t_ / step_t_; ++k) {
        optimizer_.optimize(step_t_);
        finalChi2 = optimizer_.chi2();
        double rms = computeRMS();
        if (rms < min_error_t_) {
            break;
        }
    }

    // 7) 返回最终优化结果
    Eigen::Vector3d t_opt = v_dist->translation();
    if (t_opt.hasNaN()) {
        std::cerr << "ERROR: optimized tvec is NaN, return init\n";
        return t_camera_armor_init;
    }
    return t_opt;
}