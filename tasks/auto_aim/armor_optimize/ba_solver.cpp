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
#include <opencv2/core/eigen.hpp>
Eigen::Vector<double, 5> toEigenDist(const cv::Mat& dist) {
    Eigen::Vector<double, 5> d = Eigen::Vector<double, 5>::Zero();

    if (dist.total() < 5) {
        return d; // 如果参数不足，返回 0 避免异常
    }

    if (dist.type() == CV_32F) {
        const float* p = dist.ptr<float>();
        for (int i = 0; i < 5; i++)
            d[i] = static_cast<double>(p[i]);
    } else if (dist.type() == CV_64F) {
        const double* p = dist.ptr<double>();
        for (int i = 0; i < 5; i++)
            d[i] = p[i];
    } else {
        // 兼容其他类型，逐个读取
        for (int i = 0; i < 5; i++)
            d[i] = dist.at<double>(i, 0);
    }

    return d;
}
BaSolver::BaSolver(const YAML::Node& config, const std::pair<cv::Mat, cv::Mat>& camera_info) {
    camera_info_ = camera_info;
    cv::cv2eigen(camera_info.first, K_);
    dist_eigen_ = toEigenDist(camera_info.second);
    params_.load(config);
}

std::vector<Eigen::Vector2d> reprojectionArmor(
    double yaw,
    const std::vector<Eigen::Vector3d>& object_points,
    const std::vector<cv::Point2f>& landmarks,
    const Eigen::Matrix3d& Rci,
    double pitch,
    double roll,
    const Eigen::Vector3d& t,
    const Eigen::Matrix3d& K,
    const cv::Mat& dist // 畸变参数
) {
    // 组合旋转（相机系到目标系）
    Eigen::AngleAxisd ay(yaw, Eigen::Vector3d::UnitZ());
    Eigen::AngleAxisd ap(pitch, Eigen::Vector3d::UnitY());
    Eigen::AngleAxisd ar(roll, Eigen::Vector3d::UnitX());
    Eigen::Matrix3d R = Rci * (ay * ap * ar).toRotationMatrix();

    // 转成 OpenCV rvec/tvec
    cv::Mat rvec, R_cv;
    cv::eigen2cv(R, R_cv);
    cv::Rodrigues(R_cv, rvec);

    cv::Mat tvec = (cv::Mat_<double>(3, 1) << t.x(), t.y(), t.z());

    // 取内参
    cv::Mat K_cv;
    cv::eigen2cv(K, K_cv);

    // 3D点转 cv::Point3f
    std::vector<cv::Point3f> obj_pts;
    obj_pts.reserve(object_points.size());
    for (const auto& p: object_points) {
        obj_pts.emplace_back(p.x(), p.y(), p.z());
    }

    // 投影（带畸变）
    std::vector<cv::Point2f> pts_2d;
    pts_2d.reserve(obj_pts.size());
    cv::projectPoints(obj_pts, rvec, tvec, K_cv, dist, pts_2d);

    // 输出到 Eigen，并做 NaN/越界/深度保护
    std::vector<Eigen::Vector2d> image_points;
    image_points.reserve(pts_2d.size());

    for (const auto& p: pts_2d) {
        if (!std::isfinite(p.x) || !std::isfinite(p.y)) {
            continue; // 过滤异常点
        }
        image_points.emplace_back(p.x, p.y);
    }

    return image_points;
}

double reprojectionErrorYaw(
    double yaw,
    const std::vector<Eigen::Vector3d>& object_points,
    const std::vector<cv::Point2f>& landmarks,
    const Eigen::Matrix3d& Rci,
    double pitch,
    double roll,
    const Eigen::Vector3d& t,
    const Eigen::Matrix3d& K,
    const cv::Mat& dist
) {
    auto image_points =
        reprojectionArmor(yaw, object_points, landmarks, Rci, pitch, roll, t, K, dist);
    double cost = 0.0;

    // for (size_t i = 0; i < image_points.size(); ++i) {
    //     Eigen::Vector2d obs(landmarks[i].x, landmarks[i].y);
    //     cost += (image_points[i] - obs).squaredNorm();
    // }
    auto buildSymPairs = [&](size_t n) {
        std::vector<std::pair<int, int>> pairs;
        for (int i = 0; i < n / 2; ++i) {
            pairs.emplace_back(i, n - 1 - i);
        }
        return pairs;
    };
    const double symWeight = 3.0;
    const auto symPairs = buildSymPairs(object_points.size());

    for (auto& p: symPairs) {
        Eigen::Vector2d mid = 0.5 * (image_points[p.first] + image_points[p.second]);

        Eigen::Vector2d meas = 0.5
            * (Eigen::Vector2d(landmarks[p.first].x, landmarks[p.first].y)
               + Eigen::Vector2d(landmarks[p.second].x, landmarks[p.second].y));

        cost += symWeight * (mid - meas).squaredNorm();
    }
    return cost;
}

double BaSolver::goldenYaw(
    double init,
    const std::vector<Eigen::Vector3d>& obj,
    const std::vector<cv::Point2f>& lm,
    const Eigen::Matrix3d& Rci,
    double pitch,
    double roll,
    const Eigen::Vector3d& t,
    const Eigen::Matrix3d& K
) {
    constexpr double phi = 1.618033988749894848; //(1.0 + std::sqrt(5.0)) * 0.5;
    double l = init - params_.golden_search_side_deg * M_PI / 180.0;
    double r = init + params_.golden_search_side_deg * M_PI / 180.0;

    double y1 = r - (r - l) / phi;
    double y2 = l + (r - l) / phi;

    double f1 = reprojectionErrorYaw(y1, obj, lm, Rci, pitch, roll, t, K, camera_info_.second);
    double f2 = reprojectionErrorYaw(y2, obj, lm, Rci, pitch, roll, t, K, camera_info_.second);

    while (r - l > 0.0001) { // stop threshold ~0.005 degree
        if (f1 < f2) {
            r = y2;
            y2 = y1;
            f2 = f1;
            y1 = r - (r - l) / phi;
            f1 = reprojectionErrorYaw(y1, obj, lm, Rci, pitch, roll, t, K, camera_info_.second);
        } else {
            l = y1;
            y1 = y2;
            f1 = f2;
            y2 = l + (r - l) / phi;
            f2 = reprojectionErrorYaw(y2, obj, lm, Rci, pitch, roll, t, K, camera_info_.second);
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
    double roll,
    const Eigen::Vector3d& t_camera_armor,
    const Eigen::Matrix3d& K
) {
    double yaw = initial_yaw;

    CameraProjector cam(R_camera_imu, armor_pitch, roll, t_camera_armor, K, dist_eigen_);

    ceres::Problem problem;
    problem.AddParameterBlock(&yaw, 1, new YawLocalParameterization());

    // for (size_t i = 0; i < object_points.size(); ++i) {
    //     ceres::CostFunction* cost =
    //         new ceres::AutoDiffCostFunction<ReprojectionError, 2, 1>(new ReprojectionError(
    //             Eigen::Vector2d(landmarks[i].x, landmarks[i].y),
    //             object_points[i],
    //             cam
    //         ));

    //     problem.AddResidualBlock(cost, new ceres::HuberLoss(1.0), &yaw);
    // }
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
        ceres::CostFunction* cost = new ceres::AutoDiffCostFunction<SymmetryError, 2, 1>(
            new SymmetryError(object_points[p.first], object_points[p.second], meas, cam)
        );
        problem.AddResidualBlock(cost, new ceres::HuberLoss(1.0), &yaw);
    }

    ceres::Solver::Options options;
    options.max_num_iterations = params_.ceres_max_iter;
    options.linear_solver_type = ceres::DENSE_QR;
    options.minimizer_progress_to_stdout = false;

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
    //double roll = std::atan2(R_imu_armor(2, 1), R_imu_armor(2, 2));
    double roll = 0;
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
        yaw = ceresYaw(yaw_init, objPts, lm, R_camera_imu, armor_pitch, roll, t_camera_armor, K);
    } else if (params_.mode == Params::OptMode::GOLDEN) {
        yaw = goldenYaw(yaw_init, objPts, lm, R_camera_imu, armor_pitch, roll, t_camera_armor, K);
    }

    // build yaw + pitch rotation
    double cy = std::cos(yaw), sy = std::sin(yaw);
    Eigen::Matrix3d R_yaw;
    R_yaw << cy, -sy, 0, sy, cy, 0, 0, 0, 1;

    double cp = std::cos(armor_pitch), sp = std::sin(armor_pitch);
    Eigen::Matrix3d R_pitch;
    R_pitch << cp, 0, sp, 0, 1, 0, -sp, 0, cp;

    double cr = std::cos(roll), sr = std::sin(roll);
    Eigen::Matrix3d R_roll;
    R_roll << cr, -sr, 0, sr, cr, 0, 0, 0, 1;

    Eigen::Matrix3d R_result = R_camera_imu * R_yaw * R_pitch * R_roll;
    return R_result;
}
