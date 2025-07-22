// Created by Chengfu Zou on 2024.1.19
// Copyright(C) FYT Vision Group. All rights resevred.
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

#include "common/gobal.hpp"
#include "common/logger.hpp"
#include "control/manual_compensator.hpp"
#include "type/type.hpp"
#include <Eigen/Dense>
#include <opencv2/calib3d.hpp>
#include <opencv2/core.hpp>
#include <opencv2/core/eigen.hpp>

// util functions
namespace utils {
// Convert euler angles to rotation matrix
enum class EulerOrder { XYZ, XZY, YXZ, YZX, ZXY, ZYX };
template<typename Vec3Like>
Eigen::Matrix3d eulerToMatrix(const Vec3Like& euler, EulerOrder order = EulerOrder::XYZ) {
    auto r = Eigen::AngleAxisd(euler[0], Eigen::Vector3d::UnitX());
    auto p = Eigen::AngleAxisd(euler[1], Eigen::Vector3d::UnitY());
    auto y = Eigen::AngleAxisd(euler[2], Eigen::Vector3d::UnitZ());
    switch (order) {
        case EulerOrder::XYZ:
            return (y * p * r).matrix();
        case EulerOrder::XZY:
            return (p * y * r).matrix();
        case EulerOrder::YXZ:
            return (y * r * p).matrix();
        case EulerOrder::YZX:
            return (r * y * p).matrix();
        case EulerOrder::ZXY:
            return (p * r * y).matrix();
        case EulerOrder::ZYX:
            return (r * p * y).matrix();
    }
}

inline Eigen::Vector3d
matrixToEuler(const Eigen::Matrix3d& R, EulerOrder order = EulerOrder::XYZ) noexcept {
    switch (order) {
        case EulerOrder::XYZ:
            return R.eulerAngles(0, 1, 2);
        case EulerOrder::XZY:
            return R.eulerAngles(0, 2, 1);
        case EulerOrder::YXZ:
            return R.eulerAngles(1, 0, 2);
        case EulerOrder::YZX:
            return R.eulerAngles(1, 2, 0);
        case EulerOrder::ZXY:
            return R.eulerAngles(2, 0, 1);
        case EulerOrder::ZYX:
            return R.eulerAngles(2, 1, 0);
    }
}

inline Eigen::Vector3d getRPY(const Eigen::Matrix3d& R) {
    double yaw = atan2(R(0, 1), R(0, 0));
    double c2 = Eigen::Vector2d(R(2, 2), R(1, 2)).norm();
    double pitch = atan2(-R(0, 2), c2);

    double s1 = sin(yaw);
    double c1 = cos(yaw);
    double roll = atan2(s1 * R(2, 0) - c1 * R(2, 1), c1 * R(1, 1) - s1 * R(1, 0));

    return -Eigen::Vector3d(roll, pitch, yaw);
}

template<typename _Tp, int _rows, int _cols, int _options, int _maxRows, int _maxCols>
cv::Mat eigenToCv(const Eigen::Matrix<_Tp, _rows, _cols, _options, _maxRows, _maxCols>& eigen_mat) {
    cv::Mat cv_mat;
    cv::eigen2cv(eigen_mat, cv_mat);
    return cv_mat;
}

inline Eigen::MatrixXd cvToEigen(const cv::Mat& cv_mat) noexcept {
    Eigen::MatrixXd eigen_mat = Eigen::MatrixXd::Zero(cv_mat.rows, cv_mat.cols);
    cv::cv2eigen(cv_mat, eigen_mat);
    return eigen_mat;
}

inline void transformArmorData(armor::Armors& armors, Eigen::Matrix4d T_camera_to_odom) {
    for (auto& armor: armors.armors) {
        try {
            // Step 1: Transform position from camera to odom
            Eigen::Vector4d pos_camera(armor.pos.x, armor.pos.y, armor.pos.z, 1.0);
            Eigen::Vector4d pos_odom = T_camera_to_odom * pos_camera;

            armor.target_pos.x = pos_odom.x();
            armor.target_pos.y = pos_odom.y();
            armor.target_pos.z = pos_odom.z();

            // Step 2: Transform orientation from camera to odom
            Eigen::Matrix3d R_camera_to_odom = T_camera_to_odom.block<3, 3>(0, 0);
            Eigen::Quaterniond q_camera(armor.ori.w, armor.ori.x, armor.ori.y, armor.ori.z);
            Eigen::Matrix3d R_ori_camera = q_camera.normalized().toRotationMatrix();

            Eigen::Matrix3d R_ori_odom = R_camera_to_odom * R_ori_camera;
            Eigen::Quaterniond q_odom(R_ori_odom);

            armor.target_ori.w = q_odom.w();
            armor.target_ori.x = q_odom.x();
            armor.target_ori.y = q_odom.y();
            armor.target_ori.z = q_odom.z();

            // Step 3: Extract yaw (assuming you have a function like this)
            Eigen::Vector3d euler = R_ori_odom.eulerAngles(2, 1, 0); // ZYX
            armor.yaw = euler[0]; // yaw

        } catch (const std::exception& e) {
            WUST_ERROR("tf") << "Error in camera-to-odom transform: " << e.what();
            return;
        }
    }
}
inline void transformArmorData(armor::Armor& armor, Eigen::Matrix4d T_camera_to_odom) {
    try {
        // Step 1: Transform position from camera to odom
        Eigen::Vector4d pos_camera(armor.pos.x, armor.pos.y, armor.pos.z, 1.0);
        Eigen::Vector4d pos_odom = T_camera_to_odom * pos_camera;

        armor.target_pos.x = pos_odom.x();
        armor.target_pos.y = pos_odom.y();
        armor.target_pos.z = pos_odom.z();

        // Step 2: Transform orientation from camera to odom
        Eigen::Matrix3d R_camera_to_odom = T_camera_to_odom.block<3, 3>(0, 0);
        Eigen::Quaterniond q_camera(armor.ori.w, armor.ori.x, armor.ori.y, armor.ori.z);
        Eigen::Matrix3d R_ori_camera = q_camera.normalized().toRotationMatrix();

        Eigen::Matrix3d R_ori_odom = R_camera_to_odom * R_ori_camera;
        Eigen::Quaterniond q_odom(R_ori_odom);

        armor.target_ori.w = q_odom.w();
        armor.target_ori.x = q_odom.x();
        armor.target_ori.y = q_odom.y();
        armor.target_ori.z = q_odom.z();

        // Step 3: Extract yaw (assuming you have a function like this)
        Eigen::Vector3d euler = R_ori_odom.eulerAngles(2, 1, 0); // ZYX
        armor.yaw = euler[0]; // yaw

    } catch (const std::exception& e) {
        WUST_ERROR("tf") << "Error in camera-to-odom transform: " << e.what();
        return;
    }
}
inline double getNoiseFromCameraYaw(double camera_yaw_deg, double r_front, double r_side) {
    double yaw_rad = camera_yaw_deg * M_PI / 180.0;
    double cos2 = std::cos(yaw_rad);
    cos2 *= cos2;
    return cos2 * r_front + (1.0 - cos2) * r_side;
}
inline double getNoiseVarFromCameraYaw(double camera_yaw_deg, double r_front, double r_side) {
    double noise_deg = getNoiseFromCameraYaw(camera_yaw_deg, r_front, r_side);
    return std::pow(noise_deg * M_PI / 180.0, 2);
}
inline cv::Point2f computeCenter(const std::vector<cv::Point2f>& points) {
    if (points.empty()) {
        return cv::Point2f(0.f, 0.f);
    }

    float sum_x = 0.f;
    float sum_y = 0.f;
    for (const auto& pt: points) {
        sum_x += pt.x;
        sum_y += pt.y;
    }
    return cv::Point2f(sum_x / points.size(), sum_y / points.size());
}
inline bool isStateValid(const Eigen::VectorXd& state) {
    return state.allFinite(); // 所有元素都不是 NaN 或 Inf
}
inline double clamp_pm_pi(auto&& angle) {
    while (angle >= M_PI)
        angle -= M_PI;
    while (angle <= -M_PI)
        angle += M_PI;

    return angle;
}
inline double ratio(const auto& point) {
    return atan2(point.y, point.x);
}
inline bool
checkTargetAppear(const armor::Target& target, const std::vector<armor::OneTarget>& one_targets) {
    if (target.tracking)
        return true;
    return std::any_of(one_targets.begin(), one_targets.end(), [](const armor::OneTarget& t) {
        return t.tracking;
    });
}
inline Eigen::Matrix4d computeCameraToOdomTransform(
    const Eigen::Matrix3d& R_gimbal2odom,
    const Eigen::Matrix3d& R_camera_to_gimbal,
    const Eigen::Vector3d& t_gimbal_to_camera
) {
    Eigen::Matrix4d T_gimbal_to_odom = Eigen::Matrix4d::Identity();
    T_gimbal_to_odom.block<3, 3>(0, 0) = R_gimbal2odom;

    Eigen::Vector3d t_camera_to_gimbal = -R_camera_to_gimbal * t_gimbal_to_camera;

    Eigen::Matrix4d T_camera_to_gimbal = Eigen::Matrix4d::Identity();
    T_camera_to_gimbal.block<3, 3>(0, 0) = R_camera_to_gimbal;
    T_camera_to_gimbal.block<3, 1>(0, 3) = t_camera_to_gimbal;

    Eigen::Matrix4d T_camera_to_odom = T_gimbal_to_odom * T_camera_to_gimbal;

    return T_camera_to_odom;
}
inline Eigen::Matrix3d getRGimbalToOdom(
    const Eigen::Matrix4d& T_camera_to_odom,
    const Eigen::Matrix3d& R_camera2gimbal,
    const Eigen::Vector3d& t_gimbal_to_camera
) {
    Eigen::Matrix3d R_camera_to_gimbal = R_camera2gimbal;
    Eigen::Vector3d t_camera_to_gimbal = -R_camera_to_gimbal * t_gimbal_to_camera;

    Eigen::Matrix4d T_camera_to_gimbal = Eigen::Matrix4d::Identity();
    T_camera_to_gimbal.block<3, 3>(0, 0) = R_camera_to_gimbal;
    T_camera_to_gimbal.block<3, 1>(0, 3) = t_camera_to_gimbal;

    Eigen::Matrix4d T_gimbal_to_camera = T_camera_to_gimbal.inverse();
    Eigen::Matrix4d T_gimbal_to_odom = T_camera_to_odom * T_gimbal_to_camera;
    Eigen::Matrix3d R_gimbal2odom = T_gimbal_to_odom.block<3, 3>(0, 0);
    return R_gimbal2odom;
}
inline void addVelFromAccDt(tf::Position& vel, const tf::Position& acc, double dt) {
    vel.x += acc.x * dt;
    vel.y += acc.y * dt;
    vel.z += acc.z * dt;
}
inline void addPosFromVelDt(tf::Position pos, const tf::Position& vel, double dt) {
    pos.x += vel.x * dt;
    pos.y += vel.y * dt;
    pos.z += vel.z * dt;
}
template<typename T>
bool tryGetValue(const YAML::Node& node, const char* key, T& out_val) {
    if (!node[key]) {
        return false;
    }
    try {
        out_val = node[key].template as<T>();
        return true;
    } catch (...) {
        return false;
    }
}
inline std::vector<OffsetEntry> getOffsetEntry(const YAML::Node& config) {
    std::vector<OffsetEntry> entries;
    if (config["trajectory_offset"]) {
        for (const auto& node: config["trajectory_offset"]) {
            OffsetEntry e;
            bool valid = true;

            auto tryGetValue = [](const YAML::Node& n, const char* key, double& out_val) -> bool {
                if (!n[key])
                    return false;
                try {
                    out_val = n[key].template as<double>();
                    return true;
                } catch (...) {
                    return false;
                }
            };

            valid &= tryGetValue(node, "d_min", e.d_min);
            valid &= tryGetValue(node, "d_max", e.d_max);
            valid &= tryGetValue(node, "h_min", e.h_min);
            valid &= tryGetValue(node, "h_max", e.h_max);
            valid &= tryGetValue(node, "pitch_off", e.pitch_off);
            valid &= tryGetValue(node, "yaw_off", e.yaw_off);

            if (valid) {
                entries.push_back(e);
            } else {
                std::cerr
                    << "Warning: skipping invalid trajectory_offset entry due to missing or invalid fields."
                    << std::endl;
            }
        }
    }
    return entries;
}
} // namespace utils
