// Created by Chengfu Zou on 2024.1.19
// Copyright(C) FYT Vision Group. All rights resevred.
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

inline void transformArmorData(Armors& armors, std::string target_frame_) {
    for (auto& armor: armors.armors) {
        try {
            tf::Transform tf(armor.pos, armor.ori, armors.timestamp);
            auto pose_in_target_frame =
                gobal::tf_tree_.transform(tf, armors.frame_id, target_frame_, armors.timestamp);

            armor.target_pos = pose_in_target_frame.position;
            armor.target_ori = pose_in_target_frame.orientation;

            armor.yaw = getRPYFromQuaternion(armor.target_ori).yaw;
            double yaw = armor.yaw * 180 / M_PI;

        } catch (const std::exception& e) {
            WUST_ERROR("tf") << "Can't find transform from " << armors.frame_id << " to "
                             << target_frame_ << ": " << e.what();
            return;
        }
    }
}
inline void transformArmorData(Armors& armors, Eigen::Matrix4d T_camera_to_odom) {
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
            // std::cout << "x" << armor.target_pos.x<< "y"<< armor.target_pos.y <<
            // "z"<< armor.target_pos.z << std::endl;

            // Step 3: Extract yaw (assuming you have a function like this)
            Eigen::Vector3d euler = R_ori_odom.eulerAngles(2, 1, 0); // ZYX
            armor.yaw = euler[0]; // yaw

        } catch (const std::exception& e) {
            WUST_ERROR("tf") << "Error in camera-to-odom transform: " << e.what();
            return;
        }
    }
}
inline void transformArmorData(Armor& armor, Eigen::Matrix4d T_camera_to_odom) {
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
        // std::cout << "x" << armor.target_pos.x<< "y"<< armor.target_pos.y <<
        // "z"<< armor.target_pos.z << std::endl;

        // Step 3: Extract yaw (assuming you have a function like this)
        Eigen::Vector3d euler = R_ori_odom.eulerAngles(2, 1, 0); // ZYX
        armor.yaw = euler[0]; // yaw

    } catch (const std::exception& e) {
        WUST_ERROR("tf") << "Error in camera-to-odom transform: " << e.what();
        return;
    }
}

} // namespace utils
