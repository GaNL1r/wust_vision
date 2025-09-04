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

#include "tasks/auto_aim/type.hpp"
#include "tasks/type_common.hpp"
#include "wust_vl/common/utils/logger.hpp"
#include "wust_vl/common/utils/manual_compensator.hpp"
#include <Eigen/Dense>
#include <opencv2/calib3d.hpp>
#include <opencv2/core.hpp>
#include <opencv2/core/eigen.hpp>
#include <pwd.h>
#include <wust_vl/video/image.hpp>
#include <yaml-cpp/node/node.h>

// util functions
namespace utils {
// Convert euler angles to rotation matrix
enum class EulerOrder { XYZ, XZY, YXZ, YZX, ZXY, ZYX };

inline double limit_rad(double angle) {
    while (angle > M_PI)
        angle -= 2.0 * M_PI;
    while (angle < -M_PI)
        angle += 2.0 * M_PI;
    return angle;
}

inline Eigen::Vector3d
quatToEuler(const Eigen::Quaterniond& q, int axis0, int axis1, int axis2, bool extrinsic = true) {
    if (!extrinsic)
        std::swap(axis0, axis2);

    auto i = axis0, j = axis1, k = axis2;
    bool is_proper = (i == k);
    if (is_proper)
        k = 3 - i - j;
    int sign = (i - j) * (j - k) * (k - i) / 2;

    double a, b, c, d;
    Eigen::Vector4d xyzw = q.coeffs(); // [x,y,z,w]
    if (is_proper) {
        a = xyzw[3];
        b = xyzw[i];
        c = xyzw[j];
        d = xyzw[k] * sign;
    } else {
        a = xyzw[3] - xyzw[j];
        b = xyzw[i] + xyzw[k] * sign;
        c = xyzw[j] + xyzw[3];
        d = xyzw[k] * sign - xyzw[i];
    }

    Eigen::Vector3d eulers;
    double n2 = a * a + b * b + c * c + d * d;
    eulers[1] = std::acos(2 * (a * a + b * b) / n2 - 1);

    double half_sum = std::atan2(b, a);
    double half_diff = std::atan2(-d, c);

    double eps = 1e-7;
    bool safe1 = std::abs(eulers[1]) >= eps;
    bool safe2 = std::abs(eulers[1] - M_PI) >= eps;
    bool safe = safe1 && safe2;

    if (safe) {
        eulers[0] = half_sum + half_diff;
        eulers[2] = half_sum - half_diff;
    } else {
        if (!extrinsic) {
            eulers[0] = 0;
            eulers[2] = !safe1 ? 2 * half_sum : -2 * half_diff;
        } else {
            eulers[2] = 0;
            eulers[0] = !safe1 ? 2 * half_sum : 2 * half_diff;
        }
    }

    for (int idx = 0; idx < 3; idx++)
        eulers[idx] = limit_rad(eulers[idx]);

    if (!is_proper) {
        eulers[2] *= sign;
        eulers[1] -= M_PI / 2;
    }

    if (!extrinsic)
        std::swap(eulers[0], eulers[2]);

    return eulers;
}

inline Eigen::Quaterniond
eulerToQuat(const Eigen::Vector3d& euler, int axis0, int axis1, int axis2, bool extrinsic = true) {
    double rz = euler[0], ry = euler[1], rx = euler[2];
    Eigen::Quaterniond qx(Eigen::AngleAxisd(rx, Eigen::Vector3d::UnitX()));
    Eigen::Quaterniond qy(Eigen::AngleAxisd(ry, Eigen::Vector3d::UnitY()));
    Eigen::Quaterniond qz(Eigen::AngleAxisd(rz, Eigen::Vector3d::UnitZ()));

    if (!extrinsic)
        std::swap(axis0, axis2);
    Eigen::Quaterniond q;

    // 生成四元数
    if (axis0 == 0 && axis1 == 1 && axis2 == 2)
        q = qx * qy * qz;
    else if (axis0 == 0 && axis1 == 2 && axis2 == 1)
        q = qx * qz * qy;
    else if (axis0 == 1 && axis1 == 0 && axis2 == 2)
        q = qy * qx * qz;
    else if (axis0 == 1 && axis1 == 2 && axis2 == 0)
        q = qy * qz * qx;
    else if (axis0 == 2 && axis1 == 0 && axis2 == 1)
        q = qz * qx * qy;
    else if (axis0 == 2 && axis1 == 1 && axis2 == 0)
        q = qz * qy * qx;
    else
        throw std::invalid_argument("Unsupported axis order");

    return q;
}

inline Eigen::Matrix3d eulerToMatrix(
    const Eigen::Vector3d& euler,
    int axis0,
    int axis1,
    int axis2,
    bool extrinsic = true
) {
    return eulerToQuat(euler, axis0, axis1, axis2, extrinsic).toRotationMatrix();
}

inline Eigen::Vector3d
matrixToEuler(const Eigen::Matrix3d& R, int axis0, int axis1, int axis2, bool extrinsic = true) {
    Eigen::Quaterniond q(R);
    return quatToEuler(q, axis0, axis1, axis2, extrinsic);
}

inline Eigen::Vector3d
quatToEuler(const Eigen::Quaterniond& q, EulerOrder order, bool extrinsic = true) {
    switch (order) {
        case EulerOrder::XYZ:
            return quatToEuler(q, 0, 1, 2, extrinsic);
        case EulerOrder::XZY:
            return quatToEuler(q, 0, 2, 1, extrinsic);
        case EulerOrder::YXZ:
            return quatToEuler(q, 1, 0, 2, extrinsic);
        case EulerOrder::YZX:
            return quatToEuler(q, 1, 2, 0, extrinsic);
        case EulerOrder::ZXY:
            return quatToEuler(q, 2, 0, 1, extrinsic);
        case EulerOrder::ZYX:
            return quatToEuler(q, 2, 1, 0, extrinsic);
        default:
            throw std::invalid_argument("Unsupported EulerOrder");
    }
}

inline Eigen::Quaterniond
eulerToQuat(const Eigen::Vector3d& euler, EulerOrder order, bool extrinsic = true) {
    switch (order) {
        case EulerOrder::XYZ:
            return eulerToQuat(euler, 0, 1, 2, extrinsic);
        case EulerOrder::XZY:
            return eulerToQuat(euler, 0, 2, 1, extrinsic);
        case EulerOrder::YXZ:
            return eulerToQuat(euler, 1, 0, 2, extrinsic);
        case EulerOrder::YZX:
            return eulerToQuat(euler, 1, 2, 0, extrinsic);
        case EulerOrder::ZXY:
            return eulerToQuat(euler, 2, 0, 1, extrinsic);
        case EulerOrder::ZYX:
            return eulerToQuat(euler, 2, 1, 0, extrinsic);
        default:
            throw std::invalid_argument("Unsupported EulerOrder");
    }
}

inline Eigen::Matrix3d
eulerToMatrix(const Eigen::Vector3d& euler, EulerOrder order, bool extrinsic = true) {
    return eulerToQuat(euler, order, extrinsic).toRotationMatrix();
}

inline Eigen::Vector3d
matrixToEuler(const Eigen::Matrix3d& R, EulerOrder order, bool extrinsic = true) {
    return quatToEuler(Eigen::Quaterniond(R), order, extrinsic);
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
            Eigen::Vector4d pos_camera(armor.pos.x(), armor.pos.y(), armor.pos.z(), 1.0);
            Eigen::Vector4d pos_odom = T_camera_to_odom * pos_camera;

            armor.target_pos.x() = pos_odom.x();
            armor.target_pos.y() = pos_odom.y();
            armor.target_pos.z() = pos_odom.z();

            // Step 2: Transform orientation from camera to odom
            Eigen::Matrix3d R_camera_to_odom = T_camera_to_odom.block<3, 3>(0, 0);
            Eigen::Quaterniond q_camera(armor.ori.w(), armor.ori.x(), armor.ori.y(), armor.ori.z());
            Eigen::Matrix3d R_ori_camera = q_camera.normalized().toRotationMatrix();

            Eigen::Matrix3d R_ori_odom = R_camera_to_odom * R_ori_camera;
            Eigen::Quaterniond q_odom(R_ori_odom);

            armor.target_ori.w() = q_odom.w();
            armor.target_ori.x() = q_odom.x();
            armor.target_ori.y() = q_odom.y();
            armor.target_ori.z() = q_odom.z();

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
        Eigen::Vector4d pos_camera(armor.pos.x(), armor.pos.y(), armor.pos.z(), 1.0);
        Eigen::Vector4d pos_odom = T_camera_to_odom * pos_camera;

        armor.target_pos.x() = pos_odom.x();
        armor.target_pos.y() = pos_odom.y();
        armor.target_pos.z() = pos_odom.z();

        // Step 2: Transform orientation from camera to odom
        Eigen::Matrix3d R_camera_to_odom = T_camera_to_odom.block<3, 3>(0, 0);
        Eigen::Quaterniond q_camera(armor.ori.w(), armor.ori.x(), armor.ori.y(), armor.ori.z());
        Eigen::Matrix3d R_ori_camera = q_camera.normalized().toRotationMatrix();

        Eigen::Matrix3d R_ori_odom = R_camera_to_odom * R_ori_camera;
        Eigen::Quaterniond q_odom(R_ori_odom);

        armor.target_ori.w() = q_odom.w();
        armor.target_ori.x() = q_odom.x();
        armor.target_ori.y() = q_odom.y();
        armor.target_ori.z() = q_odom.z();

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
    return noise_deg;
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
inline double ratio(const auto& point) {
    return atan2(point.y, point.x);
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
inline void addVelFromAccDt(Eigen::Vector3d& vel, const Eigen::Vector3d& acc, double dt) {
    vel.x() += acc.x() * dt;
    vel.y() += acc.y() * dt;
    vel.z() += acc.z() * dt;
}
inline void addPosFromVelDt(Eigen::Vector3d& pos, const Eigen::Vector3d& vel, double dt) {
    pos.x() += vel.x() * dt;
    pos.y() += vel.y() * dt;
    pos.z() += vel.z() * dt;
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
inline void changeFileOwner(const std::string& filepath, const std::string& username) {
    struct passwd* pwd = getpwnam(username.c_str());
    if (pwd == nullptr) {
        perror("getpwnam failed");
        return;
    }
    uid_t uid = pwd->pw_uid;
    gid_t gid = pwd->pw_gid;

    if (chown(filepath.c_str(), uid, gid) != 0) {
        perror("chown failed");
    }
}
inline std::string getOriginalUsername() {
    const char* sudo_user = std::getenv("SUDO_USER");
    if (sudo_user) {
        return std::string(sudo_user);
    }
    uid_t uid = getuid();
    struct passwd* pw = getpwuid(uid);
    if (pw) {
        return std::string(pw->pw_name);
    }
    return "";
}
inline bool
setThreadAffinityAndPriority(std::thread& thread, int cpu_id, int priority, bool use_sched_fifo) {
#ifdef __linux__
    pthread_t native = thread.native_handle();

    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    CPU_SET(cpu_id, &cpuset);
    if (pthread_setaffinity_np(native, sizeof(cpu_set_t), &cpuset) != 0) {
        perror("pthread_setaffinity_np failed");
        return false;
    }

    sched_param sch_params;
    sch_params.sched_priority = priority;
    int policy = use_sched_fifo ? SCHED_FIFO : SCHED_RR;
    if (pthread_setschedparam(native, policy, &sch_params) != 0) {
        perror("pthread_setschedparam failed");
        return false;
    }

    return true;

#elif defined(_WIN32) || defined(_WIN64)
    HANDLE native = (HANDLE)thread.native_handle();

    DWORD_PTR affinityMask = 1ULL << cpu_id;
    if (SetThreadAffinityMask(native, affinityMask) == 0) {
        return false;
    }

    int win_priority = THREAD_PRIORITY_HIGHEST; // you can map `priority` if needed
    if (!SetThreadPriority(native, win_priority)) {
        return false;
    }

    return true;

#else
    // Unsupported platform
    (void)thread;
    (void)cpu_id;
    (void)priority;
    (void)use_sched_fifo;
    return false;
#endif
}
inline double rad2deg(double rad) {
    return rad * 180.0 / M_PI;
}
inline double deg2rad(double deg) {
    return deg * M_PI / 180.0;
}

inline std::tuple<double, double, double> xyz2ypd_rad(double x, double y, double z) {
    double distance = std::sqrt(x * x + y * y + z * z);
    double yaw = std::atan2(y, x);
    double pitch = std::atan2(z, std::sqrt(x * x + y * y));
    return std::make_tuple(yaw, pitch, distance);
}

inline std::tuple<double, double, double> ypd2xyz_rad(double yaw, double pitch, double distance) {
    double x = distance * std::cos(pitch) * std::cos(yaw);
    double y = distance * std::cos(pitch) * std::sin(yaw);
    double z = distance * std::sin(pitch);
    return std::make_tuple(x, y, z);
}

inline std::tuple<double, double, double> xyz2ypd_deg(double x, double y, double z) {
    double distance = std::sqrt(x * x + y * y + z * z);
    double yaw = std::atan2(y, x);
    double pitch = std::atan2(z, std::sqrt(x * x + y * y));
    return std::make_tuple(rad2deg(yaw), rad2deg(pitch), distance);
}

inline std::tuple<double, double, double>
ypd2xyz_deg(double yaw_deg, double pitch_deg, double distance) {
    double yaw = deg2rad(yaw_deg);
    double pitch = deg2rad(pitch_deg);
    double x = distance * std::cos(pitch) * std::cos(yaw);
    double y = distance * std::cos(pitch) * std::sin(yaw);
    double z = distance * std::sin(pitch);
    return std::make_tuple(x, y, z);
}
inline CommonFrame toCommonFrame(wust_vl_video::ImageFrame& frame) {
    CommonFrame common_frame;
    common_frame.timestamp = frame.timestamp;
    if (frame.src_img.empty()) {
        return common_frame;
    }
    common_frame.src_img = std::move(frame.src_img);
}
inline Eigen::Vector3d xyz2ypd(const Eigen::Vector3d& xyz) {
    auto x = xyz[0], y = xyz[1], z = xyz[2];
    auto yaw = std::atan2(y, x);
    auto pitch = std::atan2(z, std::sqrt(x * x + y * y));
    auto distance = std::sqrt(x * x + y * y + z * z);
    return { yaw, pitch, distance };
}

} // namespace utils
