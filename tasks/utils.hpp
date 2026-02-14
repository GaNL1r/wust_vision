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

#include <Eigen/Dense>
#include <concepts>
#include <opencv2/opencv.hpp>
#include <thread>
#include <yaml-cpp/yaml.h>
namespace wust_vision {
// util functions
namespace utils {
    // Convert euler angles to rotation matrix
    enum class EulerOrder { XYZ, XZY, YXZ, YZX, ZXY, ZYX };

    double limit_rad(double angle) noexcept;

    Eigen::Vector3d quatToEuler(
        const Eigen::Quaterniond& q,
        int axis0,
        int axis1,
        int axis2,
        bool extrinsic = true
    );
    Eigen::Quaterniond eulerToQuat(
        const Eigen::Vector3d& euler,
        int axis0,
        int axis1,
        int axis2,
        bool extrinsic = true
    );

    Eigen::Matrix3d eulerToMatrix(
        const Eigen::Vector3d& euler,
        int axis0,
        int axis1,
        int axis2,
        bool extrinsic = true
    );

    Eigen::Vector3d
    matrixToEuler(const Eigen::Matrix3d& R, int axis0, int axis1, int axis2, bool extrinsic = true);

    Eigen::Vector3d
    quatToEuler(const Eigen::Quaterniond& q, EulerOrder order, bool extrinsic = true);

    Eigen::Quaterniond
    eulerToQuat(const Eigen::Vector3d& euler, EulerOrder order, bool extrinsic = true);
    Eigen::Matrix3d
    eulerToMatrix(const Eigen::Vector3d& euler, EulerOrder order, bool extrinsic = true);
    Eigen::Vector3d
    matrixToEuler(const Eigen::Matrix3d& R, EulerOrder order, bool extrinsic = true);

    Eigen::MatrixXd cvToEigen(const cv::Mat& cv_mat) noexcept;
    Eigen::Vector3d transformPosition(
        const Eigen::Vector3d& pos_camera,
        const Eigen::Matrix4d& T_camera_to_odom
    ) noexcept;

    Eigen::Quaterniond transformOrientation(
        const Eigen::Quaterniond& q_camera,
        const Eigen::Matrix4d& T_camera_to_odom
    ) noexcept;
    void pnpToEigen(
        const cv::Mat& rvec,
        const cv::Mat& tvec,
        Eigen::Vector3d& t_out,
        Eigen::Quaterniond& q_out
    ) noexcept;
    void pnpToEigen(
        const cv::Vec3d& rvec,
        const cv::Vec3d& tvec,
        Eigen::Vector3d& t_out,
        Eigen::Quaterniond& q_out
    ) noexcept;

    cv::Point2f computeCenter(const std::vector<cv::Point2f>& points) noexcept;
    bool isStateValid(const Eigen::VectorXd& state) noexcept;
    Eigen::Matrix4d computeCameraToOdomTransform(
        const Eigen::Matrix3d& R_gimbal2odom,
        const Eigen::Matrix3d& R_camera_to_gimbal,
        const Eigen::Vector3d& t_camera_to_gimbal
    ) noexcept;

    void addVelFromAccDt(Eigen::Vector3d& vel, const Eigen::Vector3d& acc, double dt) noexcept;
    void addPosFromVelDt(Eigen::Vector3d& pos, const Eigen::Vector3d& vel, double dt) noexcept;
    template<typename T>
    bool tryGetValue(const YAML::Node& node, const char* key, T& out_val);
    void changeFileOwner(const std::string& filepath, const std::string& username);
    std::string getOriginalUsername();
    bool setThreadAffinityAndPriority(
        std::thread& thread,
        int cpu_id,
        int priority,
        bool use_sched_fifo
    );
    double rad2deg(double rad) noexcept;
    double deg2rad(double deg) noexcept;
    std::tuple<double, double, double> xyz2ypd_rad(double x, double y, double z) noexcept;

    std::tuple<double, double, double>
    ypd2xyz_rad(double yaw, double pitch, double distance) noexcept;

    std::tuple<double, double, double> xyz2ypd_deg(double x, double y, double z) noexcept;
    std::tuple<double, double, double>
    ypd2xyz_deg(double yaw_deg, double pitch_deg, double distance) noexcept;
    Eigen::Vector3d xyz2ypd(const Eigen::Vector3d& xyz) noexcept;
    template<typename Point>
    Point getCenter(const std::vector<Point>& points) noexcept;
    bool segmentIntersection(
        const cv::Point2f& a1,
        const cv::Point2f& a2,
        const cv::Point2f& b1,
        const cv::Point2f& b2,
        cv::Point2f& intersection
    );
    std::vector<cv::Point2f> intersectLineRotatedRect(
        const cv::RotatedRect& rect,
        const cv::Point2f& line_p1,
        const cv::Point2f& line_p2
    );
    std::string makeTimestampedFileName();
    std::string expandEnv(const std::string& s);
    double computeBrightness(const cv::Mat& frame);
    cv::Mat letterbox(
        const cv::Mat& img,
        Eigen::Matrix3f& transform_matrix,
        const int new_shape_w,
        const int new_shape_h
    ) noexcept;
    template<typename Func>
    void XSecOnce(Func&& func, double dt) noexcept {
        static auto last_call = std::chrono::steady_clock::now();

        const auto now = std::chrono::steady_clock::now();
        const double elapsed = std::chrono::duration<double>(now - last_call).count();

        if (elapsed >= dt) {
            last_call = now;
            func();
        }
    }
    template<typename T>
    concept Point2DLike = requires(T p) {
        {
            p.x
            } -> std::convertible_to<float>;
        {
            p.y
            } -> std::convertible_to<float>;
        T { 0.f, 0.f };
    };
    template<Point2DLike T>
    [[nodiscard]] inline T transformPoint2D(const Eigen::Matrix3f& H, const T& p) noexcept {
        const Eigen::Vector3f hp { p.x, p.y, 1.f };
        const Eigen::Vector3f tp = H * hp;
        return { tp.x(), tp.y() };
    }
    inline cv::Rect2f transformRect(const Eigen::Matrix3f& H, const cv::Rect2f& rect) {
        cv::Point2f p1(rect.x, rect.y);
        cv::Point2f p2(rect.x + rect.width, rect.y);
        cv::Point2f p3(rect.x, rect.y + rect.height);
        cv::Point2f p4(rect.x + rect.width, rect.y + rect.height);

        auto tp1 = utils::transformPoint2D(H, p1);
        auto tp2 = utils::transformPoint2D(H, p2);
        auto tp3 = utils::transformPoint2D(H, p3);
        auto tp4 = utils::transformPoint2D(H, p4);

        float min_x = std::min({ tp1.x, tp2.x, tp3.x, tp4.x });
        float min_y = std::min({ tp1.y, tp2.y, tp3.y, tp4.y });
        float max_x = std::max({ tp1.x, tp2.x, tp3.x, tp4.x });
        float max_y = std::max({ tp1.y, tp2.y, tp3.y, tp4.y });

        return cv::Rect2f(min_x, min_y, max_x - min_x, max_y - min_y);
    }
} // namespace utils
} // namespace wust_vision