// Copyright 2023 Yunlong Feng
// Copyright 2025 Lihan Chen
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

#include <string>
#include <vector>

#include "common/tf.hpp"

#include "opencv2/opencv.hpp"
#include "type/type.hpp"
struct Target_info {
    std::vector<std::vector<cv::Point2f>> pts;
    std::vector<tf::Position> pos;
    std::vector<tf::Quaternion> ori;
    int select_id;
    std::vector<bool> is_ok;
};

class MonoMeasureTool {
public:
    MonoMeasureTool() = default;

    /**
   * @brief Solve Perspective-n-Point problem in camera
   * 3d点坐标求解（use solve pnp）
   * @param points2d a list of points in image frame
   * @param points3d a list of points correspondent to points2d
   * @param position output position of the origin point of 3d coordinate system
   * @return true
   * @return false
   */
    bool solvePnp(
        const std::vector<cv::Point2f>& points2d,
        const std::vector<cv::Point3f>& points3d,
        cv::Point3f& position,
        cv::Mat& rvec,
        const cv::Mat& camera_intrinsic_,
        const cv::Mat& camera_distortion_,
        cv::SolvePnPMethod pnp_method = cv::SOLVEPNP_ITERATIVE
    );
    /**
   * @brief 逆投影，已知深度，2d->3d点求解
   *
   * @param p 图像上点坐标
   * @param distance 已知的真实距离
   * @return cv::Point3f 对应的真实3d点坐标
   */
    cv::Point3f unproject(cv::Point2f p, double distance);
    /**
   * @brief 视角求解
   *
   * @param p 图像上点坐标
   * @param pitch 视角pitch
   * @param yaw 视角yaw
   */
    void calcViewAngle(cv::Point2f p, float& pitch, float& yaw);

    /**
   * @brief 装甲板目标位姿求解
   *
   * @param obj 装甲板目标
   * @param position 返回的坐标
   * @param rvec 相对旋转向量
   * @return true
   * @return false
   */
    bool calcArmorTarget(
        const armor::ArmorObject& obj,
        cv::Point3f& position,
        cv::Mat& rvec,
        std::string& armor_type,
        const cv::Mat& camera_intrinsic_,
        const cv::Mat& camera_distortion_
    );

    bool projectRTargetToImage(
        const Eigen::Matrix4d& TRodom,
        const Eigen::Matrix4d& T_camera_to_odom,
        std::vector<cv::Point2f>& manual_r_box,
        const cv::Mat& camera_intrinsic_,
        const cv::Mat& camera_distortion_
    );
    bool calcRTarget(
        const std::vector<cv::Point2f>& manual_r_box,
        Eigen::Matrix4d& TRodom,
        const Eigen::Matrix4d& T_camera_to_odom,
        const cv::Mat& camera_intrinsic_,
        const cv::Mat& camera_distortion_
    );

    float calcDistanceToCenter(
        const armor::ArmorObject& obj,
        const cv::Mat& camera_intrinsic_,
        const cv::Mat& camera_distortion_
    );

    bool reprojectArmorsCorners(
        armor::Armors& armors,
        Target_info& target_info,
        const cv::Mat& camera_intrinsic_,
        const cv::Mat& camera_distortion_
    );

    bool reprojectArmorCorners(
        const armor::Armor& armor,
        std::vector<cv::Point2f>& image_points,
        const cv::Mat& camera_intrinsic_,
        const cv::Mat& camera_distortion_
    );
    bool reprojectArmorCorners_raw(
        const armor::Armor& armor,
        std::vector<cv::Point2f>& image_points,
        const cv::Mat& camera_intrinsic_,
        const cv::Mat& camera_distortion_
    );
    void processDetectedArmors(
        const std::vector<armor::ArmorObject>& objs,
        armor::Armors& armors_out,
        Eigen::Matrix4d T_camera_to_odom,
        const cv::Mat& camera_intrinsic_,
        const cv::Mat& camera_distortion_
    );

    static std::vector<cv::Point3f> SMALL_ARMOR_3D_POINTS;
    static std::vector<cv::Point3f> BIG_ARMOR_3D_POINTS;
    static std::vector<cv::Point3f> SMALL_ARMOR_3D_POINTS_NET;
    static std::vector<cv::Point3f> BIG_ARMOR_3D_POINTS_NET;
    static std::vector<cv::Point3f> R_BOX_POINTS;

private:
    // 相机参数
    cv::Mat prev_rvec_, prev_tvec_;
    bool has_prev_ { false };

    std::string mono_logger = "mono_measure_tool";

    double fx_ { 0 }, fy_ { 0 }, u0_ { 0 }, v0_ { 0 };
};
