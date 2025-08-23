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

#include "tasks/auto_aim/tf.hpp"

#include "opencv2/opencv.hpp"
#include "tasks/auto_aim/type.hpp"
struct Target_info {
    std::vector<std::vector<cv::Point2f>> pts;
    std::vector<tf::Position> pos;
    std::vector<tf::Quaternion> ori;
    int select_id;
    std::vector<bool> is_ok;
};

namespace mono_measure_tool {
bool solvePnp(
    const std::vector<cv::Point2f>& points2d,
    const std::vector<cv::Point3f>& points3d,
    cv::Point3f& position,
    cv::Mat& rvec,
    const cv::Mat& camera_intrinsic_,
    const cv::Mat& camera_distortion_,
    cv::SolvePnPMethod pnp_method = cv::SOLVEPNP_ITERATIVE
);
cv::Point3f unproject(cv::Point2f p, double distance);

void calcViewAngle(cv::Point2f p, float& pitch, float& yaw);
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

static std::vector<cv::Point3f> SMALL_ARMOR_3D_POINTS = {
    { 0, 0.025, -0.066 },
    { 0, -0.025, -0.066 },
    { 0, -0.025, 0.066 },
    { 0, 0.025, 0.066 },
};

static std::vector<cv::Point3f> BIG_ARMOR_3D_POINTS = {
    { 0, 0.025, -0.1125 },
    { 0, -0.025, -0.1125 },
    { 0, -0.025, 0.1125 },
    { 0, 0.025, 0.1125 },
};
static std::vector<cv::Point3f> SMALL_ARMOR_3D_POINTS_NET = {
    { 0, 0.027, -0.066 },
    { 0, -0.027, -0.066 },
    { 0, -0.027, 0.066 },
    { 0, 0.027, 0.066 },
};

static std::vector<cv::Point3f> BIG_ARMOR_3D_POINTS_NET = {
    { 0, 0.027, -0.1125 },
    { 0, -0.027, -0.1125 },
    { 0, -0.027, 0.1125 },
    { 0, 0.027, 0.1125 },
};

static std::vector<cv::Point3f> R_BOX_POINTS = {
    { 0, 0.05, -0.05 },
    { 0, -0.05, -0.05 },
    { 0, -0.05, 0.05 },
    { 0, 0.05, 0.05 },
};

} // namespace mono_measure_tool
