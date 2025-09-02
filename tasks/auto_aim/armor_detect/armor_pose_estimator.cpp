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

#include "armor_pose_estimator.hpp"

#include "tasks/utils.hpp"
#include "wust_vl/common/utils/logger.hpp"
#include "yaml-cpp/yaml.h"
#include <iostream>

ArmorPoseEstimator::ArmorPoseEstimator(YAML::Node config, std::pair<cv::Mat, cv::Mat> camera_info) {
    pnp_solver_ = std::make_unique<PnPSolver>();
    pnp_solver_->setObjectPoints(
        "small",
        armor::ArmorObject::buildObjectPoints<cv::Point3f>(SMALL_ARMOR_WIDTH, SMALL_ARMOR_HEIGHT)
    );
    pnp_solver_->setObjectPoints(
        "large",
        armor::ArmorObject::buildObjectPoints<cv::Point3f>(LARGE_ARMOR_WIDTH, LARGE_ARMOR_HEIGHT)
    );

    std::array<double, 9> camera_matrix;
    //auto camera_info = gobal::stringanything.get_value<std::pair<cv::Mat, cv::Mat>>("camera_info");
    CV_Assert(camera_info.first.rows == 3 && camera_info.first.cols == 3);
    CV_Assert(camera_info.first.type() == CV_64F);

    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
            camera_matrix[i * 3 + j] = camera_info.first.at<double>(i, j);

    ba_solver_ = std::make_unique<BaSolver>(
        camera_matrix,
        config["armor_optimize"]["max_iter_R"].as<int>(),
        config["armor_optimize"]["max_iter_t"].as<int>(),
        config["armor_optimize"]["step_R"].as<int>(),
        config["armor_optimize"]["step_t"].as<int>(),
        config["armor_optimize"]["min_error_R"].as<double>(),
        config["armor_optimize"]["min_error_t"].as<double>()
    );
    distance_fix_a2_ = config["armor_optimize"]["distance_fix_a2"].as<double>();

    R_gimbal_camera_ = Eigen::Matrix3d::Identity();
    R_gimbal_camera_ << 0, 0, 1, -1, 0, 0, 0, -1, 0;
}

std::vector<armor::Armor> ArmorPoseEstimator::extractArmorPoses(
    const std::vector<armor::ArmorObject>& armors,
    Eigen::Matrix4d T_camera_to_odom,
    const cv::Mat& camera_intrinsic,
    const cv::Mat& camera_distortion
) {
    std::vector<armor::Armor> armors_msg;

    const Eigen::Matrix3d R_imu_cam = T_camera_to_odom.block<3, 3>(0, 0);
    auto makeArmorMsg = [&](const armor::ArmorObject& obj,
                            const Eigen::Vector3d& t,
                            const Eigen::Matrix3d& R) {
        armor::Armor msg;
        msg.type = (obj.number == armor::ArmorNumber::NO1 || obj.number == armor::ArmorNumber::BASE)
            ? "large"
            : "small";
        msg.number = obj.number;
        Eigen::Quaterniond q(R);
        Eigen::Quaterniond add_roll { Eigen::AngleAxisd(M_PI / 2, Eigen::Vector3d::UnitX()) };
        Eigen::Quaterniond new_q = q * add_roll;

        auto [yaw, pitch, dist] = utils::xyz2ypd_rad(t.x(), t.y(), t.z());
        dist += distance_fix_a2_ * dist * dist;
        auto [x, y, z] = utils::ypd2xyz_rad(yaw, pitch, dist);

        msg.pos = { x, y, z };
        msg.ori = new_q;

        utils::transformArmorData(msg, T_camera_to_odom);
        msg.distance_to_image_center =
            pnp_solver_->calculateDistanceToCenter(obj.center, camera_intrinsic);
        msg.is_ok = true;
        return msg;
    };

    for (auto const& a: armors) {
        if (!a.is_ok)
            continue;

        std::vector<cv::Mat> rvecs, tvecs;
        std::string type =
            (a.number == armor::ArmorNumber::NO1 || a.number == armor::ArmorNumber::BASE) ? "large"
                                                                                          : "small";

        if (!pnp_solver_->solvePnPGeneric(
                a.landmarks(),
                rvecs,
                tvecs,
                type,
                camera_intrinsic,
                camera_distortion
            ))
        {
            WUST_WARN("PNP") << "PNP failed";
            continue;
        }

        sortPnPResult(a, rvecs, tvecs, type, camera_intrinsic, camera_distortion);
        cv::Mat R_cv;
        cv::Rodrigues(rvecs[0], R_cv);
        Eigen::Matrix3d R = utils::cvToEigen(R_cv);
        Eigen::Vector3d t = utils::cvToEigen(tvecs[0]);

        double roll_deg =
            utils::matrixToEuler(R_gimbal_camera_ * R, utils::EulerOrder::ZXY)[0] * 180 / M_PI;
        if (use_ba_ && ba_solver_) {
            Eigen::Matrix3d R0 = R;
            R = ba_solver_
                    ->solveBa_R(a, t, R, R_imu_cam, type, camera_intrinsic, camera_distortion);
            //t = ba_solver_->solveBa_t(a, t, R0, R_imu_cam, type);
        }

        armors_msg.push_back(makeArmorMsg(a, t, R));
    }

    return armors_msg;
}

void ArmorPoseEstimator::sortPnPResult(
    const armor::ArmorObject& armor,
    std::vector<cv::Mat>& rvecs,
    std::vector<cv::Mat>& tvecs,
    std::string coord_frame_name,
    const cv::Mat& camera_intrinsic,
    const cv::Mat& camera_distortion
) const {
    constexpr double ERR_RATIO_TH = 3.0;
    constexpr double ROLL_TH_RAD = 10.0 * M_PI / 180.0;

    if (rvecs.size() < 2 || tvecs.size() < 2)
        return;

    struct Candidate {
        cv::Mat& rvec;
        cv::Mat& tvec;
        Eigen::Matrix3d R;
        Eigen::Vector3d rpy;
        double reprojErr = std::numeric_limits<double>::infinity();
    } c[2] = { { rvecs[0], tvecs[0] }, { rvecs[1], tvecs[1] } };

    // 计算旋转、RPY 和重投影误差
    for (int i = 0; i < 2; ++i) {
        cv::Mat R_cv;
        cv::Rodrigues(c[i].rvec, R_cv);
        c[i].R = utils::cvToEigen(R_cv);
        c[i].rpy = utils::matrixToEuler(R_gimbal_camera_ * c[i].R, utils::EulerOrder::ZXY);

        c[i].reprojErr = pnp_solver_->calculateReprojectionError(
            armor.landmarks(),
            c[i].rvec,
            c[i].tvec,
            coord_frame_name,
            camera_intrinsic,
            camera_distortion
        );
    }

    // 如果误差比过大或任一解的 roll 超限，则不调整
    if (c[1].reprojErr / c[0].reprojErr > ERR_RATIO_TH || std::abs(c[0].rpy(0)) > ROLL_TH_RAD
        || std::abs(c[1].rpy(0)) > ROLL_TH_RAD)
    {
        return;
    }

    double l_ang = std::atan2(armor.lights[0].axis.y, armor.lights[0].axis.x) * 180.0 / M_PI;
    double r_ang = std::atan2(armor.lights[1].axis.y, armor.lights[1].axis.x) * 180.0 / M_PI;
    double boardTilt = (l_ang + r_ang) * 0.5 + 90.0;
    if (armor.number == armor::ArmorNumber::OUTPOST)
        boardTilt = -boardTilt;

    // 根据倾斜方向选解：左倾时选 yaw<0 解，右倾时选 yaw>0 解
    bool leftTilt = boardTilt > 0;
    double yaw0 = c[0].rpy(2), yaw1 = c[1].rpy(2);
    if ((leftTilt && yaw0 > 0 && yaw1 < 0) || (!leftTilt && yaw0 < 0 && yaw1 > 0)) {
        std::swap(rvecs[0], rvecs[1]);
        std::swap(tvecs[0], tvecs[1]);
    }
}
