// Copyright Chen Jun 2023. Licensed under the MIT License.
//
// Additional modifications and features by Chengfu Zou, Labor. Licensed under
// Apache License 2.0.
//
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

#pragma once

// std
#include <memory>
#include <string>
#include <variant>

// third party
#include <Eigen/Eigen>
#include <vector>

// project
#include "KalmanHyLib/kalman_hybird_lib.hpp"
#include "tracker/motion_models/acc_model.hpp"
#include "tracker/motion_models/motion_modela.hpp"
#include "tracker/motion_models/motion_modelypd.hpp"
#include "type/type.hpp"

inline double normalizeAngle(double angle) {
    while (angle > M_PI)
        angle -= 2 * M_PI;
    while (angle < -M_PI)
        angle += 2 * M_PI;
    return angle;
}

class Tracker {
public:
    Tracker(
        double max_match_distance,
        double max_match_yaw,
        double max_match_z_diff,
        double jump_thresh
    );

    void init(const armor::Armors& armors_msg) noexcept;
    void update(const armor::Armors& armors_msg) noexcept;

    enum State {
        LOST,
        DETECTING,
        TRACKING,
        TEMP_LOST,
    } tracker_state;

    std::function<
        Eigen::
            VectorXd(const std::unique_ptr<ypdarmor_motion_model::RobotStateEKF>&, const std::unique_ptr<ypdarmor_motion_model::RobotStateESEKF>&)>
        predict_func_;
    std::function<
        void(const std::unique_ptr<ypdarmor_motion_model::RobotStateEKF>&, const std::unique_ptr<ypdarmor_motion_model::RobotStateESEKF>&, const Eigen::VectorXd&)>
        update_func_;
    std::function<
        void(const std::unique_ptr<ypdarmor_motion_model::RobotStateEKF>&, const std::unique_ptr<ypdarmor_motion_model::RobotStateESEKF>&, const Eigen::VectorXd&)>
        setstate_func_;
    std::unique_ptr<ypdarmor_motion_model::RobotStateEKF> ekf_ypd_;
    std::unique_ptr<ypdarmor_motion_model::RobotStateESEKF> esekf_ypd_;
    bool use_esekf_;
    Eigen::VectorXd center_velocity_measurement_;
    Eigen::VectorXd acc_state_;
    std::unique_ptr<acc_model::VelocityAccelEKF> acc_ekf_;
    int tracking_thres_;
    int lost_thres_;

    armor::Armor tracked_armor_;

    armor::ArmorNumber tracked_id_;
    armor::ArmorsNum tracked_armors_num_;
    std::string type_;
    int retype_;
    Eigen::VectorXd measurement_;
    Eigen::VectorXd target_state_;

    double d_za_, another_r_;
    double d_zc_;
    float yaw_diff_;
    float position_diff_;

    double jump_thresh_ = 0.4;

private:
    void initEKF(const armor::Armor& a) noexcept;
    void handleArmorJump(const armor::Armor& a) noexcept;

    double orientationToYaw(const tf::Quaternion& q) noexcept;
    static Eigen::Vector3d getArmorPositionFromState(const Eigen::VectorXd& x) noexcept;

    double max_match_distance_;
    double max_match_yaw_diff_;
    double max_match_z_diff_;

    int detect_count_;
    int lost_count_;

    double last_yaw_;
    double last_ypd_y;

    std::string tracker_logger_ = "tracker";
};
