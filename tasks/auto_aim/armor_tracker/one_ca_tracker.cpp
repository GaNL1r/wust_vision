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
#include "one_ca_tracker.hpp"
#include "3rdparty/angles.h"
#include "tasks/utils.hpp"
#include "wust_vl/common/utils/logger.hpp"

// std
#include <algorithm>
#include <cfloat>
#include <fmt/format.h>
#include <iostream>
#include <memory>
#include <ostream>
#include <string>

OneCaTracker::OneCaTracker(
    double max_match_distance,
    double max_match_yaw_diff,
    double max_match_z_diff,
    double jump_thresh
):
    tracker_state(LOST),
    tracked_id_(armor::ArmorNumber::UNKNOWN),
    measurement_(Eigen::VectorXd::Zero(4)),
    target_state_(Eigen::VectorXd::Zero(10)),
    max_match_distance_(max_match_distance),
    max_match_yaw_diff_(max_match_yaw_diff),
    max_match_z_diff_(max_match_z_diff),
    jump_thresh(jump_thresh),
    detect_count_(0),
    lost_count_(0),
    last_yaw_(0) {}

void OneCaTracker::init(const armor::Armor& armor_msg) noexcept {
    if (armor_msg.is_none_purple) {
        return;
    }
    double min_distance = DBL_MAX;
    tracked_armor_ = armor_msg;

    min_distance = armor_msg.distance_to_image_center;
    tracked_armor_ = armor_msg;

    type_ = armor_msg.type;

    WUST_DEBUG(tracker_logger_) << "INIT EKF";
    initEKF(tracked_armor_);
    tracked_id_ = tracked_armor_.number;
    tracker_state = DETECTING;
}

void OneCaTracker::update(const armor::Armor& armor_msg) noexcept {
    Eigen::VectorXd ekf_prediction = ekf_->predict();
    bool matched = false;
    target_state_ = ekf_prediction;
    std::vector<armor::Armor> another_armors;
    double dis = std::sqrt(
        armor_msg.pos.x() * armor_msg.pos.x() + armor_msg.pos.y() * armor_msg.pos.y()
        + armor_msg.pos.z() * armor_msg.pos.z()
    );
    if (dis > 0.1) {
        tracked_armor_ = armor_msg;
        tracked_armor_.timestamp = armor_msg.timestamp;
        matched = true;
        auto p = tracked_armor_.target_pos;
        double measured_yaw = orientationToYaw(tracked_armor_.target_ori);
        measurement_ = Eigen::Vector4d(p.x(), p.y(), p.z(), measured_yaw);
        target_state_ = ekf_->update(measurement_);
        distance_to_image_center_ = armor_msg.distance_to_image_center;
    } else {
        matched = false;
    }

    // 状态机管理
    if (tracker_state == DETECTING) {
        if (matched) {
            detect_count_++;
            if (detect_count_ > tracking_thres_) {
                detect_count_ = 0;
                tracker_state = TRACKING;
            }
        } else {
            detect_count_ = 0;
            tracker_state = LOST;
        }
    } else if (tracker_state == TRACKING) {
        if (!matched) {
            tracker_state = TEMP_LOST;
            lost_count_++;
        }
    } else if (tracker_state == TEMP_LOST) {
        if (!matched) {
            lost_count_++;
            if (lost_count_ > lost_thres_) {
                lost_count_ = 0;
                tracker_state = LOST;
            }
        } else {
            tracker_state = TRACKING;
            lost_count_ = 0;
        }
    }
}

void OneCaTracker::initEKF(const armor::Armor& a) noexcept {
    double xa = a.target_pos.x();
    double ya = a.target_pos.y();
    double za = a.target_pos.z();
    last_yaw_ = 0;
    double yaw = orientationToYaw(a.target_ori);

    target_state_ = Eigen::VectorXd::Zero(onecaarmor_motion_model::X_N);
    target_state_ << xa, 0, 0, ya, 0, 0, za, 0, yaw, 0;
    ekf_->setState(target_state_);
}

double OneCaTracker::orientationToYaw(const Eigen::Quaterniond& q) noexcept {
    double roll, pitch, yaw;
    Eigen::Vector3d euler = utils::quatToEuler(q, utils::EulerOrder::ZYX, false);
    yaw = euler[0];
    yaw = this->last_yaw_ + angles::shortest_angular_distance(this->last_yaw_, yaw);
    this->last_yaw_ = yaw;
    return yaw;
}

Eigen::Vector3d OneCaTracker::getArmorPositionFromState(const Eigen::VectorXd& x) noexcept {
    double xa = x(0), ya = x(3), za = x(6);
    return Eigen::Vector3d(xa, ya, za);
}
