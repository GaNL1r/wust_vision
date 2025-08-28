// Copyright Chen Jun 2023. Licensed under the MIT License.
//
// Additional modifications and features by Chengfu Zou, Labor. Licensed under Apache License 2.0.
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
#include "tracker.hpp"
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

Tracker::Tracker(
    double max_match_distance,
    double max_match_yaw_diff,
    double max_match_z_diff,
    double jump_thresh
):
    tracker_state(LOST),
    tracked_id_(armor::ArmorNumber::UNKNOWN),
    measurement_(Eigen::VectorXd::Zero(4)),
    target_state_(Eigen::VectorXd::Zero(9)),
    max_match_distance_(max_match_distance),
    max_match_yaw_diff_(max_match_yaw_diff),
    max_match_z_diff_(max_match_z_diff),
    jump_thresh_(jump_thresh),
    yaw_diff_(0),
    position_diff_(0),
    detect_count_(0),
    lost_count_(0),
    last_yaw_(0),
    last_ypd_y(0) {}

void Tracker::init(const armor::Armors& armors_msg) noexcept {
    if (armors_msg.armors.empty())
        return;

    double min_distance = DBL_MAX;
    tracked_armor_ = armors_msg.armors[0];
    for (const auto& armor: armors_msg.armors) {
        if (!armor.is_ok) {
            continue;
        }
        if (armor.is_none_purple) {
            continue;
        }
        if (armor.distance_to_image_center < min_distance) {
            min_distance = armor.distance_to_image_center;
            tracked_armor_ = armor;
            type_ = armor.type;
        }
    }
    WUST_DEBUG(tracker_logger_) << "INIT EKF";
    initEKF(tracked_armor_);
    tracked_id_ = tracked_armor_.number;
    tracker_state = DETECTING;

    if (tracked_id_ == armor::ArmorNumber::OUTPOST) {
        tracked_armors_num_ = armor::ArmorsNum::OUTPOST_3;
    } else {
        tracked_armors_num_ = armor::ArmorsNum::NORMAL_4;
    }
}

void Tracker::update(const armor::Armors& armors_msg) noexcept {
    Eigen::VectorXd ekf_prediction;

    ekf_prediction = predict_func_(ekf_ypd_, esekf_ypd_);

    bool matched = false;
    target_state_ = ekf_prediction;
    acc_state_ = acc_ekf_->predict();
    center_velocity_measurement_ =
        Eigen::Vector3d(target_state_(1), target_state_(3), target_state_(5));
    acc_ekf_->update(center_velocity_measurement_);

    if (!armors_msg.armors.empty()) {
        armor::Armor same_id_armor;
        int same_id_armors_count = 0;
        auto predicted_position = getArmorPositionFromState(ekf_prediction);

        double min_position_diff = DBL_MAX;
        double min_z_diff = DBL_MAX;
        double yaw_diff = DBL_MAX;

        for (auto& armor: armors_msg.armors) {
            if (!armor.is_ok) {
                continue;
            }

            if (isSameTarget(armor.number, tracked_id_)) {
                same_id_armor = armor;
                same_id_armors_count++;

                auto p = armor.target_pos;
                Eigen::Vector3d position_vec(p.x(), p.y(), p.z());
                double position_diff = (predicted_position - position_vec).norm();
                double z_diff = std::abs(armor.target_pos.z() - predicted_position.z());

                if (position_diff < min_position_diff) {
                    min_position_diff = position_diff;
                    min_z_diff = z_diff;

                    yaw_diff = std::abs(orientationToYaw(armor.target_ori) - ekf_prediction(6));
                    tracked_armor_ = armor;
                    tracked_armor_.timestamp = armors_msg.timestamp;
                    yaw_diff_ = yaw_diff;

                    if (tracked_id_ == armor::ArmorNumber::OUTPOST) {
                        tracked_armors_num_ = armor::ArmorsNum::OUTPOST_3;
                    } else {
                        tracked_armors_num_ = armor::ArmorsNum::NORMAL_4;
                    }
                } else {
                    position_diff_ = position_diff;
                }
            }
        }

        if (min_position_diff < max_match_distance_ && yaw_diff < max_match_yaw_diff_
            && min_z_diff < max_match_z_diff_)
        {
            matched = true;
            auto p = tracked_armor_.target_pos;
            double measured_yaw = orientationToYaw(tracked_armor_.target_ori);

            double ypd_y = std::atan2(p.y(), p.x());
            ypd_y = this->last_ypd_y + angles::shortest_angular_distance(this->last_ypd_y, ypd_y);
            this->last_ypd_y = ypd_y;
            double ypd_p = std::atan2(p.z(), std::sqrt(p.x() * p.x() + p.y() * p.y()));
            double ypd_d = std::sqrt(p.x() * p.x() + p.y() * p.y() + p.z() * p.z());
            measurement_ = Eigen::Vector4d(ypd_y, ypd_p, ypd_d, measured_yaw);
            update_func_(ekf_ypd_, esekf_ypd_, measurement_);

        } else if (same_id_armors_count == 1 && yaw_diff > max_match_yaw_diff_ && min_z_diff < max_match_z_diff_)
        {
            handleArmorJump(same_id_armor);
        } else {
            WUST_DEBUG(tracker_logger_) << "No matched armor found!";
        }
    }

    if (target_state_(8) < 0.15) {
        target_state_(8) = 0.15;

        setstate_func_(ekf_ypd_, esekf_ypd_, target_state_);

    } else if (target_state_(8) > 0.4) {
        target_state_(8) = 0.4;

        setstate_func_(ekf_ypd_, esekf_ypd_, target_state_);
    }

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

void Tracker::initEKF(const armor::Armor& a) noexcept {
    double xa = a.target_pos.x();
    double ya = a.target_pos.y();
    double za = a.target_pos.z();
    last_yaw_ = 0;
    double yaw = orientationToYaw(a.target_ori);

    target_state_ = Eigen::VectorXd::Zero(ypdarmor_motion_model::X_N);
    double r = 0.24;
    double xc = xa + r * cos(yaw);
    double yc = ya + r * sin(yaw);
    double zc = za;
    d_za_ = 0, d_zc_ = 0, another_r_ = r;
    target_state_ << xc, 0, yc, 0, zc, 0, yaw, 0, r, d_zc_;
    setstate_func_(ekf_ypd_, esekf_ypd_, target_state_);
    acc_state_ = Eigen::VectorXd::Zero(acc_model::X_N);
    acc_state_ << 0, 0, 0, 0, 0, 0;
    acc_ekf_->setState(acc_state_);
}

void Tracker::handleArmorJump(const armor::Armor& current_armor) noexcept {
    double last_yaw = target_state_(6);
    double yaw = orientationToYaw(current_armor.target_ori);
    double delta_yaw = normalizeAngle(yaw - last_yaw);

    if (std::abs(delta_yaw) > jump_thresh_) {
        target_state_(6) = yaw;

        if (tracked_armors_num_ == armor::ArmorsNum::NORMAL_4) {
            d_za_ = target_state_(4) + target_state_(9) - current_armor.target_pos.z();
            std::swap(target_state_(8), another_r_);
            d_zc_ = d_zc_ == 0 ? -d_za_ : 0;
            target_state_(9) = d_zc_;
        }
        WUST_DEBUG(tracker_logger_) << "Armor Jump!";
    }

    Eigen::Vector3d current_p(
        current_armor.target_pos.x(),
        current_armor.target_pos.y(),
        current_armor.target_pos.z()
    );
    Eigen::Vector3d infer_p = getArmorPositionFromState(target_state_);

    if ((current_p - infer_p).norm() > max_match_distance_) {
        d_zc_ = 0;
        double r = target_state_(8);
        target_state_(0) = current_armor.target_pos.x() + r * cos(yaw);
        target_state_(1) = 0;
        target_state_(2) = current_armor.target_pos.y() + r * sin(yaw);
        target_state_(3) = 0;
        target_state_(4) = current_armor.target_pos.z();
        target_state_(5) = 0;
        target_state_(9) = d_zc_;
    }

    setstate_func_(ekf_ypd_, esekf_ypd_, target_state_);
}

double Tracker::orientationToYaw(const Eigen::Quaterniond& q) noexcept {
    double roll, pitch, yaw;
    Eigen::Vector3d euler = utils::quatToEuler(q, utils::EulerOrder::ZYX, false);
    yaw = euler[0];
    yaw = this->last_yaw_ + angles::shortest_angular_distance(this->last_yaw_, yaw);
    this->last_yaw_ = yaw;
    return yaw;
}

Eigen::Vector3d Tracker::getArmorPositionFromState(const Eigen::VectorXd& x) noexcept {
    double xc = x(0), yc = x(2), za = x(4) + x(9);
    double yaw = x(6), r = x(8);
    double xa = xc - r * cos(yaw);
    double ya = yc - r * sin(yaw);
    return Eigen::Vector3d(xa, ya, za);
}
