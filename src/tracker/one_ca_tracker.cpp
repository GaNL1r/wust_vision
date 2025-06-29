#include "tracker/one_ca_tracker.hpp"
#include "common/angles.h"
#include "common/gobal.hpp"
#include "common/logger.hpp"
#include "type/type.hpp"

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
    tracked_id(ArmorNumber::UNKNOWN),
    measurement(Eigen::VectorXd::Zero(4)),
    target_state(Eigen::VectorXd::Zero(10)),
    max_match_distance_(max_match_distance),
    max_match_yaw_diff_(max_match_yaw_diff),
    max_match_z_diff_(max_match_z_diff),
    jump_thresh(jump_thresh),
    detect_count_(0),
    lost_count_(0),
    last_yaw_(0) {}

void OneCaTracker::init(const Armors& armors_msg) noexcept {
    if (armors_msg.armors.empty())
        return;

    double min_distance = DBL_MAX;
    tracked_armor = armors_msg.armors[0];
    for (const auto& armor: armors_msg.armors) {
        if (armor.distance_to_image_center < min_distance) {
            min_distance = armor.distance_to_image_center;
            tracked_armor = armor;
            retype = retypetotracker(armor.number);
            type = armor.type;
        }
    }
    WUST_DEBUG(tracker_logger) << "INIT EKF";
    initEKF(tracked_armor);
    tracked_id = tracked_armor.number;
    tracker_state = DETECTING;
}
void OneCaTracker::init(const Armor& armor_msg) noexcept {
    double min_distance = DBL_MAX;
    tracked_armor = armor_msg;

    min_distance = armor_msg.distance_to_image_center;
    tracked_armor = armor_msg;
    retype = retypetotracker(armor_msg.number);
    type = armor_msg.type;

    WUST_DEBUG(tracker_logger) << "INIT EKF";
    initEKF(tracked_armor);
    tracked_id = tracked_armor.number;
    tracker_state = DETECTING;
}

void OneCaTracker::update(const Armors& armors_msg) noexcept {
    Eigen::VectorXd ekf_prediction = ekf->predict();
    bool matched = false;
    target_state = ekf_prediction;
    std::vector<Armor> another_armors;
    if (gobal::if_manual_reset) {
        tracker_state = LOST;
        return;
    }

    if (!armors_msg.armors.empty()) {
        Armor same_id_armor;
        int same_id_armors_count = 0;
        auto predicted_position = getArmorPositionFromState(ekf_prediction);
        double min_position_diff = DBL_MAX;
        double min_z_diff = DBL_MAX;

        double yaw_diff = DBL_MAX;

        for (auto& armor: armors_msg.armors) {
            if (retypetotracker(armor.number) == retype) {
                same_id_armor = armor;
                same_id_armors_count++;

                auto p = armor.target_pos;
                Eigen::Vector3d position_vec(p.x, p.y, p.z);

                double position_diff = (predicted_position - position_vec).norm();
                double z_diff = std::abs(armor.target_pos.z - predicted_position.z());

                if (position_diff < min_position_diff) {
                    min_position_diff = position_diff;
                    min_z_diff = z_diff;

                    yaw_diff = std::abs(orientationToYaw(armor.target_ori) - ekf_prediction(8));
                    tracked_armor = armor;
                    tracked_armor.timestamp = armors_msg.timestamp;
                    yaw_diff_ = yaw_diff;

                } else {
                    another_armors.push_back(armor);
                    position_diff_ = position_diff;
                }
            }
        }

        if (min_position_diff < max_match_distance_ && yaw_diff < max_match_yaw_diff_
            && min_z_diff < max_match_z_diff_)
        {
            matched = true;
            auto p = tracked_armor.target_pos;
            double measured_yaw = orientationToYaw(tracked_armor.target_ori);
            measurement = Eigen::Vector4d(p.x, p.y, p.z, measured_yaw);
            target_state = ekf->update(measurement);

        } else if (same_id_armors_count == 1 && yaw_diff > max_match_yaw_diff_ && min_z_diff < max_match_z_diff_)
        {
            // handleArmorJump(same_id_armor);

            if_have_last_track_ = false;
            matched = true;
            auto p = tracked_armor.target_pos;
            double measured_yaw = orientationToYaw(tracked_armor.target_ori);
            measurement = Eigen::Vector4d(p.x, p.y, p.z, measured_yaw);
            target_state = ekf->update(measurement);

        } else {
            // WUST_DEBUG(tracker_logger)<<"No matched armor found!";
        }
    }

    // 状态机管理
    if (tracker_state == DETECTING) {
        if (matched) {
            detect_count_++;
            if (detect_count_ > tracking_thres) {
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
            if (lost_count_ > lost_thres) {
                lost_count_ = 0;
                tracker_state = LOST;
            }
        } else {
            tracker_state = TRACKING;
            lost_count_ = 0;
        }
    }
}
void OneCaTracker::update(const Armor& armor_msg) noexcept {
    Eigen::VectorXd ekf_prediction = ekf->predict();
    bool matched = false;
    target_state = ekf_prediction;
    std::vector<Armor> another_armors;
    double dis = std::sqrt(
        armor_msg.pos.x * armor_msg.pos.x + armor_msg.pos.y * armor_msg.pos.y
        + armor_msg.pos.z * armor_msg.pos.z
    );
    if (dis > 0.1) {
        tracked_armor = armor_msg;
        tracked_armor.timestamp = armor_msg.timestamp;
        matched = true;
        auto p = tracked_armor.target_pos;
        double measured_yaw = orientationToYaw(tracked_armor.target_ori);
        measurement = Eigen::Vector4d(p.x, p.y, p.z, measured_yaw);
        target_state = ekf->update(measurement);
        distance_to_image_center = armor_msg.distance_to_image_center;
    } else {
        matched = false;
    }

    // 状态机管理
    if (tracker_state == DETECTING) {
        if (matched) {
            detect_count_++;
            if (detect_count_ > tracking_thres) {
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
            if (lost_count_ > lost_thres) {
                lost_count_ = 0;
                tracker_state = LOST;
            }
        } else {
            tracker_state = TRACKING;
            lost_count_ = 0;
        }
    }
}

void OneCaTracker::initEKF(const Armor& a) noexcept {
    double xa = a.target_pos.x;
    double ya = a.target_pos.y;
    double za = a.target_pos.z;
    last_yaw_ = 0;
    double yaw = orientationToYaw(a.target_ori);

    target_state = Eigen::VectorXd::Zero(onecaarmor_motion_model::X_N);
    target_state << xa, 0, 0, ya, 0, 0, za, 0, yaw, 0;
    ekf->setState(target_state);
}

void OneCaTracker::handleArmorJump(const Armor& current_armor) noexcept {
    double last_yaw = target_state(6);
    double yaw = orientationToYaw(current_armor.target_ori);
    double delta_yaw = onormalizeAnglem(yaw - last_yaw);

    if (std::abs(delta_yaw) > jump_thresh) {
        double v_yaw = target_state(7);
        initEKF(current_armor);
        target_state(8) = yaw;
        target_state(9) = v_yaw;

        WUST_DEBUG(tracker_logger) << "Armor Jump!";
    }

    Eigen::Vector3d current_p(
        current_armor.target_pos.x,
        current_armor.target_pos.y,
        current_armor.target_pos.z
    );
    Eigen::Vector3d infer_p = getArmorPositionFromState(target_state);

    if ((current_p - infer_p).norm() > max_match_distance_) {
        target_state(0) = current_armor.target_pos.x;
        target_state(1) = 0;
        target_state(2) = 0;
        target_state(3) = current_armor.target_pos.y;
        target_state(4) = 0;
        target_state(5) = 0;
        target_state(6) = current_armor.target_pos.z;
        target_state(7) = 0;
    }

    ekf->setState(target_state);
}

double OneCaTracker::orientationToYaw(const tf::Quaternion& q) noexcept {
    double roll, pitch, yaw;
    tf::Matrix3x3(q).getRPY(roll, pitch, yaw);
    yaw = this->last_yaw_ + angles::shortest_angular_distance(this->last_yaw_, yaw);
    this->last_yaw_ = yaw;
    return yaw;
}

Eigen::Vector3d OneCaTracker::getArmorPositionFromState(const Eigen::VectorXd& x) noexcept {
    double xa = x(0), ya = x(3), za = x(6);
    return Eigen::Vector3d(xa, ya, za);
}
