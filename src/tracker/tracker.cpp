#include "tracker/tracker.hpp"
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

Tracker::Tracker(
    double max_match_distance,
    double max_match_yaw_diff,
    double max_match_z_diff,
    double jump_thresh
):
    tracker_state(LOST),
    tracked_id(ArmorNumber::UNKNOWN),
    measurement(Eigen::VectorXd::Zero(4)),
    target_state(Eigen::VectorXd::Zero(9)),
    max_match_distance_(max_match_distance),
    max_match_yaw_diff_(max_match_yaw_diff),
    max_match_z_diff_(max_match_z_diff),
    jump_thresh(jump_thresh),
    detect_count_(0),
    lost_count_(0),
    last_yaw_(0) {}

void Tracker::init(const Armors& armors_msg) noexcept {
    if (armors_msg.armors.empty())
        return;

    double min_distance = DBL_MAX;
    tracked_armor = armors_msg.armors[0];
    for (const auto& armor: armors_msg.armors) {
        if (!armor.is_ok) {
            continue;
        }
        if (armor.distance_to_image_center < min_distance) {
            min_distance = armor.distance_to_image_center;
            tracked_armor = armor;
            type = armor.type;
        }
    }
    WUST_DEBUG(tracker_logger) << "INIT EKF";
    initEKF(tracked_armor);
    tracked_id = tracked_armor.number;
    tracker_state = DETECTING;

    if (tracked_id == ArmorNumber::OUTPOST) {
        tracked_armors_num = ArmorsNum::OUTPOST_3;
    } else {
        tracked_armors_num = ArmorsNum::NORMAL_4;
    }
}

void Tracker::update(const Armors& armors_msg) noexcept {
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
            if (!armor.is_ok) {
                continue;
            }

            if (isSameTarget(armor.number, tracked_id)) {
                same_id_armor = armor;
                same_id_armors_count++;
                // WUST_INFO(tracker_logger)<<"Same ID armor
                // found!"<<fmt::format("count: {}\n", same_id_armors_count);
                auto p = armor.target_pos;
                Eigen::Vector3d position_vec(p.x, p.y, p.z);
                double position_diff = (predicted_position - position_vec).norm();
                double z_diff = std::abs(armor.target_pos.z - predicted_position.z());
                double x_diff = std::abs(armor.target_pos.x - predicted_position.x());
                // WUST_INFO(tracker_logger)<<"Armor
                // found!"<<fmt::format("position_diff: {}\n", position_diff);

                if (position_diff < min_position_diff) {
                    min_position_diff = position_diff;
                    min_z_diff = z_diff;

                    yaw_diff = std::abs(orientationToYaw(armor.target_ori) - ekf_prediction(6));
                    tracked_armor = armor;
                    tracked_armor.timestamp = armors_msg.timestamp;
                    yaw_diff_ = yaw_diff;

                    if (tracked_id == ArmorNumber::OUTPOST) {
                        tracked_armors_num = ArmorsNum::OUTPOST_3;
                    } else {
                        tracked_armors_num = ArmorsNum::NORMAL_4;
                    }
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
            handleArmorJump(same_id_armor);
            if_have_last_track_ = false;

        } else {
            // WUST_DEBUG(tracker_logger)<<"No matched armor found!";
        }
    }

    // 限制状态变量范围
    if (target_state(8) < 0.12) {
        target_state(8) = 0.12;
        ekf->setState(target_state);
    } else if (target_state(8) > 0.4) {
        target_state(8) = 0.4;
        ekf->setState(target_state);
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

void Tracker::initEKF(const Armor& a) noexcept {
    double xa = a.target_pos.x;
    double ya = a.target_pos.y;
    double za = a.target_pos.z;
    last_yaw_ = 0;
    double yaw = orientationToYaw(a.target_ori);

    target_state = Eigen::VectorXd::Zero(armor_motion_model::X_N);
    double r = 0.24;
    double xc = xa + r * cos(yaw);
    double yc = ya + r * sin(yaw);
    double zc = za;
    d_za = 0, d_zc = 0, another_r = r;
    target_state << xc, 0, yc, 0, zc, 0, yaw, 0, r, d_zc;
    ekf->setState(target_state);
}

void Tracker::handleArmorJump(const Armor& current_armor) noexcept {
    double last_yaw = target_state(6);
    double yaw = orientationToYaw(current_armor.target_ori);
    double delta_yaw = normalizeAngle(yaw - last_yaw);

    if (std::abs(delta_yaw) > jump_thresh) {
        target_state(6) = yaw;

        if (tracked_armors_num == ArmorsNum::NORMAL_4) {
            d_za = target_state(4) + target_state(9) - current_armor.target_pos.z;
            std::swap(target_state(8), another_r);
            // std::cout<<d_za<<"c"<<d_zc<<"t4"<<target_state(4)<<"az"<<current_armor.target_pos.z<<std::endl;
            d_zc = d_zc == 0 ? -d_za : 0;

            target_state(9) = d_zc;
        }
        WUST_DEBUG(tracker_logger) << "Armor Jump!";
    }

    Eigen::Vector3d current_p(
        current_armor.target_pos.x,
        current_armor.target_pos.y,
        current_armor.target_pos.z
    );
    Eigen::Vector3d infer_p = getArmorPositionFromState(target_state);

    if ((current_p - infer_p).norm() > max_match_distance_) {
        d_zc = 0;
        double r = target_state(8);
        target_state(0) = current_armor.target_pos.x + r * cos(yaw);
        target_state(1) = 0;
        target_state(2) = current_armor.target_pos.y + r * sin(yaw);
        target_state(3) = 0;
        target_state(4) = current_armor.target_pos.z;
        target_state(5) = 0;
        target_state(9) = d_zc;
    }

    ekf->setState(target_state);
}

double Tracker::orientationToYaw(const tf::Quaternion& q) noexcept {
    double roll, pitch, yaw;
    tf::Matrix3x3(q).getRPY(roll, pitch, yaw);
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
