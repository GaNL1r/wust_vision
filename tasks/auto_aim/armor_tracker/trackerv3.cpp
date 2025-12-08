#include "trackerv3.hpp"
TrackerV3::TrackerV3(const YAML::Node& config) {
    tracker_state = LOST;
    target_ = Target();
    tracking_thres_ = config["armor_tracker"]["tracking_thres"].as<int>(5);
    lost_dt_ = config["armor_tracker"]["lost_time_thres"].as<double>();
    max_yaw_diff_deg_ = config["armor_tracker"]["max_yaw_diff_deg"].as<double>(80.0);
    max_dis_diff_ = config["armor_tracker"]["max_dis_diff"].as<double>(0.5);
    target_config_.loadConfig(config["armor_tracker"]);
}
Target TrackerV3::track(const armor::Armors& armors_msg) {
    double dt = std::chrono::duration<double>(armors_msg.timestamp - last_time_).count();
    last_time_ = armors_msg.timestamp;
    lost_thres_ = std::abs(static_cast<int>(lost_dt_ / dt));
    armor::Armors armors;
    armors = armors_msg;
    std::erase_if(armors.armors, [this](const armor::Armor& a) {
        double center_yaw = std::atan2(target_.position().y(), target_.position().x());
        bool state_check = tracker_state == TRACKING;
        bool outpost_check = target_.tracked_id_ == armor::ArmorNumber::OUTPOST && !a.is_ok;
        bool pose_check =
            (std::abs(
                 angles::normalize_angle(orientationToYaw(a.target_ori, center_yaw) - center_yaw)
             ) > (max_yaw_diff_deg_ * M_PI / 180.0)
             || std::abs((a.target_pos - target_.position()).norm()) > max_dis_diff_)
            && target_.is_inited
            && std::abs(time_utils::durationMs(target_.timestamp_, time_utils::now())) < 1000.0;

        return state_check && outpost_check || pose_check;
    });

    std::sort(
        armors.armors.begin(),
        armors.armors.end(),
        [](const armor::Armor& a, const armor::Armor& b) {
            return a.distance_to_image_center < b.distance_to_image_center;
        }
    );
    bool found;
    if (tracker_state == LOST) {
        found = initTarget(armors);
    } else {
        found = updateTarget(armors);
    }
    if (tracker_state == DETECTING) {
        if (found) {
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
        if (!found) {
            tracker_state = TEMP_LOST;
            lost_count_++;
        }
    } else if (tracker_state == TEMP_LOST) {
        if (!found) {
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
    if ((target_.esekf_ypd_.isRecentlyInconsistent() || target_.diverged())
        && tracker_state != LOST) {
        initTarget(armors);
        tracker_state = TRACKING;
        WUST_WARN("tracker") << "Bad Converge Found!";
    }
    if (tracker_state == LOST || tracker_state == DETECTING) {
        target_.is_tracking = false;
    } else {
        target_.is_tracking = true;
    }
    if (tracker_state == TEMP_LOST) {
        target_.is_temp_lost_ = true;
    } else {
        target_.is_temp_lost_ = false;
    }
    if (found) {
        found_count_++;
    }

    return target_;
}
bool TrackerV3::initTarget(const armor::Armors& armors) {
    if (armors.armors.empty()) {
        return false;
    }
    auto a = armors.armors.front();
    if (a.is_none_purple) {
        return false;
    }
    Eigen::DiagonalMatrix<double, ypdv2armor_motion_model::X_N> p0;
    if (a.number == armor::ArmorNumber::OUTPOST) {
        p0.diagonal() << 1, 64, 1, 64, 1, 81, 0.4, 100, 1e-4, 0.1, 0.1;
        target_ = Target(a, target_config_, 0.2765, 3, p0);
    } else if (a.number == armor::ArmorNumber::BASE) {
        p0.diagonal() << 1, 64, 1, 64, 1, 64, 0.4, 100, 1e-4, 0, 0;
        target_ = Target(a, target_config_, 0.3205, 3, p0);
    } else {
        p0.diagonal() << 1, 64, 1, 64, 1, 64, 0.4, 100, 1, 1, 1;
        target_ = Target(a, target_config_, 0.2, 4, p0);
    }
    tracker_state = DETECTING;
    return true;
}
bool TrackerV3::updateTarget(const armor::Armors& armors) {
    if (armors.armors.empty()) {
        return false;
    }
    int found_count = 0;
    target_.predict(armors.timestamp, armors.v);

    std::vector<armor::Armor> valid_armors;
    for (const auto& armor: armors.armors) {
        if (!armor::isSameTarget(armor.number, target_.tracked_id_))
            continue;
        found_count++;
        valid_armors.push_back(armor);
    }
    if (found_count == 0)
        return false;
    found_count = 0;
    auto matched_armors = target_.match(valid_armors);
    for (auto [id, armor]: matched_armors) {
        if (armor.is_none_purple) {
            is_none_purple_count_++;
        } else {
            is_none_purple_count_ = 0;
        }
        if (is_none_purple_count_ > 100 && armor.is_none_purple) {
            continue;
        }
        if (target_.update(std::make_pair(id, armor))) {
            found_count++;
        }
    }

    if (found_count == 0)
        return false;
    return true;
}