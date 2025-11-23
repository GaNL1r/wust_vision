#include "trackerv3.hpp"
TrackerV3::TrackerV3(
    int tracking_thres,
    double lost_dt,
    double max_yaw_diff_deg,
    double max_dis_diff,
    const TargetConfig& target_config
):
    tracking_thres_(tracking_thres),
    lost_dt_(lost_dt),
    max_yaw_diff_deg_(max_yaw_diff_deg),
    max_dis_diff_(max_dis_diff),
    target_config_(target_config) {
    tracker_state = LOST;
    target_ = Target();
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
    // if (tracker_state != LOST && target_.diverged()) {
    //     tracker_state = LOST;
    //     WUST_WARN("tracker") << "Target diverged!";
    // }
    auto state = target_.esekf_ypd_.getState();
    Eigen::VectorXd state_dynamic = state;
    target_.clampState(state_dynamic);
    target_.esekf_ypd_.setState(state_dynamic);
    if (tracker_state != LOST && target_.esekf_ypd_.isRecentlyInconsistent()) {
        tracker_state = LOST;
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
    target_.predict(armors.timestamp, armors.v, false);
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
    const std::vector<Eigen::Vector4d>& xyza_list = target_.getArmorPosAndYaw();

    std::vector<std::pair<Eigen::Vector4d, int>> xyza_i_list;
    for (int i = 0; i < target_.armor_num_; i++) {
        xyza_i_list.push_back({ xyza_list[i], i });
    }

    std::sort(
        xyza_i_list.begin(),
        xyza_i_list.end(),
        [](const std::pair<Eigen::Vector4d, int>& a, const std::pair<Eigen::Vector4d, int>& b) {
            Eigen::Vector3d ypd1 = utils::xyz2ypd(a.first.head(3));
            Eigen::Vector3d ypd2 = utils::xyz2ypd(b.first.head(3));
            return ypd1[2] < ypd2[2];
        }
    );

    // key: xyza_index, value: pair<armor*, error>
    std::unordered_map<int, std::pair<armor::Armor*, double>> xyza_best_map;

    for (auto& armor: valid_armors) {
        int best_id = -1;
        double min_angle_error = 1e10;

        if (target_.armor_num_ > 3) {
            for (int i = 0; i < 3; i++) {
                const auto& xyza = xyza_i_list[i].first;
                Eigen::Vector3d ypd = utils::xyz2ypd(xyza.head(3));

                double angle_error = std::abs(angles::normalize_angle(
                                         target_.orientationToYaw(armor.target_ori) - xyza[3]
                                     ))
                    + std::abs(angles::normalize_angle(utils::xyz2ypd(armor.target_pos)[0] - ypd[0])
                    )
                    + std::abs(angles::normalize_angle(utils::xyz2ypd(armor.target_pos)[1] - ypd[1])
                    );

                if (angle_error < min_angle_error) {
                    min_angle_error = angle_error;
                    best_id = xyza_i_list[i].second;
                }
            }
        } else {
            for (int i = 0; i < 3; i++) {
                const auto& xyza = xyza_i_list[i].first;
                Eigen::Vector3d ypd = utils::xyz2ypd(xyza.head(3));
                auto angle_error = std::abs(
                    angles::normalize_angle(target_.orientationToYaw(armor.target_ori) - xyza[3])
                );
                if (std::abs(angle_error) < std::abs(min_angle_error)) {
                    best_id = xyza_i_list[i].second;
                    min_angle_error = angle_error;
                }
            }
        }

        if (best_id >= 0) {
            auto it = xyza_best_map.find(best_id);
            if (it == xyza_best_map.end() || min_angle_error < it->second.second) {
                xyza_best_map[best_id] = { &armor, min_angle_error };
            }
        }
    }

    std::vector<armor::Armor> filtered_armors;
    for (auto& kv: xyza_best_map) {
        filtered_armors.push_back(*(kv.second.first));
    }
    for (auto& armor: filtered_armors) {
        if (!armor::isSameTarget(armor.number, target_.tracked_id_))
            continue;
        if (armor.is_none_purple) {
            is_none_purple_count_++;
        } else {
            is_none_purple_count_ = 0;
        }
        if (is_none_purple_count_ > 100) {
            continue;
        }
        if (target_.update(armor)) {
            found_count++;
        }
    }

    if (found_count == 0)
        return false;
    return true;
}