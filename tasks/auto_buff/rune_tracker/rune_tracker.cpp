#include "rune_tracker.hpp"

RuneTracker::RuneTracker(const YAML::Node& config) {
    tracker_state = LOST;
    target_ = rune::RuneTarget();
    tracking_thres_ = config["tracking_thres"].as<int>();
    lost_thres_ = config["lost_time_thres"].as<double>();
    max_dis_diff_ = config["max_dis_diff"].as<double>();
    target_config_.loadFromYaml(config);
}
rune::RuneTarget RuneTracker::track(const rune::RuneFan& fan) {
    double dt = std::chrono::duration<double>(fan.timestamp - last_time_).count();
    last_time_ = fan.timestamp;
    lost_thres_ = std::abs(static_cast<int>(lost_dt_ / dt));
    // bool pose_check = std::abs((fan.target_pos - target_.centerPos()).norm()) < max_dis_diff_
    //     && fan.target_pos.norm() > 1.0;
    bool found;
    bool ok;
    if (tracker_state == LOST) {
        found = initTarget(fan);
        ok = found;
    } else {
        found = updateTarget(fan);
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
    if (tracker_state != LOST && target_.diverged()) {
        tracker_state = LOST;
        WUST_WARN("tracker") << "Target diverged!";
    }
    if (tracker_state != LOST && target_.esekf_ypd_.isRecentlyInconsistent()) {
        //tracker_state = LOST;
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
    if (target_.is_tracking) {
        pre_v_roll_ = target_.v_roll();
    }

    return target_;
}
bool RuneTracker::initTarget(const rune::RuneFan& fan) {
    if (!fan.is_valid || fan.fans.empty()) {
        return false;
    }
    target_ = rune::RuneTarget(fan.is_big, fan, target_config_, pre_v_roll_);
    tracker_state = DETECTING;
    return true;
}
bool RuneTracker::updateTarget(const rune::RuneFan& fan) {
    if (!fan.is_valid || fan.fans.empty()) {
        return false;
    }
    auto fan_copy = fan;
    std::erase_if(fan_copy.fans, [this](const rune::RuneFan::Simple& f) {
        bool pose_check = std::abs((f.target_pos - target_.centerPos()).norm()) < max_dis_diff_
            && f.target_pos.norm() > 1.0;

        return !pose_check;
    });
    target_.predict(fan_copy.timestamp);
    return target_.update(fan_copy);
}