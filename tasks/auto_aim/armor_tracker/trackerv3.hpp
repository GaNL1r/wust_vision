#pragma once

#include "target.hpp"
class TrackerV3 {
public:
    TrackerV3(const YAML::Node& config) noexcept;
    Target track(const armor::Armors& armors_msg) noexcept;
    void updateFsm(bool found) noexcept;
    enum State {
        LOST,
        DETECTING,
        TRACKING,
        TEMP_LOST,
    } tracker_state = LOST;
    bool initTarget(const armor::Armors& armors) noexcept;
    bool updateTarget(const armor::Armors& armors) noexcept;
    int tracking_thres_;
    int lost_thres_;
    int detect_count_ = 0;
    int lost_count_ = 0;
    double lost_dt_;
    double max_yaw_diff_deg_;
    double max_dis_diff_;
    int is_none_purple_count_ = 0;
    int found_count_ = 0;
    Target target_;
    std::chrono::steady_clock::time_point last_time_;
    TargetConfig target_config_;

    double orientationToYaw(const Eigen::Quaterniond& q, double from) noexcept {
        double roll, pitch, yaw;
        Eigen::Vector3d euler = utils::quatToEuler(q, utils::EulerOrder::ZYX, false);
        yaw = euler[0];
        yaw = from + angles::shortest_angular_distance(from, yaw);
        return yaw;
    }
};