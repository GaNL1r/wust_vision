#pragma once

#include "target.hpp"
class TrackerV3 {
public:
    TrackerV3(
        int tracking_thres,
        double lost_dt,
        double max_yaw_diff_deg,
        const TargetConfig& target_config
    );
    Target track(const armor::Armors& armors_msg);
    enum State {
        LOST,
        DETECTING,
        TRACKING,
        TEMP_LOST,
    } tracker_state = LOST;
    bool initTarget(const armor::Armors& armors);
    bool updateTarget(const armor::Armors& armors);
    int tracking_thres_;
    int lost_thres_;
    int detect_count_ = 0;
    int lost_count_ = 0;
    double lost_dt_;
    double max_yaw_diff_deg_;

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