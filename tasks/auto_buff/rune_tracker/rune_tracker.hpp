#pragma once
#include "rune_target.hpp"
class RuneTracker {
public:
    RuneTracker(const YAML::Node& config);
    rune::RuneTarget track(const rune::RuneFan& fan);
    bool initTarget(const rune::RuneFan& fan);
    bool updateTarget(const rune::RuneFan& fan);
    enum State {
        LOST,
        DETECTING,
        TRACKING,
        TEMP_LOST,
    } tracker_state = LOST;
    rune::RuneTarget target_;
    int tracking_thres_;
    int lost_thres_;
    int detect_count_ = 0;
    int lost_count_ = 0;
    double lost_dt_;
    double max_dis_diff_;
    int found_count_ = 0;
    double pre_v_roll_ = 0;
    std::chrono::steady_clock::time_point last_time_;
    rune::RuneTargetConfig target_config_;
};