#pragma once
#include "tasks/auto_guidance/guidance_tracker/guidance_target.hpp"

namespace auto_guidance {
class GuidanceTracker {
public:
    using GuidanceTrackerPtr = std::unique_ptr<GuidanceTracker>;
    GuidanceTracker(const YAML::Node& config);
    static inline GuidanceTrackerPtr create(const YAML::Node& config) {
        return std::make_unique<GuidanceTracker>(config);
    }
    GuidanceTarget track(const GreenLights& lights);
    bool initTarget(const GreenLights& lights);
    bool updateTarget(const GreenLights& lights);
    enum State {
        LOST,
        DETECTING,
        TRACKING,
        TEMP_LOST,
    } tracker_state = LOST;
    GuidanceTarget guidance_target_;
    int tracking_thres_;
    int lost_thres_;
    int detect_count_ = 0;
    int lost_count_ = 0;
    double lost_dt_;
    std::chrono::steady_clock::time_point last_time_;
    TargetConfig target_config_;
};
} // namespace auto_guidance