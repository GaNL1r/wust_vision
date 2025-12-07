#include "guidance_tracker.hpp"

namespace auto_guidance {
GuidanceTracker::GuidanceTracker(const YAML::Node& config) {
    target_config_.load(config["target"]);
    tracking_thres_ = config["tracking_thres"].as<int>(5);
    lost_dt_ = config["lost_time_thres"].as<double>();
}
GuidanceTarget GuidanceTracker::track(const GreenLights& lights) {
    double dt = std::chrono::duration<double>(lights.timestamp - last_time_).count();
    last_time_ = lights.timestamp;
    lost_thres_ = std::abs(static_cast<int>(lost_dt_ / dt));
    bool found;
    if (tracker_state == LOST) {
        found = initTarget(lights);
    } else {
        found = updateTarget(lights);
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
    if (tracker_state == LOST || tracker_state == DETECTING) {
        guidance_target_.is_tracking_ = false;
    } else {
        guidance_target_.is_tracking_ = true;
    }

    return guidance_target_;
}
bool GuidanceTracker::initTarget(const GreenLights& lights) {
    int best_id = -1;
    double max_score = -1e9;
    for (int i = 0; i < lights.lights.size(); i++) {
        if (lights.lights[i].score > max_score) {
            max_score = lights.lights[i].score;
            best_id = i;
        }
    }
    if (best_id == -1) {
        return false;
    }
    tracker_state = DETECTING;
    guidance_target_ = GuidanceTarget(lights.lights[best_id], target_config_);
    return true;
}
bool GuidanceTracker::updateTarget(const GreenLights& lights) {
    guidance_target_.predict(lights.timestamp);
    return guidance_target_.update(lights);
}
} // namespace auto_guidance