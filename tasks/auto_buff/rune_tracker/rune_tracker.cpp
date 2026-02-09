#include "rune_tracker.hpp"
namespace wust_vision {
namespace auto_buff {
    struct RuneTracker::Impl {
    public:
        Impl(wust_vl::common::utils::Parameter::Ptr auto_buff_config_parameter) {
            tracker_state = LOST;
            target_ = auto_buff::RuneTarget();
            target_config_ = RuneTargetConfig::create();
            auto_buff_config_parameter->registerGroup(*target_config_);
            auto_buff_config_parameter->reloadFromOldPath();
        }
        auto_buff::RuneTarget track(const auto_buff::RuneFan& fan) {
            double dt = std::chrono::duration<double>(fan.timestamp - last_time_).count();
            last_time_ = fan.timestamp;
            lost_thres_ =
                std::abs(static_cast<int>(target_config_->lost_time_thres_param.get() / dt));
            bool found;
            if (tracker_state == LOST) {
                found = initTarget(fan);
            } else {
                found = updateTarget(fan);
            }
            updateFsm(found);
            return target_;
        }
        void updateFsm(bool found) {
            switch (tracker_state) {
                case DETECTING:
                    if (found) {
                        if (++detect_count_ > target_config_->tracking_thres_param.get()) {
                            detect_count_ = 0;
                            tracker_state = TRACKING;
                        }
                    } else {
                        detect_count_ = 0;
                        tracker_state = LOST;
                    }
                    break;

                case TRACKING:
                    if (!found) {
                        tracker_state = TEMP_LOST;
                        lost_count_ = 1;
                    }
                    break;

                case TEMP_LOST:
                    if (!found) {
                        if (++lost_count_ > lost_thres_) {
                            lost_count_ = 0;
                            tracker_state = LOST;
                        }
                    } else {
                        lost_count_ = 0;
                        tracker_state = TRACKING;
                    }
                    break;

                default:
                    break;
            }

            target_.is_tracking = (tracker_state == TRACKING || tracker_state == TEMP_LOST);

            if (found)
                ++found_count_;
            if (target_.is_tracking) {
                pre_v_roll_ = target_.v_roll();
            }
        }
        bool initTarget(const auto_buff::RuneFan& fan) {
            if (!fan.is_valid || fan.fans.empty()) {
                return false;
            }
            target_ = auto_buff::RuneTarget(fan, target_config_, pre_v_roll_);
            tracker_state = DETECTING;
            return true;
        }
        bool updateTarget(const auto_buff::RuneFan& fan) {
            if (!fan.is_valid || fan.fans.empty()) {
                return false;
            }
            auto fan_copy = fan;
            std::erase_if(fan_copy.fans, [this](const auto_buff::RuneFan::Simple& f) {
                bool pose_check = std::abs((f.target_pos - target_.centerPos()).norm())
                        < target_config_->max_dis_diff_param.get()
                    && f.target_pos.norm() > 1.0;

                return !pose_check;
            });
            target_.predict(fan_copy.timestamp);
            return target_.update(fan_copy);
        }
        enum State {
            LOST,
            DETECTING,
            TRACKING,
            TEMP_LOST,
        } tracker_state = LOST;
        auto_buff::RuneTarget target_;
        int detect_count_ = 0;
        int lost_count_ = 0;
        int found_count_ = 0;
        double pre_v_roll_ = 0;
        int lost_thres_ = 0;
        std::chrono::steady_clock::time_point last_time_;
        RuneTargetConfig::Ptr target_config_;
    };
    RuneTracker::RuneTracker(wust_vl::common::utils::Parameter::Ptr auto_buff_config_parameter) {
        _impl = std::make_unique<Impl>(auto_buff_config_parameter);
    }
    RuneTracker::~RuneTracker() {
        _impl.reset();
    }
    auto_buff::RuneTarget RuneTracker::track(const auto_buff::RuneFan& fan) {
        return _impl->track(fan);
    }

} // namespace auto_buff
} // namespace wust_vision