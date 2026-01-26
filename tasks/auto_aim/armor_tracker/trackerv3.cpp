#include "trackerv3.hpp"

namespace wust_vision {
namespace auto_aim {
    struct Tracker::Impl {
    public:
        enum State {
            LOST,
            DETECTING,
            TRACKING,
            TEMP_LOST,
        } tracker_state = LOST;
        Impl(wust_vl::common::utils::Parameter::Ptr auto_aim_config_parameter) {
            tracker_state = LOST;
            target_config_ = TargetConfig::create();
            auto_aim_config_parameter->registerGroup(*target_config_);
            auto_aim_config_parameter->reloadFromOldPath();
            target_ = Target();
        }

        Target track(const Armors& armors_msg) noexcept {
            const double dt =
                std::chrono::duration<double>(armors_msg.timestamp - last_time_).count();
            last_time_ = armors_msg.timestamp;
            lost_thres_ =
                std::abs(static_cast<int>(target_config_->lost_time_thres_param.get() / dt));
            Armors armors;
            armors = armors_msg;
            std::erase_if(armors.armors, [this](const Armor& a) {
                double center_yaw = std::atan2(target_.position().y(), target_.position().x());
                bool state_check = tracker_state == TRACKING;
                bool outpost_check = target_.tracked_id_ == ArmorNumber::OUTPOST && !a.is_ok;
                bool pose_check =
                    (std::abs(angles::normalize_angle(
                         orientationToYaw(a.target_ori, center_yaw) - center_yaw
                     )) > (target_config_->max_yaw_diff_deg_param.get() * M_PI / 180.0)
                     || std::abs((a.target_pos - target_.position()).norm())
                         > target_config_->max_dis_diff_param.get())
                    && target_.is_inited
                    && std::abs(wust_vl::common::utils::time_utils::durationMs(
                           target_.timestamp_,
                           wust_vl::common::utils::time_utils::now()
                       ))
                        < 1000.0;

                return state_check && outpost_check || pose_check;
            });

            std::sort(
                armors.armors.begin(),
                armors.armors.end(),
                [](const Armor& a, const Armor& b) {
                    return a.distance_to_image_center < b.distance_to_image_center;
                }
            );
            bool found;
            if (tracker_state == LOST) {
                found = initTarget(armors);
            } else {
                found = updateTarget(armors);
            }
            updateFsm(found);
            if ((target_.diverged()) && tracker_state != LOST) {
                tracker_state = LOST;
                WUST_WARN("tracker") << "Target diverged!";
            }

            return target_;
        }
        void updateFsm(bool found) noexcept {
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
            target_.is_temp_lost_ = (tracker_state == TEMP_LOST);

            if (found)
                ++found_count_;
        }

        bool initTarget(const Armors& armors) noexcept {
            if (armors.armors.empty()) {
                return false;
            }
            bool found = false;
            Armor init_target;
            Armors others = armors;
            others.armors.clear();
            for (auto& a: armors.armors) {
                if (!a.is_none_purple && !found) {
                    init_target = a;
                    found = true;
                    continue;
                }
                others.armors.push_back(a);
            }
            if (!found) {
                return false;
            }
            target_ = Target(init_target, target_config_);
            updateTarget(others);
            tracker_state = DETECTING;
            return true;
        }
        bool updateTarget(const Armors& armors) noexcept {
            if (armors.armors.empty())
                return false;

            target_.predict(armors.timestamp, armors.v);

            std::vector<Armor> candidates;
            candidates.reserve(armors.armors.size());

            for (const auto& a: armors.armors) {
                if (isSameTarget(a.number, target_.tracked_id_) && !a.is_none_purple) {
                    candidates.emplace_back(a);
                }
            }

            if (candidates.empty())
                return false;

            int updated = 0;
            const auto matches = target_.match(candidates);

            for (const auto& m: matches) {
                if (m.second.is_none_purple) {
                    if (++is_none_purple_count_ > 100)
                        continue;
                } else {
                    is_none_purple_count_ = 0;
                }

                if (target_.update(m))
                    ++updated;
            }

            return updated > 0;
        }

        int lost_thres_;
        int detect_count_ = 0;
        int lost_count_ = 0;
        int is_none_purple_count_ = 0;
        int found_count_ = 0;
        Target target_;
        std::chrono::steady_clock::time_point last_time_;
        TargetConfig::Ptr target_config_;

        double orientationToYaw(const Eigen::Quaterniond& q, double from) noexcept {
            double roll, pitch, yaw;
            Eigen::Vector3d euler = utils::quatToEuler(q, utils::EulerOrder::ZYX, false);
            yaw = euler[0];
            yaw = from + angles::shortest_angular_distance(from, yaw);
            return yaw;
        }
    };
    Tracker::Tracker(wust_vl::common::utils::Parameter::Ptr auto_aim_config_parameter) {
        _impl = std::make_unique<Impl>(auto_aim_config_parameter);
    }
    Tracker::~Tracker() {
        _impl.reset();
    }
    Target Tracker::track(const Armors& armors) noexcept {
        return _impl->track(armors);
    }
    int Tracker::getFoundCount() const noexcept {
        return _impl->found_count_;
    }
    void Tracker::setFoundCount(int count) noexcept {
        _impl->found_count_ = count;
    }
    std::chrono::steady_clock::time_point Tracker::getLastTime() const noexcept {
        return _impl->last_time_;
    }
    void Tracker::setLastTime(std::chrono::steady_clock::time_point t) noexcept {
        _impl->last_time_ = t;
    }

} // namespace auto_aim
} // namespace wust_vision