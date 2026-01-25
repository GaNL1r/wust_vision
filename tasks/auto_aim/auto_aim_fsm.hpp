#pragma once

#include <cmath>
#include <string>
#include <yaml-cpp/yaml.h>
namespace auto_aim {
enum class AutoAimFsm {
    AIM_WHOLE_CAR_ARMOR,
    AIM_WHOLE_CAR_CENTER,
    AIM_SINGLE_ARMOR,
    AIM_WHOLE_CAR_PAIR
};

inline std::string auto_aim_fsm_to_string(AutoAimFsm state) {
    switch (state) {
        case AutoAimFsm::AIM_WHOLE_CAR_ARMOR:
            return "AIM_WHOLE_CAR_ARMOR";
        case AutoAimFsm::AIM_WHOLE_CAR_CENTER:
            return "AIM_WHOLE_CAR_CENTER";
        case AutoAimFsm::AIM_SINGLE_ARMOR:
            return "AIM_SINGLE_ARMOR";
        case AutoAimFsm::AIM_WHOLE_CAR_PAIR:
            return "AIM_WHOLE_CAR_PAIR";
        default:
            return "UNKNOWN";
    }
}

class AutoAimFsmController {
public:
    AutoAimFsm fsm_state_ { AutoAimFsm::AIM_SINGLE_ARMOR };
    int overflow_count_ = 0;
    int transfer_thresh_ = 5; // 防抖计数阈值

    // hysteresis thresholds
    double single_whole_up = 1.0;
    double single_whole_down = 0.5;

    double whole_pair_up = 6.0;
    double whole_pair_down = 5.0;

    double pair_center_up = 10.0;
    double pair_center_down = 8.5;

    AutoAimFsmController() = default;

    void load(const YAML::Node& config) {
        const auto node = config["auto_aim_fsm"];
        transfer_thresh_ = node["transfer_thresh"].as<int>(5);
        single_whole_up = node["single_whole_up"].as<double>(1.0);
        single_whole_down = node["single_whole_down"].as<double>(0.5);
        whole_pair_up = node["whole_pair_up"].as<double>(6.0);
        whole_pair_down = node["whole_pair_down"].as<double>(5.0);
        pair_center_up = node["pair_center_up"].as<double>(10.0);
        pair_center_down = node["pair_center_down"].as<double>(8.5);
    }

    void update(double v_yaw, bool target_jumped) {
        // 无跳变：直接退回单装甲，并清状态
        if (!target_jumped) {
            fsm_state_ = AutoAimFsm::AIM_SINGLE_ARMOR;
            overflow_count_ = 0;
            return;
        }

        const double av = std::abs(v_yaw);

        switch (fsm_state_) {
            case AutoAimFsm::AIM_SINGLE_ARMOR: {
                overflow_count_ = (av > single_whole_up) ? overflow_count_ + 1 : 0;
                if (overflow_count_ > transfer_thresh_) {
                    fsm_state_ = AutoAimFsm::AIM_WHOLE_CAR_ARMOR;
                    overflow_count_ = 0;
                }
                break;
            }

            case AutoAimFsm::AIM_WHOLE_CAR_ARMOR: {
                if (av > whole_pair_up)
                    ++overflow_count_;
                else if (av < single_whole_down)
                    --overflow_count_;
                else
                    overflow_count_ = 0;

                if (std::abs(overflow_count_) > transfer_thresh_) {
                    fsm_state_ = (overflow_count_ > 0) ? AutoAimFsm::AIM_WHOLE_CAR_PAIR
                                                       : AutoAimFsm::AIM_SINGLE_ARMOR;
                    overflow_count_ = 0;
                }
                break;
            }

            case AutoAimFsm::AIM_WHOLE_CAR_PAIR: {
                if (av > pair_center_up)
                    ++overflow_count_;
                else if (av < whole_pair_down)
                    --overflow_count_;
                else
                    overflow_count_ = 0;

                if (std::abs(overflow_count_) > transfer_thresh_) {
                    fsm_state_ = (overflow_count_ > 0) ? AutoAimFsm::AIM_WHOLE_CAR_CENTER
                                                       : AutoAimFsm::AIM_WHOLE_CAR_ARMOR;
                    overflow_count_ = 0;
                }
                break;
            }

            case AutoAimFsm::AIM_WHOLE_CAR_CENTER: {
                overflow_count_ = (av < pair_center_down) ? overflow_count_ + 1 : 0;
                if (overflow_count_ > transfer_thresh_) {
                    fsm_state_ = AutoAimFsm::AIM_WHOLE_CAR_PAIR;
                    overflow_count_ = 0;
                }
                break;
            }

            default:
                fsm_state_ = AutoAimFsm::AIM_SINGLE_ARMOR;
                overflow_count_ = 0;
                break;
        }
    }
};
} // namespace auto_aim