#pragma once

#include <yaml-cpp/yaml.h>
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

    // 上下阈值
    double single_whole_up = 1.0;
    double single_whole_down = 0.5;

    double whole_pair_up = 6.0;
    double whole_pair_down = 5.0;

    double pair_center_up = 10.0;
    double pair_center_down = 8.5;
    void load(const YAML::Node& config) {
        transfer_thresh_ = config["auto_aim_fsm"]["transfer_thresh"].as<int>(5.0);
        single_whole_up = config["auto_aim_fsm"]["single_whole_up"].as<double>(1.0);
        single_whole_down = config["auto_aim_fsm"]["single_whole_down"].as<double>(0.5);
        whole_pair_up = config["auto_aim_fsm"]["whole_pair_up"].as<double>(6.0);
        whole_pair_down = config["auto_aim_fsm"]["whole_pair_down"].as<double>(5.0);
        pair_center_up = config["auto_aim_fsm"]["pair_center_up"].as<double>(10.0);
        pair_center_down = config["auto_aim_fsm"]["pair_center_down"].as<double>(8.5);
    }
    AutoAimFsmController(

    ) {}
    void update(double v_yaw, bool target_jumped) {
        if (!target_jumped) {
            fsm_state_ = AutoAimFsm::AIM_SINGLE_ARMOR;
            return;
        }
        // double r_per_sec = v_yaw / (2.0*M_PI);
        switch (fsm_state_) {
            case AutoAimFsm::AIM_SINGLE_ARMOR:
                if (std::abs(v_yaw) > single_whole_up)
                    ++overflow_count_;
                else
                    overflow_count_ = 0;

                if (overflow_count_ > transfer_thresh_) {
                    fsm_state_ = AutoAimFsm::AIM_WHOLE_CAR_ARMOR;
                    overflow_count_ = 0;
                }
                break;

            case AutoAimFsm::AIM_WHOLE_CAR_ARMOR:
                if (std::abs(v_yaw) > whole_pair_up)
                    ++overflow_count_;
                else if (std::abs(v_yaw) < single_whole_down)
                    ++overflow_count_;
                else
                    overflow_count_ = 0;

                if (overflow_count_ > transfer_thresh_) {
                    if (std::abs(v_yaw) > whole_pair_up) {
                        fsm_state_ = AutoAimFsm::AIM_WHOLE_CAR_PAIR;

                    } else {
                        fsm_state_ = AutoAimFsm::AIM_SINGLE_ARMOR;
                    }
                    overflow_count_ = 0;
                }
                break;
            case AutoAimFsm::AIM_WHOLE_CAR_PAIR:
                if (std::abs(v_yaw) > pair_center_up)
                    ++overflow_count_;
                else if (std::abs(v_yaw) < whole_pair_down)
                    ++overflow_count_;
                else
                    overflow_count_ = 0;

                if (overflow_count_ > transfer_thresh_) {
                    if (std::abs(v_yaw) > pair_center_up) {
                        fsm_state_ = AutoAimFsm::AIM_WHOLE_CAR_CENTER;

                    } else {
                        fsm_state_ = AutoAimFsm::AIM_WHOLE_CAR_ARMOR;
                    }
                    overflow_count_ = 0;
                }
                break;
            case AutoAimFsm::AIM_WHOLE_CAR_CENTER:
                if (std::abs(v_yaw) < pair_center_down)
                    ++overflow_count_;
                else
                    overflow_count_ = 0;

                if (overflow_count_ > transfer_thresh_) {
                    fsm_state_ = AutoAimFsm::AIM_WHOLE_CAR_PAIR;
                    overflow_count_ = 0;
                }
                break;

            default:
                fsm_state_ = AutoAimFsm::AIM_SINGLE_ARMOR;
                overflow_count_ = 0;
                break;
        }
    }
};
