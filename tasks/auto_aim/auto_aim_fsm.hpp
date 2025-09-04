#pragma once

#include <yaml-cpp/yaml.h>
enum class AutoAimFsm { AIM_WHOLE_CAR_ARMOR, AIM_WHOLE_CAR_CENTER, AIM_SINGLE_ARMOR };
inline std::string auto_aim_fsm_to_string(AutoAimFsm state) {
    switch (state) {
        case AutoAimFsm::AIM_WHOLE_CAR_ARMOR:
            return "AIM_WHOLE_CAR_ARMOR";
        case AutoAimFsm::AIM_WHOLE_CAR_CENTER:
            return "AIM_WHOLE_CAR_CENTER";
        case AutoAimFsm::AIM_SINGLE_ARMOR:
            return "AIM_SINGLE_ARMOR";
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
    double thres_up_1 = 1.0; // SINGLE -> WHOLE_ARMOR
    double thres_down_1 = 0.5; // WHOLE_ARMOR -> SINGLE

    double thres_up_2 = 6.0; // WHOLE_ARMOR -> CENTER
    double thres_down_2 = 5.0; // CENTER -> WHOLE_ARMOR
    void load(const YAML::Node& config) {
        thres_up_1 = config["auto_aim_fsm"]["thres_up_1"].as<double>(1.0);
        thres_down_1 = config["auto_aim_fsm"]["thres_down_1"].as<double>(0.5);
        thres_up_2 = config["auto_aim_fsm"]["thres_up_2"].as<double>(6.0);
        thres_down_2 = config["auto_aim_fsm"]["thres_down_2"].as<double>(5.0);
        transfer_thresh_ = config["auto_aim_fsm"]["transfer_thresh"].as<int>(5.0);
    }
    AutoAimFsmController(
        double up1 = 1.0,
        double down1 = 0.5,
        double up2 = 2.0,
        double down2 = 1.5,
        int transfer = 3
    ):
        thres_up_1(up1),
        thres_down_1(down1),
        thres_up_2(up2),
        thres_down_2(down2),
        transfer_thresh_(transfer) {}
    void update(double v_yaw, bool target_jumped) {
        if (!target_jumped) {
            fsm_state_ = AutoAimFsm::AIM_SINGLE_ARMOR;
            return;
        }
        switch (fsm_state_) {
            case AutoAimFsm::AIM_SINGLE_ARMOR:
                if (std::abs(v_yaw) > thres_up_1)
                    ++overflow_count_;
                else
                    overflow_count_ = 0;

                if (overflow_count_ > transfer_thresh_) {
                    fsm_state_ = AutoAimFsm::AIM_WHOLE_CAR_ARMOR;
                    overflow_count_ = 0;
                }
                break;

            case AutoAimFsm::AIM_WHOLE_CAR_ARMOR:
                if (std::abs(v_yaw) > thres_up_2)
                    ++overflow_count_;
                else if (std::abs(v_yaw) < thres_down_1)
                    ++overflow_count_;
                else
                    overflow_count_ = 0;

                if (overflow_count_ > transfer_thresh_) {
                    if (std::abs(v_yaw) > thres_up_2) {
                        fsm_state_ = AutoAimFsm::AIM_WHOLE_CAR_CENTER;
                    } else {
                        fsm_state_ = AutoAimFsm::AIM_SINGLE_ARMOR;
                    }
                    overflow_count_ = 0;
                }
                break;

            case AutoAimFsm::AIM_WHOLE_CAR_CENTER:
                if (std::abs(v_yaw) < thres_down_2)
                    ++overflow_count_;
                else
                    overflow_count_ = 0;

                if (overflow_count_ > transfer_thresh_) {
                    fsm_state_ = AutoAimFsm::AIM_WHOLE_CAR_ARMOR;
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
