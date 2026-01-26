#pragma once

#include "wust_vl/common/utils/parameter.hpp"
namespace wust_vision {
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
        AutoAimFsmController() {
            config_ = std::make_shared<AutoAimFsmConfig>();
        }
        AutoAimFsm fsm_state_ { AutoAimFsm::AIM_SINGLE_ARMOR };
        struct AutoAimFsmConfig: wust_vl::common::utils::ParamGroup {
        public:
            static constexpr const char* Logger = "Config: auto_aim::auto_aim_fsm";
            static constexpr const char* kKey = "auto_aim_fsm";
            const char* key() const override {
                return kKey;
            }
            using Ptr = std::shared_ptr<AutoAimFsmConfig>;
            AutoAimFsmConfig() {}
            GEN_PARAM(int, transfer_thresh);
            GEN_PARAM(double, single_whole_up);
            GEN_PARAM(double, single_whole_down);
            GEN_PARAM(double, whole_pair_up);
            GEN_PARAM(double, whole_pair_down);
            GEN_PARAM(double, pair_center_up);
            GEN_PARAM(double, pair_center_down);
            void loadSelf(const YAML::Node& node) override {
                transfer_thresh_param.load(node);
                single_whole_up_param.load(node);
                single_whole_down_param.load(node);
                whole_pair_up_param.load(node);
                whole_pair_down_param.load(node);
                pair_center_up_param.load(node);
                pair_center_down_param.load(node);
            }
        };
        AutoAimFsmConfig::Ptr config_;
        int overflow_count_ = 0;

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
                    overflow_count_ =
                        (av > config_->single_whole_up_param.get()) ? overflow_count_ + 1 : 0;
                    if (overflow_count_ > config_->transfer_thresh_param.get()) {
                        fsm_state_ = AutoAimFsm::AIM_WHOLE_CAR_ARMOR;
                        overflow_count_ = 0;
                    }
                    break;
                }

                case AutoAimFsm::AIM_WHOLE_CAR_ARMOR: {
                    if (av > config_->whole_pair_up_param.get())
                        ++overflow_count_;
                    else if (av < config_->single_whole_down_param.get())
                        --overflow_count_;
                    else
                        overflow_count_ = 0;

                    if (std::abs(overflow_count_) > config_->transfer_thresh_param.get()) {
                        fsm_state_ = (overflow_count_ > 0) ? AutoAimFsm::AIM_WHOLE_CAR_PAIR
                                                           : AutoAimFsm::AIM_SINGLE_ARMOR;
                        overflow_count_ = 0;
                    }
                    break;
                }

                case AutoAimFsm::AIM_WHOLE_CAR_PAIR: {
                    if (av > config_->pair_center_up_param.get())
                        ++overflow_count_;
                    else if (av < config_->whole_pair_down_param.get())
                        --overflow_count_;
                    else
                        overflow_count_ = 0;

                    if (std::abs(overflow_count_) > config_->transfer_thresh_param.get()) {
                        fsm_state_ = (overflow_count_ > 0) ? AutoAimFsm::AIM_WHOLE_CAR_CENTER
                                                           : AutoAimFsm::AIM_WHOLE_CAR_ARMOR;
                        overflow_count_ = 0;
                    }
                    break;
                }

                case AutoAimFsm::AIM_WHOLE_CAR_CENTER: {
                    overflow_count_ =
                        (av < config_->pair_center_down_param.get()) ? overflow_count_ + 1 : 0;
                    if (overflow_count_ > config_->transfer_thresh_param.get()) {
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
} // namespace wust_vision