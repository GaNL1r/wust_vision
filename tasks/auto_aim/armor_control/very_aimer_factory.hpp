#pragma once
#include "very_aimer_mpc.hpp"
#include "very_aimer_seg.hpp"
namespace wust_vision {
namespace auto_aim {
    class VeryAimerFactory {
    public:
        static VeryAimerBase::Ptr create(
            const YAML::Node& config,
            wust_vl::common::utils::Parameter::Ptr auto_aim_config_parameter
        ) {
            const std::string very_aimer_type = config["type"].as<std::string>();
            if (very_aimer_type == "seg" || very_aimer_type == "SEG") {
                return VeryAimerSeg::create(auto_aim_config_parameter);
            } else if (very_aimer_type == "mpc" || very_aimer_type == "MPC") {
                return VeryAimerMpc::create(auto_aim_config_parameter);
            } else {
                return VeryAimerSeg::create(auto_aim_config_parameter);
            }
        }
    };

} // namespace auto_aim
} // namespace wust_vision