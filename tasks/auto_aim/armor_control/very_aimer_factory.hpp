#pragma once
#include "very_aimer_mpc.hpp"
#include "very_aimer_seg.hpp"

namespace auto_aim {
class VeryAimerFactory {
public:
    static VeryAimerBase::Ptr create(
        const YAML::Node& config,
        std::shared_ptr<TrajectoryCompensator> trajectory_compensator
    ) {
        const std::string very_aimer_type = config["type"].as<std::string>();
        if (very_aimer_type == "seg" || very_aimer_type == "SEG") {
            return VeryAimerSeg::create(config, trajectory_compensator);
        } else if (very_aimer_type == "mpc" || very_aimer_type == "MPC") {
            return VeryAimerMpc::create(config, trajectory_compensator);
        } else {
            return VeryAimerSeg::create(config, trajectory_compensator);
        }
    }
};

} // namespace auto_aim