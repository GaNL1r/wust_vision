#pragma once

#include "tasks/auto_aim/armor_tracker/target.hpp"
#include "tasks/auto_aim/auto_aim_fsm.hpp"
#include "tasks/type_common.hpp"
#include "wust_vl/common/utils/trajectory_compensator.hpp"
namespace auto_aim {
namespace very_aimer_opt {
    class VeryAimerOpt {
    public:
        using Ptr = std::unique_ptr<VeryAimerOpt>;

        VeryAimerOpt(
            const YAML::Node& config,
            std::shared_ptr<TrajectoryCompensator> trajectory_compensator
        );
        ~VeryAimerOpt();
        static Ptr create(
            const YAML::Node& config,
            std::shared_ptr<TrajectoryCompensator> trajectory_compensator
        ) {
            return std::make_unique<VeryAimerOpt>(config, trajectory_compensator);
        }
        GimbalCmd veryAim(Target target, double bullet_speed, const AutoAimFsm& auto_aim_fsm);
        class Impl;
        std::unique_ptr<Impl> _impl;
    };
} // namespace very_aimer_opt

}; // namespace auto_aim
