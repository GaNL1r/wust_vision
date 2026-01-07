#pragma once

#include "tasks/auto_aim/armor_tracker/target.hpp"
#include "tasks/auto_aim/auto_aim_fsm.hpp"
#include "tasks/type_common.hpp"
#include "wust_vl/common/utils/trajectory_compensator.hpp"

class VeryAimer {
public:
    using Ptr = std::unique_ptr<VeryAimer>;

    VeryAimer(
        const YAML::Node& config,
        std::shared_ptr<TrajectoryCompensator> trajectory_compensator
    );
    ~VeryAimer();
    static Ptr create(
        const YAML::Node& config,
        std::shared_ptr<TrajectoryCompensator> trajectory_compensator
    ) {
        return std::make_unique<VeryAimer>(config, trajectory_compensator);
    }
    GimbalCmd veryAim(Target target, double bullet_speed, const AutoAimFsm& auto_aim_fsm);
    struct Impl;
    std::unique_ptr<Impl> _impl;
};