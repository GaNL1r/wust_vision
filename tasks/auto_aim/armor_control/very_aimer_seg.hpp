#pragma once

#include "tasks/auto_aim/armor_control/very_aimer_base.hpp"
#include "tasks/auto_aim/armor_tracker/target.hpp"
#include "tasks/auto_aim/auto_aim_fsm.hpp"
#include "tasks/type_common.hpp"
#include "wust_vl/common/utils/trajectory_compensator.hpp"
namespace auto_aim {
class VeryAimerTrajSeg: public VeryAimerTrajBase {
public:
    using Ptr = std::shared_ptr<VeryAimerTrajSeg>;
    VeryAimerTrajSeg() {}
    static Ptr create() {
        return std::make_shared<VeryAimerTrajSeg>();
    }
    GimbalState getTargetState(double t) const override {
        return target_traj.Trajectory::getStateAtTime(t);
    }
    GimbalState getControlState(double t) const override {
        return target_traj.LimitTrajectory::getStateAtTime(t);
    }
};
class VeryAimerSeg: public VeryAimerBase {
public:
    using Ptr = std::unique_ptr<VeryAimerSeg>;

    VeryAimerSeg(
        const YAML::Node& config,
        std::shared_ptr<wust_vl::common::utils::TrajectoryCompensator> trajectory_compensator
    ):
        VeryAimerBase(config, trajectory_compensator) {
        reset();
    }

    static Ptr create(
        const YAML::Node& config,
        std::shared_ptr<wust_vl::common::utils::TrajectoryCompensator> trajectory_compensator
    ) {
        return std::make_unique<VeryAimerSeg>(config, trajectory_compensator);
    }
    VeryAimerTrajSeg::Ptr buildVerAimerTraj(
        Target& target,
        int fin_target_select,
        double bullet_speed,
        const AutoAimFsm& auto_aim_fsm
    ) const;
    GimbalCmd veryAim(Target target, double bullet_speed, const AutoAimFsm& auto_aim_fsm);
};

}; // namespace auto_aim
