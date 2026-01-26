#pragma once

#include "tasks/auto_aim/armor_control/very_aimer_base.hpp"
#include "tasks/auto_aim/armor_tracker/target.hpp"
#include "tasks/auto_aim/auto_aim_fsm.hpp"
#include "tasks/type_common.hpp"
#include "wust_vl/common/utils/trajectory_compensator.hpp"
namespace wust_vision {
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

        VeryAimerSeg(wust_vl::common::utils::Parameter::Ptr auto_aim_config_parameter):
            VeryAimerBase(auto_aim_config_parameter) {
            reset();
        }

        static Ptr create(wust_vl::common::utils::Parameter::Ptr auto_aim_config_parameter) {
            return std::make_unique<VeryAimerSeg>(auto_aim_config_parameter);
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
} // namespace wust_vision