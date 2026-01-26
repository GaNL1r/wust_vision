#pragma once

#include "tasks/auto_aim/armor_control/tinympc/tiny_api.hpp"
#include "tasks/auto_aim/armor_control/tinympc/types.hpp"
#include "tasks/auto_aim/armor_control/very_aimer_base.hpp"
#include "tasks/auto_aim/armor_tracker/target.hpp"
#include "tasks/auto_aim/auto_aim_fsm.hpp"
#include "tasks/type_common.hpp"
#include "wust_vl/common/utils/trajectory_compensator.hpp"
namespace wust_vision {
namespace auto_aim {
    class VeryAimerTrajMpc: public VeryAimerTrajBase {
    public:
        using Ptr = std::shared_ptr<VeryAimerTrajMpc>;
        VeryAimerTrajMpc(

        ) {}
        static Ptr create() {
            return std::make_shared<VeryAimerTrajMpc>();
        }
        LimitTrajectory target_traj;
        Trajectory<GimbalState> control_traj;
        Trajectory<AimPoint> aim_traj;
        ControlPoint cp0;
        AutoAimFsm fsm;
        Eigen::Vector3d fin_aim_pos;
        AimTarget aim_target;
        GimbalState getTargetState(double t) const override {
            return target_traj.Trajectory::getStateAtTime(t);
        }
        GimbalState getControlState(double t) const override {
            return control_traj.getStateAtTime(t);
        }
    };
    class VeryAimerMpc: public VeryAimerBase {
    public:
        using Ptr = std::unique_ptr<VeryAimerMpc>;

        VeryAimerMpc(wust_vl::common::utils::Parameter::Ptr auto_aim_config_parameter):
            VeryAimerBase(auto_aim_config_parameter) {
            reset();
        }

        static Ptr create(wust_vl::common::utils::Parameter::Ptr auto_aim_config_parameter) {
            return std::make_unique<VeryAimerMpc>(auto_aim_config_parameter);
        }
        void reset();

        Trajectory<GimbalState> solveTrajectory(const Trajectory<GimbalState>& traj);
        VeryAimerTrajMpc::Ptr buildVerAimerTraj(
            Target& target,
            int fin_target_select,
            double bullet_speed,
            const AutoAimFsm& auto_aim_fsm
        );

        GimbalCmd
        veryAim(Target target, double bullet_speed, const AutoAimFsm& auto_aim_fsm) override;

        int max_iter_ = 10;
        TinySolver* yaw_solver_;
        TinySolver* pitch_solver_;
    };

}; // namespace auto_aim
} // namespace wust_vision
