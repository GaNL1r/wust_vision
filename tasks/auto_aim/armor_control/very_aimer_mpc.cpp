#include "very_aimer_mpc.hpp"
namespace wust_vision {
namespace auto_aim {
    void VeryAimerMpc::reset() {
        VeryAimerBase::reset();
        Eigen::MatrixXd A_pitch { { 1, DT }, { 0, 1 } };
        Eigen::MatrixXd B_pitch { { 0 }, { DT } };
        Eigen::VectorXd f_pitch { { 0, 0 } };
        Eigen::Matrix<double, 2, 1> Q_p(config_->mpc.Q_pitch.data());
        Eigen::Matrix<double, 1, 1> R_p(config_->mpc.R_pitch.data());
        tiny_setup(
            &pitch_solver_,
            A_pitch,
            B_pitch,
            f_pitch,
            Q_p.asDiagonal(),
            R_p.asDiagonal(),
            1.0,
            2,
            1,
            HORIZON,
            0
        );

        Eigen::MatrixXd x_min_pitch = Eigen::MatrixXd::Constant(2, HORIZON, -1e17);
        Eigen::MatrixXd x_max_pitch = Eigen::MatrixXd::Constant(2, HORIZON, 1e17);
        Eigen::MatrixXd u_min_pitch =
            Eigen::MatrixXd::Constant(1, HORIZON - 1, -config_->max_pitch_acc_param.get());
        Eigen::MatrixXd u_max_pitch =
            Eigen::MatrixXd::Constant(1, HORIZON - 1, config_->max_pitch_acc_param.get());
        tiny_set_bound_constraints(
            pitch_solver_,
            x_min_pitch,
            x_max_pitch,
            u_min_pitch,
            u_max_pitch
        );
        pitch_solver_->settings->max_iter = max_iter_;
        Eigen::MatrixXd A_yaw { { 1, DT }, { 0, 1 } };
        Eigen::MatrixXd B_yaw { { 0 }, { DT } };
        Eigen::VectorXd f_yaw { { 0, 0 } };
        Eigen::Matrix<double, 2, 1> Q_y(config_->mpc.Q_yaw.data());
        Eigen::Matrix<double, 1, 1> R_y(config_->mpc.R_yaw.data());
        tiny_setup(
            &yaw_solver_,
            A_yaw,
            B_yaw,
            f_yaw,
            Q_y.asDiagonal(),
            R_y.asDiagonal(),
            1.0,
            2,
            1,
            HORIZON,
            0
        );

        Eigen::MatrixXd x_min_yaw = Eigen::MatrixXd::Constant(2, HORIZON, -1e17);
        Eigen::MatrixXd x_max_yaw = Eigen::MatrixXd::Constant(2, HORIZON, 1e17);
        Eigen::MatrixXd u_min_yaw =
            Eigen::MatrixXd::Constant(1, HORIZON - 1, -config_->max_yaw_acc_param.get());
        Eigen::MatrixXd u_max_yaw =
            Eigen::MatrixXd::Constant(1, HORIZON - 1, config_->max_yaw_acc_param.get());
        tiny_set_bound_constraints(yaw_solver_, x_min_yaw, x_max_yaw, u_min_yaw, u_max_yaw);
        yaw_solver_->settings->max_iter = max_iter_;
    }

    Trajectory<GimbalState> VeryAimerMpc::solveTrajectory(const Trajectory<GimbalState>& traj) {
        const double total_time = traj.getTotalDuration();
        const auto trajVecToEigen = [&](const Trajectory<GimbalState>& traj) {
            Eigen::Matrix<double, 4, Eigen::Dynamic> mat(4, HORIZON);

            const double half_t = total_time * 0.5;
            for (int k = 0; k < HORIZON; ++k) {
                int i = k - HALF_HORIZON;
                double t = i * DT + half_t;
                auto state = traj.Trajectory::getStateAtTime(t);
                mat(0, k) = state.yaw_state.p;
                mat(1, k) = state.yaw_state.v;
                mat(2, k) = state.pitch_state.p;
                mat(3, k) = state.pitch_state.v;
            }
            return mat;
        };

        auto traj_eigen = trajVecToEigen(traj);
        Eigen::VectorXd x0(2);
        x0 << traj_eigen(0, 0), traj_eigen(1, 0);
        tiny_set_x0(yaw_solver_, x0);
        yaw_solver_->work->Xref = traj_eigen.block(0, 0, 2, HORIZON);
        tiny_solve(yaw_solver_);

        x0 << traj_eigen(2, 0), traj_eigen(3, 0);
        tiny_set_x0(pitch_solver_, x0);
        pitch_solver_->work->Xref = traj_eigen.block(2, 0, 2, HORIZON);
        tiny_solve(pitch_solver_);
        Trajectory<GimbalState> control_traj;
        control_traj.reserve(HORIZON);
        for (int i = 0; i < HORIZON; i++) {
            GimbalState tp;
            tp.yaw_state.p = yaw_solver_->work->x(0, i);
            tp.yaw_state.v = yaw_solver_->work->x(1, i);
            tp.pitch_state.p = pitch_solver_->work->x(0, i);
            tp.pitch_state.v = pitch_solver_->work->x(1, i);
            tp.yaw_state.a = yaw_solver_->work->u(0, i);
            tp.pitch_state.a = pitch_solver_->work->u(0, i);
            control_traj.push_back(tp, DT);
        }
        return control_traj;
    }

    VeryAimerTrajMpc::Ptr VeryAimerMpc::buildVerAimerTraj(
        Target& target,
        int fin_target_select,
        double bullet_speed,
        const AutoAimFsm& auto_aim_fsm
    ) {
        auto res = VeryAimerTrajMpc::create();

        const auto fin_armors_xyza = target.getArmorPosAndYaw();
        res->fin_aim_pos = fin_armors_xyza[fin_target_select].head<3>();

        const auto [aim_first, aim_center, aim_pair] = getAimStatus(auto_aim_fsm);
        if (aim_center) {
            const double raw_z = res->fin_aim_pos.z();
            double c_xy_dis = std::hypot(target.position().x(), target.position().y());
            const double c_yaw = std::atan2(target.position().y(), target.position().x());

            c_xy_dis -= target.getArmor2CenterXYDis(fin_target_select);

            res->fin_aim_pos.x() = c_xy_dis * std::cos(c_yaw);
            res->fin_aim_pos.y() = c_xy_dis * std::sin(c_yaw);
            res->fin_aim_pos.z() = raw_z;
        }

        {
            AimTarget at;
            at.pos = res->fin_aim_pos;
            at.valid = true;

            Eigen::Vector3d euler;
            euler.x() = M_PI / 2.0;
            euler.y() = (target.tracked_id_ == ArmorNumber::OUTPOST) ? -0.2618 : 0.2618;
            euler.z() = target.yaw();

            at.ori = utils::eulerToQuat(euler, utils::EulerOrder::ZYX);
            res->aim_target = at;
        }

        const double center_yaw = std::atan2(target.position().y(), target.position().x());

        const double d_angle =
            angles::shortest_angular_distance(center_yaw, fin_armors_xyza[fin_target_select][3]);

        res->cp0 = getControlPoint(res->fin_aim_pos, d_angle, bullet_speed);
        res->cp0.id_in_target = fin_target_select;
        res->cp0.xyza = fin_armors_xyza[fin_target_select];

        auto traj = getTrajectory(target, res->cp0, bullet_speed, auto_aim_fsm);
        res->target_traj = traj.first;
        res->aim_traj = traj.second;
        res->control_traj = solveTrajectory(res->target_traj);
        return res;
    }

    GimbalCmd
    VeryAimerMpc::veryAim(Target target, double bullet_speed, const AutoAimFsm& auto_aim_fsm) {
        GimbalCmd cmd;
        if (!trajectory_compensator_config_->trajectory_compensator) {
            cmd.appera = false;
            return cmd;
        }
        const auto [aim_first, aim_center, aim_pair] = getAimStatus(auto_aim_fsm);
        const int roughly_select = selectArmor(target, auto_aim_fsm);

        const auto now = wust_vl::common::utils::time_utils::now();
        target.predictSimple(now);

        const auto ap = target.getArmorPositions();
        const double fly_time = trajectory_compensator_config_->trajectory_compensator
                                    ->getFlyingTime(ap[roughly_select].head<3>(), bullet_speed);

        double prev_fly_time = fly_time;
        std::vector<Target> iteration_target(10, target);

        for (int iter = 0; iter < 10; ++iter) {
            iteration_target[iter].predictSimple(prev_fly_time);

            const int iter_select = selectArmor(iteration_target[iter], auto_aim_fsm);

            const auto iter_poss = iteration_target[iter].getArmorPositions();
            const double iter_fly_time = trajectory_compensator_config_->trajectory_compensator
                                             ->getFlyingTime(iter_poss[iter_select], bullet_speed);

            if (std::abs(iter_fly_time - prev_fly_time) < 0.001)
                break;

            prev_fly_time = iter_fly_time;
        }

        const double predict_time = prev_fly_time + config_->prediction_delay_param.get();
        target.predictSimple(predict_time);
        const auto fin_armors_xyza = target.getArmorPosAndYaw();

        const int fin_target_select = selectArmor(target, auto_aim_fsm);

        VeryAimerTrajMpc::Ptr build;
        try {
            build = buildVerAimerTraj(target, fin_target_select, bullet_speed, auto_aim_fsm);
        } catch (const std::exception& e) {
            WUST_WARN("very_aimer") << "mpc solver error: " << e.what();
            cmd.appera = false;
            return cmd;
        } catch (...) {
            WUST_WARN("very_aimer") << "mpc solver unknown error";
            cmd.appera = false;
            return cmd;
        }

        cmd.aim_target = build->aim_target;
        double target_yaw_rad = build->cp0.yaw;
        double target_pitch_rad = build->cp0.pitch;

        if (aim_center) {
            const double center_yaw = std::atan2(target.position().y(), target.position().x());

            const double d_angle = angles::shortest_angular_distance(
                center_yaw,
                fin_armors_xyza[fin_target_select][3]
            );
            auto cp_center = getControlPoint(
                fin_armors_xyza[fin_target_select].head<3>(),
                d_angle,
                bullet_speed
            );
            target_yaw_rad = cp_center.yaw;
            target_pitch_rad = cp_center.pitch;
        }

        const double half_t = build->target_traj.getPrefixTimeAtIdx(HALF_HORIZON);

        const auto control_state = build->getControlState(half_t);

        const double control_yaw_rad =
            angles::normalize_angle(control_state.yaw_state.p + build->cp0.yaw);

        cmd.yaw = rad2deg(control_yaw_rad);
        cmd.v_yaw = rad2deg(control_state.yaw_state.v);
        cmd.pitch = rad2deg(control_state.pitch_state.p);
        cmd.v_pitch = rad2deg(control_state.pitch_state.v);
        cmd.a_yaw = rad2deg(control_state.yaw_state.a);
        cmd.a_pitch = rad2deg(control_state.pitch_state.a);
        cmd.target_yaw = rad2deg(target_yaw_rad);
        cmd.target_pitch = rad2deg(target_pitch_rad);
        cmd.raw_yaw = cmd.target_yaw;
        cmd.raw_pitch = cmd.target_pitch;
        cmd.distance = build->fin_aim_pos.norm();
        cmd.fly_time = prev_fly_time;

        const auto fire_now = VeryAimerBase::canFireAtTime(build, half_t);
        cmd.enable_yaw_diff = fire_now.enable_yaw_diff;
        cmd.enable_pitch_diff = fire_now.enable_pitch_diff;
        cmd.fire_advice = fire_now.fire;

        cmd.appera = cmd.isValid();
        if (!cmd.appera) {
            reset();
            WUST_WARN("very_aimer") << "very_aimer nan!";
        }

        return cmd;
    }

} // namespace auto_aim
} // namespace wust_vision