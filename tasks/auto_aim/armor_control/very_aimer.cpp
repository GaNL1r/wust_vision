#include "very_aimer.hpp"
#include "gcopter/minco.hpp"
#include "tasks/auto_aim/armor_control/tinympc/tiny_api.hpp"
#include "tasks/auto_aim/armor_control/tinympc/types.hpp"
#include "traj.hpp"
#include "very_aimer_base.hpp"
#include "wust_vl/common/utils/manual_compensator.hpp"
namespace auto_aim {
namespace very_aimer {

    class VerAimerTraj: public VeryAimerTrajBase {
    public:
        using Ptr = std::shared_ptr<VerAimerTraj>;
        VerAimerTraj(

        ) {}
        static Ptr create() {
            return std::make_shared<VerAimerTraj>();
        }
        LimitTrajectory target_traj;
        Trajectory<GimbalState> control_traj;
        Trajectory<AimPoint> aim_traj;
        ControlPoint cp0;
        AutoAimFsm fsm;
        Eigen::Vector3d fin_aim_pos;
        AimTarget aim_target;
        GimbalState getTargetState(double t) const override {
            return target_traj.getStateAtTime(t);
        }
        GimbalState getControlState(double t) const override {
            return control_traj.getStateAtTime(t);
        }
    };

    struct VeryAimer::Impl: public VeryAimerBase {
    public:
        Impl(
            const YAML::Node& config,
            std::shared_ptr<TrajectoryCompensator> trajectory_compensator
        ):
            VeryAimerBase(config, trajectory_compensator) {}
        static constexpr int MPC_HORIZON = 300;
        static constexpr double MPC_DT = 1.0 / MPC_HORIZON;
        static constexpr int MPC_HALF_HORIZON = MPC_HORIZON / 2;
        void reset() {
            VeryAimerBase::reset();
            max_iter_ = config_["max_iter"].as<int>();
            auto Q_pitch = config_["Q_pitch"].as<std::vector<double>>();
            auto R_pitch = config_["R_pitch"].as<std::vector<double>>();
            Eigen::MatrixXd A_pitch { { 1, MPC_DT }, { 0, 1 } };
            Eigen::MatrixXd B_pitch { { 0 }, { MPC_DT } };
            Eigen::VectorXd f_pitch { { 0, 0 } };
            Eigen::Matrix<double, 2, 1> Q_p(Q_pitch.data());
            Eigen::Matrix<double, 1, 1> R_p(R_pitch.data());
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
                MPC_HORIZON,
                0
            );

            Eigen::MatrixXd x_min_pitch = Eigen::MatrixXd::Constant(2, MPC_HORIZON, -1e17);
            Eigen::MatrixXd x_max_pitch = Eigen::MatrixXd::Constant(2, MPC_HORIZON, 1e17);
            Eigen::MatrixXd u_min_pitch =
                Eigen::MatrixXd::Constant(1, MPC_HORIZON - 1, -max_pitch_acc_);
            Eigen::MatrixXd u_max_pitch =
                Eigen::MatrixXd::Constant(1, MPC_HORIZON - 1, max_pitch_acc_);
            tiny_set_bound_constraints(
                pitch_solver_,
                x_min_pitch,
                x_max_pitch,
                u_min_pitch,
                u_max_pitch
            );

            pitch_solver_->settings->max_iter = max_iter_;
            auto Q_yaw = config_["Q_yaw"].as<std::vector<double>>();
            auto R_yaw = config_["R_yaw"].as<std::vector<double>>();

            Eigen::MatrixXd A_yaw { { 1, MPC_DT }, { 0, 1 } };
            Eigen::MatrixXd B_yaw { { 0 }, { MPC_DT } };
            Eigen::VectorXd f_yaw { { 0, 0 } };
            Eigen::Matrix<double, 2, 1> Q_y(Q_yaw.data());
            Eigen::Matrix<double, 1, 1> R_y(R_yaw.data());
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
                MPC_HORIZON,
                0
            );

            Eigen::MatrixXd x_min_yaw = Eigen::MatrixXd::Constant(2, MPC_HORIZON, -1e17);
            Eigen::MatrixXd x_max_yaw = Eigen::MatrixXd::Constant(2, MPC_HORIZON, 1e17);
            Eigen::MatrixXd u_min_yaw =
                Eigen::MatrixXd::Constant(1, MPC_HORIZON - 1, -max_yaw_acc_);
            Eigen::MatrixXd u_max_yaw = Eigen::MatrixXd::Constant(1, MPC_HORIZON - 1, max_yaw_acc_);
            tiny_set_bound_constraints(yaw_solver_, x_min_yaw, x_max_yaw, u_min_yaw, u_max_yaw);

            yaw_solver_->settings->max_iter = max_iter_;
        }

        std::pair<LimitTrajectory, Trajectory<AimPoint>> getTrajectory(
            Target& target,
            const ControlPoint& cp0,
            double bullet_speed,
            const AutoAimFsm& auto_aim_fsm
        ) {
            LimitTrajectory traj;
            Trajectory<AimPoint> aim_traj;
            traj.reserve(MPC_HORIZON);
            aim_traj.reserve(MPC_HORIZON);

            // prepare: roll the target back so we start from same relative time as original impl
            target.predict(-MPC_DT * (MPC_HALF_HORIZON + 1));

            // compute first two cps (target state is mutated between calls but choseAndGetControlPoint takes const&)
            auto cp_last = choseAndGetControlPoint(target, bullet_speed, auto_aim_fsm);
            target.predict(MPC_DT);
            auto cp = choseAndGetControlPoint(target, bullet_speed, auto_aim_fsm);

            for (int i = 0; i < MPC_HORIZON; ++i) {
                target.predict(MPC_DT);
                const auto cp_next = choseAndGetControlPoint(target, bullet_speed, auto_aim_fsm);
                GimbalState pt;
                pt.yaw_state.p = angles::normalize_angle(cp.yaw - cp0.yaw);
                pt.pitch_state.p = cp.pitch;
                traj.push_back(pt, MPC_DT);
                AimPoint aim_pt;
                aim_pt.d_angle = cp.xyza[3];
                aim_pt.pos = cp.xyza.head<3>();
                aim_traj.push_back(aim_pt, MPC_DT);
                cp_last = cp;
                cp = cp_next;
            }
            traj.buildSimple();
            return { std::move(traj), std::move(aim_traj) };
        }

        Trajectory<GimbalState> solveTrajectory(const Trajectory<GimbalState>& traj) {
            const double total_time = traj.getTotalDuration();
            const auto trajVecToEigen = [&](const Trajectory<GimbalState>& traj) {
                Eigen::Matrix<double, 4, Eigen::Dynamic> mat(4, MPC_HORIZON);

                const double half_t = total_time * 0.5;
                for (int k = 0; k < MPC_HORIZON; ++k) {
                    int i = k - MPC_HALF_HORIZON;
                    double t = i * MPC_DT + half_t;
                    auto state = traj.getStateAtTime(t);
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
            yaw_solver_->work->Xref = traj_eigen.block(0, 0, 2, MPC_HORIZON);
            tiny_solve(yaw_solver_);

            x0 << traj_eigen(2, 0), traj_eigen(3, 0);
            tiny_set_x0(pitch_solver_, x0);
            pitch_solver_->work->Xref = traj_eigen.block(2, 0, 2, MPC_HORIZON);
            tiny_solve(pitch_solver_);

            Trajectory<GimbalState> control_traj;
            control_traj.reserve(MPC_HORIZON);
            for (int i = 0; i < MPC_HORIZON; i++) {
                GimbalState tp;
                tp.yaw_state.p = yaw_solver_->work->x(0, i);
                tp.yaw_state.v = yaw_solver_->work->x(1, i);
                tp.pitch_state.p = pitch_solver_->work->x(0, i);
                tp.pitch_state.v = pitch_solver_->work->x(1, i);
                tp.yaw_state.a = yaw_solver_->work->u(0, i);
                tp.pitch_state.a = pitch_solver_->work->u(0, i);

                control_traj.push_back(tp, traj.dt_vec[i]);
            }
            return control_traj;
        }

        VerAimerTraj::Ptr buildVerAimerTraj(
            Target& target,
            int fin_target_select,
            double bullet_speed,
            const AutoAimFsm& auto_aim_fsm
        ) {
            VerAimerTraj::Ptr res;
            res = VerAimerTraj::create();

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
                euler.y() = (target.tracked_id_ == armor::ArmorNumber::OUTPOST) ? -0.2618 : 0.2618;
                euler.z() = target.yaw();

                at.ori = utils::eulerToQuat(euler, utils::EulerOrder::ZYX);
                res->aim_target = at;
            }

            const double center_yaw = std::atan2(target.position().y(), target.position().x());

            const double d_angle = angles::shortest_angular_distance(
                center_yaw,
                fin_armors_xyza[fin_target_select][3]
            );

            res->cp0 = getControlPoint(res->fin_aim_pos, d_angle, bullet_speed);
            res->cp0.id_in_target = fin_target_select;
            res->cp0.xyza = fin_armors_xyza[fin_target_select];

            const auto traj = getTrajectory(target, res->cp0, bullet_speed, auto_aim_fsm);
            res->target_traj = traj.first;
            res->aim_traj = traj.second;
            res->control_traj = solveTrajectory(res->target_traj);
            return res;
        }

        GimbalCmd veryAim(Target target, double bullet_speed, const AutoAimFsm& auto_aim_fsm) {
            GimbalCmd cmd;

            const auto [aim_first, aim_center, aim_pair] = getAimStatus(auto_aim_fsm);
            const int roughly_select = selectArmor(target, auto_aim_fsm);

            const auto now = time_utils::now();
            const double dt0 = time_utils::durationSec(target.timestamp_, now);
            target.predict(now + std::chrono::microseconds(int(dt0 * 1e6)));

            const auto ap = target.getArmorPositions();
            double fly_time =
                trajectory_compensator_->getFlyingTime(ap[roughly_select].head<3>(), bullet_speed);

            double prev_fly_time = fly_time;
            std::vector<Target> iteration_target(10, target);

            for (int iter = 0; iter < 10; ++iter) {
                iteration_target[iter].predict(prev_fly_time);

                const int iter_select = selectArmor(iteration_target[iter], auto_aim_fsm);

                const auto iter_poss = iteration_target[iter].getArmorPositions();
                const double iter_fly_time =
                    trajectory_compensator_->getFlyingTime(iter_poss[iter_select], bullet_speed);

                if (std::abs(iter_fly_time - prev_fly_time) < 0.001)
                    break;

                prev_fly_time = iter_fly_time;
            }

            const double predict_time = prev_fly_time + prediction_delay_;
            target.predict(predict_time);
            const auto fin_armors_xyza = target.getArmorPosAndYaw();

            const int fin_target_select = selectArmor(target, auto_aim_fsm);

            VerAimerTraj::Ptr build;
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

            const double half_t = build->target_traj.getPrefixTimeAtIdx(MPC_HALF_HORIZON);

            const auto control_state = build->control_traj.getStateAtTime(half_t);

            const double control_yaw_rad =
                angles::normalize_angle(control_state.yaw_state.p + build->cp0.yaw);

            cmd.yaw = rad2deg(control_yaw_rad);
            cmd.v_yaw = rad2deg(control_state.yaw_state.v);
            cmd.pitch = rad2deg(control_state.pitch_state.p);
            cmd.v_pitch = rad2deg(control_state.pitch_state.v);
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

        int max_iter_ = 10;
        TinySolver* yaw_solver_;
        TinySolver* pitch_solver_;
    };
    VeryAimer::VeryAimer(
        const YAML::Node& config,
        std::shared_ptr<TrajectoryCompensator> trajectory_compensator
    ) {
        _impl = std::make_unique<Impl>(config, trajectory_compensator);
    }
    VeryAimer::~VeryAimer() {
        _impl.reset();
    }
    GimbalCmd
    VeryAimer::veryAim(Target target, double bullet_speed, const AutoAimFsm& auto_aim_fsm) {
        return _impl->veryAim(target, bullet_speed, auto_aim_fsm);
    }
} // namespace very_aimer
} // namespace auto_aim
