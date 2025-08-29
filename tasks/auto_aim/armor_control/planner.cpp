#include "planner.hpp"
#include <wust_vl/common/utils/timer.hpp>
Planner::Planner(
    const YAML::Node& config,
    std::shared_ptr<Aimer> aimer,
    std::shared_ptr<Shooter> shooter
) {
    aimer_ = aimer;
    shooter_ = shooter;
    setup_yaw_solver(config);
    setup_pitch_solver(config);
}
Plan Planner::plan(Target target, double bullet_speed, const AutoAimFsm& auto_aim_fsm) {
    has_changed_ = false;
    // 0. Check bullet speed
    if (bullet_speed < 10 || bullet_speed > 25) {
        bullet_speed = 22;
    }
    auto now = time_utils::now();
    double dt0 = time_utils::durationSec(target.timestamp_, now);
    std::chrono::steady_clock::time_point future = now + std::chrono::microseconds(int(dt0 * 1e6));
    target.predict(future);
    auto p_armors = target.getArmorPositions();
    auto v_armors = target.getArmorVelocities();
    bool aim_first = false;
    if (auto_aim_fsm == AutoAimFsm::AIM_SINGLE_ARMOR) {
        aim_first = true;
    }
    int idx = aimer_->selectBestArmor(target, aim_first);
    if (idx < 0 || idx >= (int)p_armors.size() || p_armors[idx].norm() < 0.1) {
        return { false };
    } else {
        bool converged = false;
        double prev_fly_time = aimer_->getFlyingTime(p_armors[idx], bullet_speed);
        auto pre_time0 = future + std::chrono::microseconds(static_cast<int>(prev_fly_time * 1e6));
        target.predict(pre_time0);
    }
    bool aim_center = false;
    if (auto_aim_fsm == AutoAimFsm::AIM_WHOLE_CAR_CENTER) {
        aim_center = true;
    }
    double yaw0;
    Trajectory traj;

    try {
        yaw0 = aim(target, bullet_speed, idx, aim_center, aim_first)(0);
        traj = get_trajectory(target, yaw0, bullet_speed, idx, aim_center, aim_first);
    } catch (const std::exception& e) {
        return { false };
    }

    // 3. Solve yaw
    Eigen::VectorXd x0(2);
    x0 << traj(0, 0), traj(1, 0);
    tiny_set_x0(yaw_solver_, x0);

    yaw_solver_->work->Xref = traj.block(0, 0, 2, HORIZON);
    tiny_solve(yaw_solver_);

    // 4. Solve pitch
    x0 << traj(2, 0), traj(3, 0);
    tiny_set_x0(pitch_solver_, x0);

    pitch_solver_->work->Xref = traj.block(2, 0, 2, HORIZON);
    tiny_solve(pitch_solver_);
    if (!yaw_solver_->work->status || !pitch_solver_->work->status) {
        WUST_WARN("planner") << "mpc solver error!";
        return { false };
    }
    Plan plan;

    plan.control = true;

    plan.target_yaw = angles::normalize_angle(traj(0, HALF_HORIZON) + yaw0);
    plan.target_pitch = traj(2, HALF_HORIZON);

    plan.yaw = angles::normalize_angle(yaw_solver_->work->x(0, HALF_HORIZON) + yaw0);
    plan.yaw_vel = angles::normalize_angle(yaw_solver_->work->x(1, HALF_HORIZON));
    plan.yaw_acc = yaw_solver_->work->u(0, HALF_HORIZON);

    plan.pitch = pitch_solver_->work->x(0, HALF_HORIZON);
    plan.pitch_vel = pitch_solver_->work->x(1, HALF_HORIZON);
    plan.pitch_acc = pitch_solver_->work->u(0, HALF_HORIZON);

    // auto shoot_offset_ = 2;
    // plan.fire = std::hypot(
    //                 traj(0, HALF_HORIZON + shoot_offset_)
    //                     - yaw_solver_->work->x(0, HALF_HORIZON + shoot_offset_),
    //                 traj(2, HALF_HORIZON + shoot_offset_)
    //                     - pitch_solver_->work->x(0, HALF_HORIZON + shoot_offset_)
    //             )
    //     < fire_thresh_;

    return plan;
}
void Planner::setup_yaw_solver(const YAML::Node& config) {
    auto max_yaw_acc = config["max_yaw_acc"].as<double>();
    auto Q_yaw = config["Q_yaw"].as<std::vector<double>>();
    auto R_yaw = config["R_yaw"].as<std::vector<double>>();

    Eigen::MatrixXd A { { 1, DT }, { 0, 1 } };
    Eigen::MatrixXd B { { 0 }, { DT } };
    Eigen::VectorXd f { { 0, 0 } };
    Eigen::Matrix<double, 2, 1> Q(Q_yaw.data());
    Eigen::Matrix<double, 1, 1> R(R_yaw.data());
    tiny_setup(&yaw_solver_, A, B, f, Q.asDiagonal(), R.asDiagonal(), 1.0, 2, 1, HORIZON, 0);

    Eigen::MatrixXd x_min = Eigen::MatrixXd::Constant(2, HORIZON, -1e17);
    Eigen::MatrixXd x_max = Eigen::MatrixXd::Constant(2, HORIZON, 1e17);
    Eigen::MatrixXd u_min = Eigen::MatrixXd::Constant(1, HORIZON - 1, -max_yaw_acc);
    Eigen::MatrixXd u_max = Eigen::MatrixXd::Constant(1, HORIZON - 1, max_yaw_acc);
    tiny_set_bound_constraints(yaw_solver_, x_min, x_max, u_min, u_max);

    yaw_solver_->settings->max_iter = 10;
}

void Planner::setup_pitch_solver(const YAML::Node& config) {
    auto max_pitch_acc = config["max_pitch_acc"].as<double>();
    auto Q_pitch = config["Q_pitch"].as<std::vector<double>>();
    auto R_pitch = config["R_pitch"].as<std::vector<double>>();
    Eigen::MatrixXd A { { 1, DT }, { 0, 1 } };
    Eigen::MatrixXd B { { 0 }, { DT } };
    Eigen::VectorXd f { { 0, 0 } };
    Eigen::Matrix<double, 2, 1> Q(Q_pitch.data());
    Eigen::Matrix<double, 1, 1> R(R_pitch.data());
    tiny_setup(&pitch_solver_, A, B, f, Q.asDiagonal(), R.asDiagonal(), 1.0, 2, 1, HORIZON, 0);

    Eigen::MatrixXd x_min = Eigen::MatrixXd::Constant(2, HORIZON, -1e17);
    Eigen::MatrixXd x_max = Eigen::MatrixXd::Constant(2, HORIZON, 1e17);
    Eigen::MatrixXd u_min = Eigen::MatrixXd::Constant(1, HORIZON - 1, -max_pitch_acc);
    Eigen::MatrixXd u_max = Eigen::MatrixXd::Constant(1, HORIZON - 1, max_pitch_acc);
    tiny_set_bound_constraints(pitch_solver_, x_min, x_max, u_min, u_max);

    pitch_solver_->settings->max_iter = 10;
}

Eigen::Matrix<double, 2, 1>
Planner::aim(const Target& target, double bullet_speed, int idx, bool aim_center, bool aim_first) {
    AimTarget aim_target;
    if (!has_changed_) {
        aim_target = aimer_->aimTargetNoPre(target, bullet_speed, aim_first);
        if (aim_target.idx != idx) {
            has_changed_ = true;
            chenged_idx_ = aim_target.idx;
        }
    } else {
        aim_target = aimer_->aimTargetNoPreWithIdx(target, bullet_speed, chenged_idx_);
    }

    GimbalCmd gimbal_cmd;

    gimbal_cmd = shooter_->calYawPitch(aim_target, false);

    return { gimbal_cmd.yaw * M_PI / 180.0, gimbal_cmd.pitch * M_PI / 180.0 };
}
Trajectory Planner::get_trajectory(
    Target& target,
    double yaw0,
    double bullet_speed,
    int idx,
    bool aim_center,
    bool aim_first
) {
    Trajectory traj;

    target.predict(-DT * (HALF_HORIZON + 1));
    idx = aimer_->selectBestArmor(target, aim_first);
    auto yaw_pitch_last = aim(target, bullet_speed, idx, aim_center, aim_first);

    target.predict(DT); // [0] = -HALF_HORIZON * DT -> [HHALF_HORIZON] = 0
    auto yaw_pitch = aim(target, bullet_speed, idx, aim_center, aim_first);

    for (int i = 0; i < HORIZON; i++) {
        target.predict(DT);
        auto yaw_pitch_next = aim(target, bullet_speed, idx, aim_center, aim_first);

        auto yaw_vel = angles::normalize_angle(yaw_pitch_next(0) - yaw_pitch_last(0)) / (2 * DT);
        auto pitch_vel = (yaw_pitch_next(1) - yaw_pitch_last(1)) / (2 * DT);

        traj.col(i) << angles::normalize_angle(yaw_pitch(0) - yaw0), yaw_vel, yaw_pitch(1),
            pitch_vel;

        yaw_pitch_last = yaw_pitch;
        yaw_pitch = yaw_pitch_next;
    }

    return traj;
}