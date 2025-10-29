#include "planner.hpp"
#include <wust_vl/common/utils/timer.hpp>
inline double rad2deg(double r) { return r / M_PI * 180.0; }
inline double deg2rad(double d) { return d * M_PI / 180.0; }
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

Plan Planner::plan(
    Target target,
    double bullet_speed,
    const AutoAimFsm& auto_aim_fsm,
    double self_v_yaw,
    double cal_dt,
    int cal_horizon
) {
    aim_first_idx_ = -1;
    if (bullet_speed < 10 || bullet_speed > 25)
        bullet_speed = 22;

    // pick nearest armor for initial estimate
    Eigen::Vector3d xyz;
    auto min_dist = 1e10;
    auto xyzas = target.getArmorPosAndYaw();
    for (int i = 0; i < xyzas.size(); i++) {
        auto dist = xyzas[i].head<2>().norm();
        if (dist < min_dist) {
            min_dist = dist;
            xyz = xyzas[i].head<3>();
            aim_first_idx_ = i;
        }
    }

    // compute time offset and future prediction time
    auto now = time_utils::now();
    double dt0 = time_utils::durationSec(target.timestamp_, now);
    double fly_time = aimer_->getFlyingTime(xyz, bullet_speed);
    double dt_total = dt0 + fly_time + aimer_->getPredelay();
    auto future = now + std::chrono::microseconds(int(dt_total * 1e6));
    target.predict(future);

    bool aim_first = (auto_aim_fsm == AutoAimFsm::AIM_SINGLE_ARMOR);
    bool aim_center = (auto_aim_fsm == AutoAimFsm::AIM_WHOLE_CAR_CENTER);

    double yaw0;
    Trajectory traj;
    try {
        // aim() returns normalized radians [−π, π] for yaw & pitch
        yaw0 = angles::normalize_angle(
            aim(target, bullet_speed, aim_center, aim_first, self_v_yaw, dt_total)(0)
        );
        traj = get_trajectory(
            target,
            yaw0,
            bullet_speed,
            aim_center,
            aim_first,
            self_v_yaw,
            dt_total,
            cal_dt,
            cal_horizon
        );
    } catch (const std::exception&) {
        return { false };
    }

    // Solve yaw MPC
    Eigen::VectorXd x0(2);
    x0 << traj(0, 0), traj(1, 0);
    tiny_set_x0(yaw_solver_, x0);
    yaw_solver_->work->Xref = traj.block(0, 0, 2, MPC_HORIZON);
    tiny_solve(yaw_solver_);

    // Solve pitch MPC
    x0 << traj(2, 0), traj(3, 0);
    tiny_set_x0(pitch_solver_, x0);
    pitch_solver_->work->Xref = traj.block(2, 0, 2, MPC_HORIZON);
    tiny_solve(pitch_solver_);

    if (!yaw_solver_->work->status || !pitch_solver_->work->status) {
        WUST_WARN("planner") << "mpc solver error!";
        return { false };
    }

    Plan plan;
    plan.control = true;

    // convert relative yaw (traj stores yaw relative to yaw0) -> absolute & normalize
    plan.target_yaw = angles::normalize_angle(traj(0, MPC_HALF_HORIZON) + yaw0);
    plan.target_pitch = traj(2, MPC_HALF_HORIZON);

    plan.yaw = angles::normalize_angle(yaw_solver_->work->x(0, MPC_HALF_HORIZON) + yaw0);
    plan.yaw_vel = yaw_solver_->work->x(1, MPC_HALF_HORIZON);
    plan.yaw_acc = yaw_solver_->work->u(0, MPC_HALF_HORIZON);

    plan.pitch = pitch_solver_->work->x(0, MPC_HALF_HORIZON);
    plan.pitch_vel = pitch_solver_->work->x(1, MPC_HALF_HORIZON);
    plan.pitch_acc = pitch_solver_->work->u(0, MPC_HALF_HORIZON);

    // fire 判定：短角差避免 ±π 跳变
    int idx = MPC_HALF_HORIZON + 2;
    if (idx >= MPC_HORIZON)
        idx = MPC_HORIZON - 1;

    double yaw_err = angles::shortest_angular_distance(traj(0, idx), yaw_solver_->work->x(0, idx));
    double pitch_err = traj(2, idx) - pitch_solver_->work->x(0, idx);
    plan.fire = std::hypot(yaw_err, pitch_err) < fire_thresh_;

    return plan;
}


void Planner::setup_yaw_solver(const YAML::Node& config) {
    max_yaw_acc_ = config["max_yaw_acc"].as<double>();
    auto Q_yaw = config["Q_yaw"].as<std::vector<double>>();
    auto R_yaw = config["R_yaw"].as<std::vector<double>>();

    Eigen::MatrixXd A { { 1, MPC_DT }, { 0, 1 } };
    Eigen::MatrixXd B { { 0 }, { MPC_DT } };
    Eigen::VectorXd f { { 0, 0 } };
    Eigen::Matrix<double, 2, 1> Q(Q_yaw.data());
    Eigen::Matrix<double, 1, 1> R(R_yaw.data());
    tiny_setup(&yaw_solver_, A, B, f, Q.asDiagonal(), R.asDiagonal(), 1.0, 2, 1, MPC_HORIZON, 0);

    Eigen::MatrixXd x_min = Eigen::MatrixXd::Constant(2, MPC_HORIZON, -1e17);
    Eigen::MatrixXd x_max = Eigen::MatrixXd::Constant(2, MPC_HORIZON, 1e17);
    Eigen::MatrixXd u_min = Eigen::MatrixXd::Constant(1, MPC_HORIZON - 1, -max_yaw_acc_);
    Eigen::MatrixXd u_max = Eigen::MatrixXd::Constant(1, MPC_HORIZON - 1, max_yaw_acc_);
    tiny_set_bound_constraints(yaw_solver_, x_min, x_max, u_min, u_max);

    yaw_solver_->settings->max_iter = 10;
}

void Planner::setup_pitch_solver(const YAML::Node& config) {
    max_pitch_acc_ = config["max_pitch_acc"].as<double>();
    auto Q_pitch = config["Q_pitch"].as<std::vector<double>>();
    auto R_pitch = config["R_pitch"].as<std::vector<double>>();
    Eigen::MatrixXd A { { 1, MPC_DT }, { 0, 1 } };
    Eigen::MatrixXd B { { 0 }, { MPC_DT } };
    Eigen::VectorXd f { { 0, 0 } };
    Eigen::Matrix<double, 2, 1> Q(Q_pitch.data());
    Eigen::Matrix<double, 1, 1> R(R_pitch.data());
    tiny_setup(&pitch_solver_, A, B, f, Q.asDiagonal(), R.asDiagonal(), 1.0, 2, 1, MPC_HORIZON, 0);

    Eigen::MatrixXd x_min = Eigen::MatrixXd::Constant(2, MPC_HORIZON, -1e17);
    Eigen::MatrixXd x_max = Eigen::MatrixXd::Constant(2, MPC_HORIZON, 1e17);
    Eigen::MatrixXd u_min = Eigen::MatrixXd::Constant(1, MPC_HORIZON - 1, -max_pitch_acc_);
    Eigen::MatrixXd u_max = Eigen::MatrixXd::Constant(1, MPC_HORIZON - 1, max_pitch_acc_);
    tiny_set_bound_constraints(pitch_solver_, x_min, x_max, u_min, u_max);

    pitch_solver_->settings->max_iter = 10;
}

Eigen::Matrix<double, 2, 1> Planner::aim(
    const Target& target,
    double bullet_speed,
    bool aim_center,
    bool aim_first,
    double self_v_yaw,
    double dt
) {
    static int lock_id = -1;

    // pick nearest armor + prepare lists
    auto armor_list = target.getArmorPosAndYaw();
    int armor_num = (int)armor_list.size();
    Eigen::Vector3d chosen;
    double chosen_yaw = 0.0;
    if (armor_list.empty()) {
        chosen = Eigen::Vector3d::Zero();
        chosen_yaw = 0.0;
    } else {
        chosen = armor_list[0].head<3>();
        chosen_yaw = armor_list[0][3];
    }

    // compute center yaw & delta angles
    double center_yaw = std::atan2(target.position().y(), target.position().x());
    std::vector<double> delta_angles;
    delta_angles.reserve(armor_num);
    for (int i = 0; i < armor_num; ++i)
        delta_angles.push_back(angles::normalize_angle(armor_list[i][3] - center_yaw));

    // handle AIM_SINGLE_ARMOR / candidate selection
    if (aim_first && target.tracked_id_ != armor::ArmorNumber::OUTPOST) {
        std::vector<int> candidates;
        for (int i = 0; i < armor_num; ++i)
            if (std::abs(delta_angles[i]) <= 60.0 / 57.3) candidates.push_back(i);

        if (!candidates.empty()) {
            if (candidates.size() > 1) {
                int a = candidates[0], b = candidates[1];
                if (lock_id != a && lock_id != b)
                    lock_id = (std::abs(delta_angles[a]) < std::abs(delta_angles[b])) ? a : b;
                chosen = armor_list[lock_id].head<3>();
            } else {
                lock_id = -1;
                chosen = armor_list[candidates[0]].head<3>();
            }
        }
    } else {
        double coming_angle = 70.0/57.3, leaving_angle = 30.0/57.3;
        if (target.tracked_id_ != armor::ArmorNumber::OUTPOST) {
            auto cl = aimer_->getCommingLeaving();
            coming_angle = cl.first / 57.3;
            leaving_angle = cl.second / 57.3;
        }
        for (int i = 0; i < armor_num; ++i) {
            if (std::abs(delta_angles[i]) > coming_angle) continue;
            if (target.v_yaw() > 0 && delta_angles[i] < leaving_angle) { chosen = armor_list[i].head<3>(); break; }
            if (target.v_yaw() < 0 && delta_angles[i] > -leaving_angle) { chosen = armor_list[i].head<3>(); break; }
        }
    }

    AimTarget at;
    at.pos = chosen;
    double raw_pitch = at.calRawPitch();
    aimer_->compensate(at.pos, raw_pitch, bullet_speed);
    at.shoot_pitch = raw_pitch;
    at.valid = true;
    at.host_pos = target.position();
    at.host_vel = target.velocity();

    GimbalCmd cmd;
    double control_yaw = 0.0;
    double v_yaw = 0.0;

    if (aim_center) {
        control_yaw = std::atan2(at.host_pos.y(), at.host_pos.x()) - self_v_yaw * dt;
        v_yaw = at.calHostVYaw();
    } else {
        control_yaw = at.calYaw() - self_v_yaw * dt;
        v_yaw = at.calVYaw();
    }

    cmd.timestamp = std::chrono::steady_clock::now();
    cmd.distance = at.distance();
    cmd.yaw = rad2deg(control_yaw);
    cmd.pitch = rad2deg(at.shoot_pitch);
    cmd.v_pitch = rad2deg(at.calVPitch());
    cmd.v_yaw = rad2deg(v_yaw);
    cmd.aim_target = at;
    cmd.appera = true;

    GimbalCmd gimbal_cmd = shooter_->shoot(cmd, bullet_speed);
    return { angles::normalize_angle(deg2rad(gimbal_cmd.yaw)),
             angles::normalize_angle(deg2rad(gimbal_cmd.pitch)) };
}

Trajectory Planner::get_trajectory(
    Target& target,
    double yaw0,
    double bullet_speed,
    bool aim_center,
    bool aim_first,
    double self_v_yaw,
    double dt,
    double cal_dt,
    int cal_horizon
) {
    Eigen::Matrix<double, 4, Eigen::Dynamic> cal_traj(4, cal_horizon);

    auto unwrap_continuous = [&](double prev, double cur) {
        return prev + angles::shortest_angular_distance(prev, cur);
    };

    // prime predictions (center the window)
    target.predict(-cal_dt * (cal_horizon / 2.0 + 1));
    auto last = aim(target, bullet_speed, aim_center, aim_first, self_v_yaw, dt);
    target.predict(cal_dt);
    auto cur = aim(target, bullet_speed, aim_center, aim_first, self_v_yaw, dt);

    double yaw_cont_last = last(0);
    double yaw_cont = unwrap_continuous(yaw_cont_last, cur(0));

    for (int i = 0; i < cal_horizon; ++i) {
        target.predict(cal_dt);
        auto next = aim(target, bullet_speed, aim_center, aim_first, self_v_yaw, dt);
        double yaw_cont_next = unwrap_continuous(yaw_cont, next(0));

        double yaw_diff = yaw_cont_next - yaw_cont_last;
        double yaw_vel = yaw_diff / (2.0 * cal_dt);
        double pitch_vel = (next(1) - last(1)) / (2.0 * cal_dt);

        double yaw_rel = angles::shortest_angular_distance(yaw0, yaw_cont);

        cal_traj.col(i) << yaw_rel, yaw_vel, cur(1), pitch_vel;

        last = cur;
        cur = next;
        yaw_cont_last = yaw_cont;
        yaw_cont = yaw_cont_next;
    }

    // 插值至 MPC_HORIZON
    Trajectory mpc_traj;
    for (int i = 0; i < MPC_HORIZON; ++i) {
        double t = i * MPC_DT;
        double idx_f = t / cal_dt;
        int idx0 = std::min(static_cast<int>(idx_f), cal_horizon - 1);
        int idx1 = std::min(idx0 + 1, cal_horizon - 1);
        double alpha = idx_f - idx0;

        double y0 = cal_traj(0, idx0);
        double y1 = cal_traj(0, idx1);
        double ydiff = angles::shortest_angular_distance(y0, y1);
        double yaw_interp = y0 + alpha * ydiff;

        double pitch_interp = cal_traj(2, idx0) + alpha * (cal_traj(2, idx1) - cal_traj(2, idx0));
        double yaw_vel_interp = cal_traj(1, idx0) + alpha * (cal_traj(1, idx1) - cal_traj(1, idx0));
        double pitch_vel_interp = cal_traj(3, idx0) + alpha * (cal_traj(3, idx1) - cal_traj(3, idx0));

        mpc_traj.col(i) << yaw_interp, yaw_vel_interp, pitch_interp, pitch_vel_interp;
    }
    return mpc_traj;
}