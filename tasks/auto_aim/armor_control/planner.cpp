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

Plan Planner::plan(
    Target target,
    double bullet_speed,
    const AutoAimFsm& auto_aim_fsm,
    double self_v_yaw,
    double cal_dt,
    int cal_horizon
) {
    aim_first_idx_ = -1;
    // 0. Check bullet speed
    if (bullet_speed < 10 || bullet_speed > 25) {
        bullet_speed = 22;
    }
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
    auto now = time_utils::now();
    // auto bullet_traj = tools::Trajectory(bullet_speed, min_dist, xyz.z());
    double dt0 = time_utils::durationSec(target.timestamp_, now);
    double fly_time = aimer_->getFlyingTime(xyz, bullet_speed);
    double dt_total = dt0 + fly_time + aimer_->getPredelay();
    std::chrono::steady_clock::time_point future =
        now + std::chrono::microseconds(int((dt_total)*1e6));

    target.predict(future);
    bool aim_first = false;
    if (auto_aim_fsm == AutoAimFsm::AIM_SINGLE_ARMOR) {
        aim_first = true;
    }
    bool aim_center = false;
    if (auto_aim_fsm == AutoAimFsm::AIM_WHOLE_CAR_CENTER) {
        aim_center = true;
    }
    double yaw0;
    Trajectory traj;

    try {
        yaw0 = aim(target, bullet_speed, aim_center, aim_first, self_v_yaw, dt_total)(0);
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
    } catch (const std::exception& e) {
        return { false };
    }

    // 3. Solve yaw
    Eigen::VectorXd x0(2);
    x0 << traj(0, 0), traj(1, 0);
    tiny_set_x0(yaw_solver_, x0);

    yaw_solver_->work->Xref = traj.block(0, 0, 2, MPC_HORIZON);
    tiny_solve(yaw_solver_);

    // 4. Solve pitch
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

    plan.target_yaw = angles::normalize_angle(traj(0, MPC_HALF_HORIZON) + yaw0);
    plan.target_pitch = traj(2, MPC_HALF_HORIZON);

    plan.yaw = angles::normalize_angle(yaw_solver_->work->x(0, MPC_HALF_HORIZON) + yaw0);
    plan.yaw_vel = angles::normalize_angle(yaw_solver_->work->x(1, MPC_HALF_HORIZON));
    plan.yaw_acc = yaw_solver_->work->u(0, MPC_HALF_HORIZON);

    plan.pitch = pitch_solver_->work->x(0, MPC_HALF_HORIZON);
    plan.pitch_vel = pitch_solver_->work->x(1, MPC_HALF_HORIZON);
    plan.pitch_acc = pitch_solver_->work->u(0, MPC_HALF_HORIZON);

    auto shoot_offset_ = 2;
    plan.fire = std::hypot(
                    traj(0, MPC_HALF_HORIZON + shoot_offset_)
                        - yaw_solver_->work->x(0, MPC_HALF_HORIZON + shoot_offset_),
                    traj(2, MPC_HALF_HORIZON + shoot_offset_)
                        - pitch_solver_->work->x(0, MPC_HALF_HORIZON + shoot_offset_)
                )
        < fire_thresh_;

    return plan;
}
void Planner::setup_yaw_solver(const YAML::Node& config) {
    auto max_yaw_acc = config["max_yaw_acc"].as<double>();
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
    Eigen::MatrixXd u_min = Eigen::MatrixXd::Constant(1, MPC_HORIZON - 1, -max_yaw_acc);
    Eigen::MatrixXd u_max = Eigen::MatrixXd::Constant(1, MPC_HORIZON - 1, max_yaw_acc);
    tiny_set_bound_constraints(yaw_solver_, x_min, x_max, u_min, u_max);

    yaw_solver_->settings->max_iter = 10;
}

void Planner::setup_pitch_solver(const YAML::Node& config) {
    auto max_pitch_acc = config["max_pitch_acc"].as<double>();
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
    Eigen::MatrixXd u_min = Eigen::MatrixXd::Constant(1, MPC_HORIZON - 1, -max_pitch_acc);
    Eigen::MatrixXd u_max = Eigen::MatrixXd::Constant(1, MPC_HORIZON - 1, max_pitch_acc);
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
    AimTarget aim_target;
    Eigen::Vector3d xyz;
    double yaw;
    auto min_dist = 1e10;
    for (auto& xyza: target.getArmorPosAndYaw()) {
        auto dist = xyza.head<2>().norm();
        if (dist < min_dist) {
            min_dist = dist;
            xyz = xyza.head<3>();
            yaw = xyza[3];
        }
    }
    std::vector<Eigen::Vector4d> armor_xyza_list = target.getArmorPosAndYaw();
    auto armor_num = armor_xyza_list.size();
    double center_yaw = std::atan2(target.position().y(), target.position().x());
    std::vector<double> delta_angle_list;
    for (int i = 0; i < armor_num; i++) {
        auto delta_angle = angles::normalize_angle(armor_xyza_list[i][3] - center_yaw);
        delta_angle_list.emplace_back(delta_angle);
    }
    if (aim_first && target.tracked_id_ != armor::ArmorNumber::OUTPOST) {
        std::vector<int> id_list;
        for (int i = 0; i < armor_num; i++) {
            if (std::abs(delta_angle_list[i]) > 60 / 57.3)
                continue;
            id_list.push_back(i);
        }

        if (!id_list.empty()) {
            if (id_list.size() > 1) {
                int id0 = id_list[0], id1 = id_list[1];

                if (lock_id != id0 && lock_id != id1)
                    lock_id = (std::abs(delta_angle_list[id0]) < std::abs(delta_angle_list[id1]))
                        ? id0
                        : id1;

                xyz = armor_xyza_list[lock_id].head<3>();

            } else {
                lock_id = -1;
                xyz = armor_xyza_list[id_list[0]].head<3>();
            }
        }
    } else {
        double coming_angle, leaving_angle;
        if (target.tracked_id_ == armor::ArmorNumber::OUTPOST) {
            coming_angle = 70 / 57.3;
            leaving_angle = 30 / 57.3;
        } else {
            auto commingleaving = aimer_->getCommingLeaving();
            coming_angle = commingleaving.first / 57.3;
            leaving_angle = commingleaving.second / 57.3;
        }

        // 在小陀螺时，一侧的装甲板不断出现，另一侧的装甲板不断消失，显然前者被打中的概率更高
        for (int i = 0; i < armor_num; i++) {
            if (std::abs(delta_angle_list[i]) > coming_angle)
                continue;
            if (target.v_yaw() > 0 && delta_angle_list[i] < leaving_angle) {
                xyz = armor_xyza_list[i].head<3>();
                break;
            }

            if (target.v_yaw() < 0 && delta_angle_list[i] > -leaving_angle) {
                xyz = armor_xyza_list[i].head<3>();
                break;
            }
        }
    }

    aim_target.pos = xyz;
    double raw_pitch = aim_target.calRawPitch();
    aimer_->compensate(aim_target.pos, raw_pitch, bullet_speed);
    aim_target.shoot_pitch = raw_pitch;
    aim_target.valid = true;
    aim_target.host_pos = target.position();
    aim_target.host_vel = target.velocity();
    GimbalCmd gimbal_cmd;
    double control_yaw = 0.0, control_pitch = 0.0;
    double v_yaw = 0.0, v_pitch = 0.0;
    GimbalCmd cmd;

    v_pitch = aim_target.calVPitch();
    cmd.timestamp = std::chrono::steady_clock::now();
    cmd.distance = aim_target.distance();
    if (aim_center) {
        control_yaw =
            std::atan2(aim_target.host_pos.y(), aim_target.host_pos.x()) - self_v_yaw * dt;
        v_yaw = aim_target.calHostVYaw();
    } else {
        control_yaw = aim_target.calYaw() - self_v_yaw * dt;
        v_yaw = aim_target.calVYaw();
    }
    control_pitch = aim_target.shoot_pitch;
    cmd.yaw = control_yaw / M_PI * 180.0;
    cmd.pitch = control_pitch / M_PI * 180.0;
    cmd.v_pitch = v_pitch / M_PI * 180.0;
    cmd.v_yaw = v_yaw / M_PI * 180.0;
    cmd.aim_target = aim_target;
    cmd.appera = true;
    gimbal_cmd = shooter_->shoot(cmd, bullet_speed);

    return { gimbal_cmd.yaw * M_PI / 180.0, gimbal_cmd.pitch * M_PI / 180.0 };
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
    // 生成 CAL 参考轨迹
    Eigen::Matrix<double, 4, Eigen::Dynamic> cal_traj(4, cal_horizon);

    target.predict(-cal_dt * (cal_horizon / 2.0 + 1));
    auto yaw_pitch_last = aim(target, bullet_speed, aim_center, aim_first, self_v_yaw, dt);

    target.predict(cal_dt);
    auto yaw_pitch = aim(target, bullet_speed, aim_center, aim_first, self_v_yaw, dt);

    for (int i = 0; i < cal_horizon; i++) {
        target.predict(cal_dt);
        auto yaw_pitch_next = aim(target, bullet_speed, aim_center, aim_first, self_v_yaw, dt);

        // 角度速度计算，处理 wrap-around
        double yaw_diff = yaw_pitch_next(0) - yaw_pitch_last(0);
        if (yaw_diff > M_PI)
            yaw_diff -= 2 * M_PI;
        if (yaw_diff < -M_PI)
            yaw_diff += 2 * M_PI;

        double yaw_vel = yaw_diff / (2 * cal_dt);
        double pitch_vel = (yaw_pitch_next(1) - yaw_pitch_last(1)) / (2 * cal_dt);

        cal_traj.col(i) << yaw_pitch(0) - yaw0, yaw_vel, yaw_pitch(1), pitch_vel;

        yaw_pitch_last = yaw_pitch;
        yaw_pitch = yaw_pitch_next;
    }

    // 插值到 MPC_HORIZON
    Trajectory mpc_traj;
    for (int i = 0; i < MPC_HORIZON; i++) {
        double t = i * MPC_DT;
        double idx_f = t / cal_dt;
        int idx0 = std::min(static_cast<int>(idx_f), cal_horizon - 1);
        int idx1 = std::min(idx0 + 1, cal_horizon - 1);
        double alpha = idx_f - idx0;

        // yaw 插值，处理 wrap-around
        double yaw0_cal = cal_traj(0, idx0);
        double yaw1_cal = cal_traj(0, idx1);
        double yaw_diff = yaw1_cal - yaw0_cal;
        if (yaw_diff > M_PI)
            yaw_diff -= 2 * M_PI;
        if (yaw_diff < -M_PI)
            yaw_diff += 2 * M_PI;
        double yaw_interp = yaw0_cal + alpha * yaw_diff;

        // pitch 和速度线性插值
        double pitch_interp = cal_traj(2, idx0) + alpha * (cal_traj(2, idx1) - cal_traj(2, idx0));
        double yaw_vel_interp = cal_traj(1, idx0) + alpha * (cal_traj(1, idx1) - cal_traj(1, idx0));
        double pitch_vel_interp =
            cal_traj(3, idx0) + alpha * (cal_traj(3, idx1) - cal_traj(3, idx0));

        mpc_traj.col(i) << yaw_interp, yaw_vel_interp, pitch_interp, pitch_vel_interp;
    }

    return mpc_traj;
}
