#include "planner.hpp"
#include <wust_vl/common/utils/timer.hpp>

inline double rad2deg(double r) {
    return r / M_PI * 180.0;
}
inline double deg2rad(double d) {
    return d * M_PI / 180.0;
}
Planner::Planner(
    const YAML::Node& config,
    std::shared_ptr<Aimer> aimer,
    std::shared_ptr<Shooter> shooter
) {
    aimer_ = aimer;
    shooter_ = shooter;
    max_iter_ = config["max_iter"].as<int>();
    control_delay_ = config["control_delay"].as<double>();
    enable_fire_error_ = config["enable_fire_error"].as<double>();
    setup_yaw_solver(config);
    setup_pitch_solver(config);
}
Eigen::Vector4d getStateAtTime(const Trajectory& traj, double t) {
    const double total_time = (traj.cols() - 1) * MPC_DT;

    // 限制 t 在有效时间范围
    if (t <= 0.0) {
        return traj.col(0);
    }
    if (t >= total_time) {
        return traj.col(traj.cols() - 1);
    }

    // 根据时间计算轨迹索引
    double index = t / MPC_DT;

    int i0 = floor(index);
    int i1 = ceil(index);

    if (i0 == i1) {
        return traj.col(i0);
    }

    double alpha = index - i0;

    // 线性插值
    return (1.0 - alpha) * traj.col(i0) + alpha * traj.col(i1);
}

Plan Planner::plan(
    Target target,
    double bullet_speed,
    const AutoAimFsm& auto_aim_fsm,
    double self_v_yaw
) {
    bool aim_first = (auto_aim_fsm == AutoAimFsm::AIM_SINGLE_ARMOR);
    bool aim_center = (auto_aim_fsm == AutoAimFsm::AIM_WHOLE_CAR_CENTER);

    // pick nearest armor for initial estimate
    auto [xyz, no_use] = selectArmor(target, aim_first);

    // compute time offset and future prediction time
    auto now = time_utils::now();
    double dt0 = time_utils::durationSec(target.timestamp_, now);
    double fly_time = aimer_->getFlyingTime(xyz, bullet_speed);
    double dt_total = dt0 + fly_time + aimer_->getPredelay();
    auto future = now + std::chrono::microseconds(int(dt_total * 1e6));
    target.predict(future);

    Trajectory traj;
    std::vector<Eigen::Vector4d> armor_list;
    AimTarget aim_target;
    double yaw0 = 0.0;

    try {
        auto result = calaim(target, bullet_speed, aim_center, aim_first, self_v_yaw, dt_total);
        double tmp_yaw = std::get<0>(result);
        double tmp_pitch = std::get<1>(result);
        armor_list = std::get<2>(result);
        aim_target = std::get<3>(result);
        yaw0 = angles::normalize_angle(tmp_yaw);

        traj =
            get_trajectory(target, yaw0, bullet_speed, aim_center, aim_first, self_v_yaw, dt_total);
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
    // plan.target_yaw = angles::normalize_angle(traj(0, MPC_HALF_HORIZON) + yaw0);
    // plan.target_yaw = angles::normalize_angle(traj(0, MPC_HALF_HORIZON+2) + yaw0);

    // plan.target_pitch = traj(2, MPC_HALF_HORIZON);
    const double total_time = MPC_HORIZON * MPC_DT;
    auto target_state = getStateAtTime(traj, total_time / 2.0);
    plan.target_yaw = angles::normalize_angle(target_state(0) + yaw0);
    plan.target_pitch = angles::normalize_angle(target_state(2));
    plan.yaw = angles::normalize_angle(yaw_solver_->work->x(0, MPC_HALF_HORIZON) + yaw0);
    plan.yaw_vel = yaw_solver_->work->x(1, MPC_HALF_HORIZON);
    plan.yaw_acc = yaw_solver_->work->u(0, MPC_HALF_HORIZON);

    plan.pitch = pitch_solver_->work->x(0, MPC_HALF_HORIZON);
    plan.pitch_vel = pitch_solver_->work->x(1, MPC_HALF_HORIZON);
    plan.pitch_acc = pitch_solver_->work->u(0, MPC_HALF_HORIZON);

    plan.armor_posandyaw = armor_list;
    plan.aim_target = aim_target;
    target_state = getStateAtTime(traj, total_time / 2.0 + control_delay_);
    Trajectory control_traj;
    control_traj.block(0, 0, 2, MPC_HORIZON) = yaw_solver_->work->x;
    control_traj.block(2, 0, 2, MPC_HORIZON) = pitch_solver_->work->x;
    auto control_state = getStateAtTime(control_traj, total_time / 2.0 + control_delay_);
    plan.fire =
        std::hypot(
            angles::normalize_angle(target_state(0) + yaw0)
                - angles::normalize_angle(control_state(0) + yaw0),
            angles::normalize_angle(target_state(2)) - angles::normalize_angle(control_state(2))
        )
        < enable_fire_error_;

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

    yaw_solver_->settings->max_iter = max_iter_;
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

    pitch_solver_->settings->max_iter = max_iter_;
}
std::pair<Eigen::Vector3d, double> Planner::selectArmor(const Target& target, bool aim_first) {
    static int lock_id = -1;

    auto armor_list = target.getArmorPosAndYaw();
    int armor_num = static_cast<int>(armor_list.size());

    Eigen::Vector3d chosen = Eigen::Vector3d::Zero();
    double chosen_yaw = 0.0;

    if (armor_num > 0) {
        chosen = armor_list[0].head<3>();
        chosen_yaw = armor_list[0][3];
    }

    double center_yaw = std::atan2(target.position().y(), target.position().x());

    std::vector<double> delta_angles;
    delta_angles.reserve(armor_num);
    for (int i = 0; i < armor_num; ++i) {
        delta_angles.push_back(angles::normalize_angle(armor_list[i][3] - center_yaw));
    }

    auto pick_best_by_min_abs = [&](const std::vector<int>& idxs) -> int {
        int best = -1;
        double best_val = std::numeric_limits<double>::infinity();
        for (int i: idxs) {
            double val = std::abs(delta_angles[i]);
            if (val < best_val) {
                best_val = val;
                best = i;
            }
        }
        return best;
    };

    // -------------------- AIM_FIRST 逻辑 --------------------
    if (aim_first && target.tracked_id_ != armor::ArmorNumber::OUTPOST && armor_num > 0) {
        std::vector<int> candidates;
        for (int i = 0; i < armor_num; ++i)
            if (std::abs(delta_angles[i]) <= 60.0 / 57.3)
                candidates.push_back(i);

        if (!candidates.empty()) {
            if (candidates.size() > 1) {
                int a = candidates[0], b = candidates[1];
                if (lock_id != a && lock_id != b) {
                    lock_id = (std::abs(delta_angles[a]) < std::abs(delta_angles[b])) ? a : b;
                }

                int pick = (lock_id >= 0 && lock_id < armor_num) ? lock_id
                                                                 : pick_best_by_min_abs(candidates);

                if (pick >= 0) {
                    chosen = armor_list[pick].head<3>();
                    chosen_yaw = armor_list[pick][3];
                }
            } else {
                lock_id = -1;
                int pick = candidates[0];
                chosen = armor_list[pick].head<3>();
                chosen_yaw = armor_list[pick][3];
            }
        }

        return { chosen, chosen_yaw };
    }

    // -------------------- 普通非 AIM_FIRST 逻辑 --------------------
    if (armor_num > 0) {
        int best_idx = -1;

        if (target.tracked_id_ != armor::ArmorNumber::OUTPOST) {
            auto [com, lea] = aimer_->getCommingLeaving();
            double coming_angle = com * M_PI / 180.0;
            double leaving_angle = lea * M_PI / 180.0;

            for (int i = 0; i < armor_num; ++i) {
                if (std::abs(delta_angles[i]) > coming_angle)
                    continue;

                if (target.v_yaw() > 0 && delta_angles[i] < leaving_angle)
                    best_idx = i;
                if (target.v_yaw() < 0 && delta_angles[i] > -leaving_angle)
                    best_idx = i;
            }
        }

        if (best_idx < 0) {
            std::vector<int> all(armor_num);
            std::iota(all.begin(), all.end(), 0);
            best_idx = pick_best_by_min_abs(all);
        }

        chosen = armor_list[best_idx].head<3>();
        chosen_yaw = armor_list[best_idx][3];
    }

    return { chosen, chosen_yaw };
}

std::tuple<double, double, std::vector<Eigen::Vector4d>, AimTarget> Planner::calaim(
    const Target& target,
    double bullet_speed,
    bool aim_center,
    bool aim_first,
    double self_v_yaw,
    double dt
) {
    auto [chosen, chosen_yaw] = selectArmor(target, aim_first);
    std::vector<Target> iteration_target(10, target);
    bool converged = false;
    double prev_fly_time = 0.0;
    for (int iter = 0; iter < 100; ++iter) {
        auto predict_time = prev_fly_time;
        iteration_target[iter].predict(predict_time);
        auto [iter_chosen, iter_yaw] = selectArmor(iteration_target[iter], aim_first);
        double iter_fly_time = aimer_->getFlyingTime(iter_chosen, bullet_speed);
        if (std::abs(iter_fly_time - prev_fly_time) < 0.001) {
            converged = true;
            break;
        }
        prev_fly_time = iter_fly_time;
    }
    auto fin_target = target;
    auto predict_time = prev_fly_time;
    fin_target.predict(predict_time);
    auto [fin_chosen, fin_chosen_yaw] = selectArmor(fin_target, aim_first);
    // build AimTarget
    AimTarget at;
    at.pos = fin_chosen;
    Eigen::Vector4d chosen_and_yaw =
        Eigen::Vector4d(fin_chosen.x(), fin_chosen.y(), fin_chosen.z(), fin_chosen_yaw);
    double raw_pitch = at.calRawPitch();
    aimer_->compensate(at.pos, raw_pitch, bullet_speed);
    at.shoot_pitch = raw_pitch;
    at.valid = true;
    at.host_pos = target.position();
    at.host_vel = target.velocity();

    // build orientation quaternion (ZYX order)
    Eigen::Vector3d euler;
    euler.x() = M_PI / 2.0;
    euler.y() = (target.tracked_id_ == armor::ArmorNumber::OUTPOST) ? -0.2618 : 0.2618;
    euler.z() = fin_chosen_yaw;
    Eigen::Quaterniond ori = utils::eulerToQuat(euler, utils::EulerOrder::ZYX);
    at.ori = ori;

    // compute control yaw/pitch and rates
    GimbalCmd cmd;
    double control_yaw = at.calYaw() - self_v_yaw * dt;
    double v_yaw = at.calVYaw();

    cmd.timestamp = std::chrono::steady_clock::now();
    cmd.distance = at.distance();
    cmd.yaw = rad2deg(control_yaw);
    cmd.pitch = rad2deg(at.shoot_pitch);
    cmd.v_pitch = rad2deg(at.calVPitch());
    cmd.v_yaw = rad2deg(v_yaw);
    cmd.aim_target = at;
    cmd.appera = true;
    cmd.armor_posandyaw = { chosen_and_yaw };

    GimbalCmd gimbal_cmd = shooter_->shoot(cmd, bullet_speed);

    return { angles::normalize_angle(deg2rad(gimbal_cmd.yaw)),
             angles::normalize_angle(deg2rad(gimbal_cmd.pitch)),
             { chosen_and_yaw },
             at };
}

Eigen::Matrix<double, 2, 1> Planner::aim(
    const Target& target,
    double bullet_speed,
    bool aim_center,
    bool aim_first,
    double self_v_yaw,
    double dt
) {
    auto [yaw, pitch, armor_list, at] =
        calaim(target, bullet_speed, aim_center, aim_first, self_v_yaw, dt);
    return Eigen::Matrix<double, 2, 1>(yaw, pitch);
}

Trajectory Planner::get_trajectory(
    Target& target,
    double yaw0,
    double bullet_speed,
    bool aim_center,
    bool aim_first,
    double self_v_yaw,
    double dt
) {
    double yaw_rel_arr[CAL_HORIZON];
    double yaw_vel_arr[CAL_HORIZON];
    double pitch_arr[CAL_HORIZON];
    double pitch_vel_arr[CAL_HORIZON];

    auto unwrap_continuous = [&](double prev, double cur) {
        return prev + angles::shortest_angular_distance(prev, cur);
    };

    target.predict(-CAL_DT * (CAL_HORIZON / 2.0 + 1));
    auto last = aim(target, bullet_speed, aim_center, aim_first, self_v_yaw, dt);
    target.predict(CAL_DT);
    auto cur = aim(target, bullet_speed, aim_center, aim_first, self_v_yaw, dt);

    double yaw_cont_last = last(0);
    double yaw_cont = unwrap_continuous(yaw_cont_last, cur(0));

    for (int i = 0; i < CAL_HORIZON; ++i) {
        target.predict(CAL_DT);
        auto next = aim(target, bullet_speed, aim_center, aim_first, self_v_yaw, dt);

        double yaw_cont_next = unwrap_continuous(yaw_cont, next(0));

        double yaw_diff = yaw_cont_next - yaw_cont_last;
        double yaw_vel = yaw_diff * (0.5 / CAL_DT);
        double pitch_vel = (next(1) - last(1)) * (0.5 / CAL_DT);
        double yaw_rel = angles::shortest_angular_distance(yaw0, yaw_cont);
        yaw_rel_arr[i] = yaw_rel;
        yaw_vel_arr[i] = yaw_vel;
        pitch_arr[i] = cur(1);
        pitch_vel_arr[i] = pitch_vel;

        last = cur;
        cur = next;

        yaw_cont_last = yaw_cont;
        yaw_cont = yaw_cont_next;
    }

    Trajectory mpc_traj;

    for (int i = 0; i < MPC_HORIZON; ++i) {
        double t = i * MPC_DT;
        double idx_f = t / CAL_DT;

        int idx0 = std::min(static_cast<int>(idx_f), CAL_HORIZON - 1);
        int idx1 = std::min(idx0 + 1, CAL_HORIZON - 1);
        double alpha = idx_f - idx0;

        double y0 = yaw_rel_arr[idx0];
        double y1 = yaw_rel_arr[idx1];
        double ydiff = angles::shortest_angular_distance(y0, y1);
        double yaw_interp = y0 + alpha * ydiff;

        double yaw_vel_interp = yaw_vel_arr[idx0] + alpha * (yaw_vel_arr[idx1] - yaw_vel_arr[idx0]);
        double pitch_interp = pitch_arr[idx0] + alpha * (pitch_arr[idx1] - pitch_arr[idx0]);
        double pitch_vel_interp =
            pitch_vel_arr[idx0] + alpha * (pitch_vel_arr[idx1] - pitch_vel_arr[idx0]);

        mpc_traj.col(i) << yaw_interp, yaw_vel_interp, pitch_interp, pitch_vel_interp;
    }

    return mpc_traj;
}
