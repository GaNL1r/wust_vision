#include "very_aimer.hpp"
#include "gcopter/minco.hpp"
#include "tasks/auto_aim/armor_control/tinympc/tiny_api.hpp"
#include "tasks/auto_aim/armor_control/tinympc/types.hpp"
#include "wust_vl/common/utils/manual_compensator.hpp"
namespace auto_aim {
struct ControlPoint {
    double yaw;
    double pitch;
    int id_in_target;
};
constexpr int MPC_HORIZON = 300;
constexpr double MPC_DT = 1.0 / MPC_HORIZON;
constexpr int MPC_HALF_HORIZON = MPC_HORIZON / 2;
struct TrajPoint {
    double yaw;

    double v_yaw;

    double pitch;

    double v_pitch;
};
struct VeryAimer::Impl {
public:
    using TrajectoryVec = std::vector<TrajPoint>;

    Impl(const YAML::Node& config, std::shared_ptr<TrajectoryCompensator> trajectory_compensator) {
        trajectory_compensator_ = trajectory_compensator;
        config_ = config;
        reset();
    }
    void reset() {
        shooting_range_w_ = config_["shooting_range_w"].as<double>(0.12);
        shooting_range_h_ = config_["shooting_range_h"].as<double>(0.12);
        double yaw_limit_deg = config_["yaw_limit"].as<double>(0.0);
        yaw_limit = yaw_limit_deg / 180.0 * M_PI;
        min_enable_pitch_deg_ = config_["min_enable_pitch_deg"].as<double>(0.0);
        min_enable_yaw_deg_ = config_["min_enable_yaw_deg"].as<double>(0.0);
        manual_compensator_ = std::make_unique<ManualCompensator>();
        std::vector<OffsetEntry> entries;

        if (config_["trajectory_offset"]) {
            for (const auto& node: config_["trajectory_offset"]) {
                OffsetEntry e;
                e.d_min = node["d_min"].as<double>();
                e.d_max = node["d_max"].as<double>();
                e.h_min = node["h_min"].as<double>();
                e.h_max = node["h_max"].as<double>();
                e.pitch_off = node["pitch_off"].as<double>();
                e.yaw_off = node["yaw_off"].as<double>();
                entries.push_back(e);
            }
            manual_compensator_->setBasePitch(config_["base_offset"]["pitch"].as<double>());
            manual_compensator_->setBaseYaw(config_["base_offset"]["yaw"].as<double>());
        }
        if (!manual_compensator_->updateMapFlow(entries) || entries.size() < 1) {
            std::cout << "Trajectory compensator init failed" << std::endl;
        }
        prediction_delay_ = config_["prediction_delay"].as<double>(0.0);
        comming_angle_ = config_["comming_angle"].as<double>(5.0);
        leaving_angle_ = config_["leaving_angle"].as<double>(5.0);
        control_delay_ = config_["control_delay"].as<double>();
        max_iter_ = config_["max_iter"].as<int>();
        max_pitch_acc_ = config_["max_pitch_acc"].as<double>();
        delay_enable_fire_error_ = config_["delay_enable_fire_error"].as<double>(0.0);
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
        Eigen::MatrixXd u_max_pitch = Eigen::MatrixXd::Constant(1, MPC_HORIZON - 1, max_pitch_acc_);
        tiny_set_bound_constraints(
            pitch_solver_,
            x_min_pitch,
            x_max_pitch,
            u_min_pitch,
            u_max_pitch
        );

        pitch_solver_->settings->max_iter = max_iter_;
        max_yaw_acc_ = config_["max_yaw_acc"].as<double>();
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
        Eigen::MatrixXd u_min_yaw = Eigen::MatrixXd::Constant(1, MPC_HORIZON - 1, -max_yaw_acc_);
        Eigen::MatrixXd u_max_yaw = Eigen::MatrixXd::Constant(1, MPC_HORIZON - 1, max_yaw_acc_);
        tiny_set_bound_constraints(yaw_solver_, x_min_yaw, x_max_yaw, u_min_yaw, u_max_yaw);

        yaw_solver_->settings->max_iter = max_iter_;
    }
    static Ptr create(
        const YAML::Node& config,
        std::shared_ptr<TrajectoryCompensator> trajectory_compensator
    ) {
        return std::make_unique<VeryAimer>(config, trajectory_compensator);
    }
    int selectArmor(const Target& target, bool aim_first, bool aim_pair) {
        static int lock_id = -1;

        auto armor_list = target.getArmorPosAndYaw();
        int armor_num = static_cast<int>(armor_list.size());
        int i_chosen = 0;

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
                    int pick = (lock_id >= 0 && lock_id < armor_num)
                        ? lock_id
                        : pick_best_by_min_abs(candidates);
                    if (pick >= 0) {
                        i_chosen = pick;
                    }
                } else {
                    lock_id = -1;
                    int pick = candidates[0];
                    i_chosen = pick;
                }
            }

            return i_chosen;
        }
        if (armor_num > 0) {
            int best_idx = -1;

            if (target.tracked_id_ != armor::ArmorNumber::OUTPOST) {
                double coming_angle = comming_angle_ * M_PI / 180.0;
                double leaving_angle = leaving_angle_ * M_PI / 180.0;

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
            if (aim_pair) {
                std::vector<int> all;
                all.push_back(0);
                all.push_back(2);
                best_idx = pick_best_by_min_abs(all);
            }

            i_chosen = best_idx;
        }

        return i_chosen;
    }
    ControlPoint
    choseAndGetControlPoint(Target target, double bullet_speed, const AutoAimFsm& auto_aim_fsm) {
        const bool aim_first = (auto_aim_fsm == AutoAimFsm::AIM_SINGLE_ARMOR);
        const bool aim_center = (auto_aim_fsm == AutoAimFsm::AIM_WHOLE_CAR_CENTER);
        const bool aim_pair = (auto_aim_fsm == AutoAimFsm::AIM_WHOLE_CAR_PAIR);
        int target_select = selectArmor(target, aim_first, aim_pair);
        auto armors_xyza = target.getArmorPosAndYaw();
        Eigen::Vector3d aim_pos = armors_xyza[target_select].head<3>();

        if (aim_center) {
            double raw_z = aim_pos.z();
            double c_xy_dis = std::sqrt(
                target.position().x() * target.position().x()
                + target.position().y() * target.position().y()
            );
            double c_yaw = std::atan2(target.position().y(), target.position().x());
            c_xy_dis -= target.getArmor2CenterXYDis(target_select);
            aim_pos.x() = c_xy_dis * std::cos(c_yaw);
            aim_pos.y() = c_xy_dis * std::sin(c_yaw);
            aim_pos.z() = raw_z;
        }
        ControlPoint cp = getControlPoint(aim_pos, bullet_speed);
        cp.id_in_target = target_select;
        return cp;
    }
    TrajectoryVec getTrajectory(
        Target& target,
        const ControlPoint& cp0,
        double bullet_speed,
        const AutoAimFsm& auto_aim_fsm
    ) {
        TrajectoryVec traj;
        traj.reserve(MPC_HORIZON);

        target.predict(-MPC_DT * (MPC_HALF_HORIZON + 1));
        auto cp_last = choseAndGetControlPoint(target, bullet_speed, auto_aim_fsm);
        target.predict(MPC_DT);
        auto cp = choseAndGetControlPoint(target, bullet_speed, auto_aim_fsm);

        for (int i = 0; i < MPC_HORIZON; i++) {
            target.predict(MPC_DT);
            auto cp_next = choseAndGetControlPoint(target, bullet_speed, auto_aim_fsm);

            double yaw_vel = angles::normalize_angle(cp_next.yaw - cp_last.yaw) / (2 * MPC_DT);
            double pitch_vel = (cp_next.pitch - cp_last.pitch) / (2 * MPC_DT);

            TrajPoint pt;
            pt.yaw = angles::normalize_angle(cp.yaw - cp0.yaw);
            pt.v_yaw = yaw_vel;
            pt.pitch = cp.pitch;
            pt.v_pitch = pitch_vel;

            traj.push_back(pt);

            cp_last = cp;
            cp = cp_next;
        }

        return traj;
    }
    ControlPoint getControlPoint(Eigen::Vector3d aim_target_pos, double bullet_speed) {
        ControlPoint cp;
        double control_yaw = std::atan2(aim_target_pos.y(), aim_target_pos.x());
        double raw_pitch = std::atan2(
            aim_target_pos.z(),
            std::sqrt(
                aim_target_pos.x() * aim_target_pos.x() + aim_target_pos.y() * aim_target_pos.y()
            )
        );
        try {
            trajectory_compensator_->compensate(aim_target_pos, raw_pitch, bullet_speed);
        } catch (std::exception& e) {
            std::cout << "compensate error: " << e.what() << std::endl;
        }

        double control_pitch = raw_pitch;
        auto offs = manual_compensator_->angleHardCorrect(
            aim_target_pos.head(2).norm(),
            aim_target_pos.z()
        );
        control_yaw = angles::normalize_angle((control_yaw + offs[1] * M_PI / 180.0));
        control_pitch = (control_pitch + offs[0] * M_PI / 180.0);
        cp.pitch = control_pitch;
        cp.yaw = control_yaw;
        return cp;
    }
    TrajPoint getStateAtTime(const TrajectoryVec& traj, double t) {
        double total_time = (traj.size() - 1) * MPC_DT;
        Eigen::Vector4d result_yaw_vyaw_pitch_vpitch;
        if (t <= 0.0)
            return traj.front();
        if (t >= total_time) {
            return traj.back();
        }

        const double idx = t / MPC_DT;
        int i0 = floor(idx);
        int i1 = ceil(idx);
        if (i0 == i1)
            return traj[i0];

        const double a = idx - i0;
        const Eigen::Vector4d s0 { traj[i0].yaw, traj[i0].v_yaw, traj[i0].pitch, traj[i0].v_pitch };
        const Eigen::Vector4d s1 { traj[i1].yaw, traj[i1].v_yaw, traj[i1].pitch, traj[i1].v_pitch };
        result_yaw_vyaw_pitch_vpitch = (1 - a) * s0 + a * s1;
        TrajPoint result;
        result.yaw = result_yaw_vyaw_pitch_vpitch[0];
        result.v_yaw = result_yaw_vyaw_pitch_vpitch[1];
        result.pitch = result_yaw_vyaw_pitch_vpitch[2];
        result.v_pitch = result_yaw_vyaw_pitch_vpitch[3];
        return result;
    }
    std::tuple<double, double>
    calEnableDiff(Eigen::Vector3d aim_target_pos, double diff_yaw, const AutoAimFsm& auto_aim_fsm) {
        const double distance = aim_target_pos.norm();
        double shooting_range_yaw = std::abs(atan2(shooting_range_w_ / 2, distance));
        double shooting_range_pitch = std::abs(atan2(shooting_range_h_ / 2, distance));
        double yaw_factor = 0.0;
        const double yaw_rad = diff_yaw;
        if (auto_aim_fsm != AutoAimFsm::AIM_SINGLE_ARMOR) {
            if (std::abs(yaw_rad) <= yaw_limit) {
                yaw_factor = std::cos(yaw_rad);
            }
        } else {
            yaw_factor = std::cos(yaw_rad);
        }

        const double pitch_factor = std::cos(15.0 * M_PI / 180);

        shooting_range_yaw = std::max(shooting_range_yaw, min_enable_yaw_deg_ * M_PI / 180);
        shooting_range_pitch = std::max(shooting_range_pitch, min_enable_pitch_deg_ * M_PI / 180);
        shooting_range_yaw *= yaw_factor;
        shooting_range_pitch *= pitch_factor;

        return std::make_tuple(std::abs(shooting_range_yaw), std::abs(shooting_range_pitch));
    }
    inline double rad2deg(double r) {
        return r / M_PI * 180.0;
    }
    GimbalCmd veryAim(Target target, double bullet_speed, const AutoAimFsm& auto_aim_fsm) {
        const bool aim_first = (auto_aim_fsm == AutoAimFsm::AIM_SINGLE_ARMOR);
        const bool aim_center = (auto_aim_fsm == AutoAimFsm::AIM_WHOLE_CAR_CENTER);
        const bool aim_pair = (auto_aim_fsm == AutoAimFsm::AIM_WHOLE_CAR_PAIR);
        const int roughly_select = selectArmor(target, aim_first, aim_pair);
        const auto ap = target.getArmorPositions();
        const auto now = time_utils::now();
        const double dt0 = time_utils::durationSec(target.timestamp_, now);
        const auto future = now + std::chrono::microseconds(int(dt0 * 1e6));
        target.predict(future);
        const double fly_time =
            trajectory_compensator_->getFlyingTime(ap[roughly_select].head<3>(), bullet_speed);
        std::vector<Target> iteration_target(10, target);
        bool converged = false;
        double prev_fly_time = fly_time;
        for (int iter = 0; iter < 10; ++iter) {
            auto predict_time = prev_fly_time;
            iteration_target[iter].predict(predict_time);
            int iter_select = selectArmor(iteration_target[iter], aim_first, aim_pair);
            auto iter_poss = iteration_target[iter].getArmorPositions();
            double iter_fly_time =
                trajectory_compensator_->getFlyingTime(iter_poss[iter_select], bullet_speed);
            if (std::abs(iter_fly_time - prev_fly_time) < 0.001) {
                converged = true;
                break;
            }
            prev_fly_time = iter_fly_time;
        }
        auto predict_time = prev_fly_time + prediction_delay_;
        target.predict(predict_time);
        const int fin_target_select = selectArmor(target, aim_first, aim_pair);
        const auto fin_armors_xyza = target.getArmorPosAndYaw();
        Eigen::Vector3d fin_aim_pos = fin_armors_xyza[fin_target_select].head<3>();

        if (aim_center)
        { //这里的逻辑要时刻与getControlPoint统一，这里为了获取变量写在了外面 ，该段的意义是使击打yaw为中心yaw，而击打点的xy要推回原选板
            double raw_z = fin_aim_pos.z();
            double c_xy_dis = std::sqrt(
                target.position().x() * target.position().x()
                + target.position().y() * target.position().y()
            );
            double c_yaw = std::atan2(target.position().y(), target.position().x());
            c_xy_dis -= target.getArmor2CenterXYDis(fin_target_select);
            fin_aim_pos.x() = c_xy_dis * std::cos(c_yaw);
            fin_aim_pos.y() = c_xy_dis * std::sin(c_yaw);
            fin_aim_pos.z() = raw_z;
        }

        AimTarget fin_target_at;
        {
            fin_target_at.pos = fin_aim_pos;
            fin_target_at.valid = true;
            Eigen::Vector3d euler;
            euler.x() = M_PI / 2.0;
            euler.y() = (target.tracked_id_ == armor::ArmorNumber::OUTPOST) ? -0.2618 : 0.2618;
            euler.z() = target.yaw();
            Eigen::Quaterniond ori = utils::eulerToQuat(euler, utils::EulerOrder::ZYX);
            fin_target_at.ori = ori;
        }

        GimbalCmd cmd;
        cmd.aim_target = fin_target_at;
        ControlPoint cp0;
        TrajectoryVec traj;
        try {
            cp0 = getControlPoint(fin_aim_pos, bullet_speed);
            cp0.id_in_target = fin_target_select;
            traj = getTrajectory(target, cp0, bullet_speed, auto_aim_fsm);
        } catch (const std::exception& e) {
            WUST_WARN("very_aimer") << "mpc solver error!";
            cmd.appera = false;
            return cmd;
        }
        auto trajVecToEigen = [](const TrajectoryVec& traj) {
            int N = traj.size();
            Eigen::Matrix<double, 4, Eigen::Dynamic> mat(4, N);
            for (int i = 0; i < N; i++) {
                mat(0, i) = traj[i].yaw;
                mat(1, i) = traj[i].v_yaw;
                mat(2, i) = traj[i].pitch;
                mat(3, i) = traj[i].v_pitch;
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

        if (!yaw_solver_->work->status || !pitch_solver_->work->status) {
            WUST_WARN("very_aimer") << "mpc solver error!";
            std::cout << "mpc solver error!" << std::endl;
            cmd.appera = false;
            return cmd;
        }
        const double total_time = MPC_HORIZON * MPC_DT;
        double target_yaw_rad = cp0.yaw;
        double target_pitch_rad = cp0.pitch;
        if (aim_center) {
            auto cp_aim_center =
                getControlPoint(fin_armors_xyza[fin_target_select].head<3>(), bullet_speed);
            target_yaw_rad = cp_aim_center.yaw;
            target_pitch_rad = cp_aim_center.pitch;
        }
        const double control_yaw_rad =
            angles::normalize_angle(yaw_solver_->work->x(0, MPC_HALF_HORIZON) + cp0.yaw);
        const double control_yaw_vel_rad = yaw_solver_->work->x(1, MPC_HALF_HORIZON);
        const double control_yaw_acc_rad = yaw_solver_->work->u(0, MPC_HALF_HORIZON);
        const double control_pitch_rad = pitch_solver_->work->x(0, MPC_HALF_HORIZON);
        const double control_pitch_vel_rad = pitch_solver_->work->x(1, MPC_HALF_HORIZON);
        const double control_pitch_acc_rad = pitch_solver_->work->u(0, MPC_HALF_HORIZON);
        cmd.yaw = rad2deg(control_yaw_rad);
        cmd.v_yaw = rad2deg(control_yaw_vel_rad);
        cmd.pitch = rad2deg(control_pitch_rad);
        cmd.v_pitch = rad2deg(control_pitch_vel_rad);
        cmd.target_yaw = rad2deg(target_yaw_rad);
        cmd.target_pitch = rad2deg(target_pitch_rad);
        cmd.raw_yaw = rad2deg(target_yaw_rad);
        cmd.raw_pitch = rad2deg(target_pitch_rad);
        cmd.distance = fin_aim_pos.norm();
        cmd.fly_time = prev_fly_time;
        auto delay_state = getStateAtTime(traj, total_time / 2.0 + control_delay_);
        TrajectoryVec control_traj;
        control_traj.resize(MPC_HORIZON);
        for (int i = 0; i < MPC_HORIZON; i++) {
            TrajPoint tp;
            tp.yaw = yaw_solver_->work->x(0, i);
            tp.v_yaw = yaw_solver_->work->x(1, i);
            tp.pitch = pitch_solver_->work->x(0, i);
            tp.v_pitch = pitch_solver_->work->x(1, i);
            control_traj[i] = tp;
        }
        const auto control_delay_state =
            getStateAtTime(control_traj, total_time / 2.0 + control_delay_);

        bool delay_noswitching = true;
        delay_noswitching = std::hypot(
                                angles::normalize_angle(delay_state.yaw + cp0.yaw)
                                    - angles::normalize_angle(control_delay_state.yaw + cp0.yaw),
                                angles::normalize_angle(delay_state.pitch)
                                    - angles::normalize_angle(control_delay_state.pitch)
                            )
            < delay_enable_fire_error_;

        static int noswitching_count = 0;
        static int switching_count = 0;
        if (delay_noswitching) {
            double center_yaw = std::atan2(target.position().y(), target.position().x());
            double d_angle = angles::shortest_angular_distance(
                center_yaw,
                fin_armors_xyza[fin_target_select][3]
            );
            const auto [enable_yaw_diff_rad, enable_pitch_diff_rad] =
                calEnableDiff(fin_armors_xyza[fin_target_select].head<3>(), d_angle, auto_aim_fsm);
            cmd.enable_pitch_diff = rad2deg(enable_pitch_diff_rad);
            cmd.enable_yaw_diff = rad2deg(enable_yaw_diff_rad);
            const double yaw_diff_rad =
                angles::shortest_angular_distance(target_yaw_rad, control_yaw_rad);
            const double pitch_diff_rad =
                angles::shortest_angular_distance(target_pitch_rad, control_pitch_rad);
            if (std::abs(yaw_diff_rad) <= enable_yaw_diff_rad
                && std::abs(pitch_diff_rad) <= enable_pitch_diff_rad)
            {
                cmd.fire_advice = true;
            }
            noswitching_count++;
        } else { //发弹延迟后进入换板阶段则该帧完全不开火
            cmd.fire_advice = false;
            cmd.enable_pitch_diff = 0;
            cmd.enable_yaw_diff = 0;
            switching_count++;
        }
        utils::XSecOnce(
            [&] {
                int total = switching_count + noswitching_count;
                WUST_INFO("very_aimer") << "switching_count: " << switching_count << " ("
                                        << (switching_count * 100.0) / total << "%)"
                                        << "  |  noswitching_count: " << noswitching_count << " ("
                                        << (noswitching_count * 100.0) / total << "%)";
                switching_count = 0;
                noswitching_count = 0;
            },
            1.0
        );
        cmd.appera = true;
        if (!cmd.isValid()) {
            cmd.appera = false;
            reset();
            WUST_WARN("very_aimer") << "very_aimer  nan!";
        }
        return cmd;
    }

    std::shared_ptr<TrajectoryCompensator> trajectory_compensator_;
    std::unique_ptr<ManualCompensator> manual_compensator_;
    double shooting_range_w_ = 0.135;
    double shooting_range_h_ = 0.135;
    double min_enable_yaw_deg_ = 0.5;
    double min_enable_pitch_deg_ = 0.5;
    double yaw_limit = 60.0 / 180.0 * M_PI;
    double prediction_delay_;
    double comming_angle_;
    double leaving_angle_;
    double control_delay_ = 0.01;
    double max_yaw_acc_, max_pitch_acc_;
    int max_iter_ = 10;
    double delay_enable_fire_error_ = 0.0035;
    TinySolver* yaw_solver_;
    TinySolver* pitch_solver_;
    minco::MINCO_S3NU1DAngle minco_yaw_;
    YAML::Node config_;
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
GimbalCmd VeryAimer::veryAim(Target target, double bullet_speed, const AutoAimFsm& auto_aim_fsm) {
    return _impl->veryAim(target, bullet_speed, auto_aim_fsm);
}
} // namespace auto_aim
