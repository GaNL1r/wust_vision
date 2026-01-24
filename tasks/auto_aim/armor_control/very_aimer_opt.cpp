#include "very_aimer_opt.hpp"
#include "gcopter/minco.hpp"
#include "tasks/auto_aim/armor_control/tinympc/tiny_api.hpp"
#include "tasks/auto_aim/armor_control/tinympc/types.hpp"
#include "traj.hpp"
#include "wust_vl/common/utils/manual_compensator.hpp"
namespace auto_aim {
namespace very_aimer_opt {
    static inline double lerpAngle(double a0, double a1, double t) {
        double d = std::remainder(a1 - a0, 2.0 * M_PI);
        return a0 + t * d;
    }
    struct ControlPoint {
        double yaw;
        double pitch;
        int id_in_target;
        Eigen::Vector4d xyza;
    };
    struct AimPoint {
        Eigen::Vector3d pos;
        double d_angle;
        static inline AimPoint lerp(const AimPoint& p0, const AimPoint& p1, double a) {
            AimPoint r;
            r.pos = (1.0 - a) * p0.pos + a * p1.pos;
            r.d_angle = lerpAngle(p0.d_angle, p1.d_angle, a);
            return r;
        }
    };
    struct GimbalState {
        struct State {
            double p;
            double v;
            double a;
        };
        State yaw_state;
        State pitch_state;
        int aim_id = 0;
        GimbalState() {}
        GimbalState(const GimbalState::State& y, const GimbalState::State& p):
            yaw_state(y),
            pitch_state(p) {}
        static GimbalState lerp(const GimbalState& s0, const GimbalState& s1, double a) {
            GimbalState r;
            r.aim_id = (a > 0.5) ? s0.aim_id : s1.aim_id;
            r.yaw_state = GimbalState::State { .p = lerpAngle(s0.yaw_state.p, s1.yaw_state.p, a),
                                               .v = std::lerp(s0.yaw_state.v, s1.yaw_state.v, a),
                                               .a = std::lerp(s0.yaw_state.a, s1.yaw_state.a, a) };
            r.pitch_state =
                GimbalState::State { .p = lerpAngle(s0.pitch_state.p, s1.pitch_state.p, a),
                                     .v = std::lerp(s0.pitch_state.v, s1.pitch_state.v, a),
                                     .a = std::lerp(s0.pitch_state.a, s1.pitch_state.a, a) };

            return r;
        }
    };

    struct QuinticSegment {
        double T = 0.0;
        Eigen::Matrix<double, 6, 1> c;
        GimbalState::State head;
        GimbalState::State tail;

        static Eigen::Matrix<double, 6, 1>
        solve1d(double p0, double v0, double a0, double p1, double v1, double a1, double T) {
            Eigen::Matrix<double, 6, 6> A;
            Eigen::Matrix<double, 6, 1> b;

            double T2 = T * T, T3 = T2 * T, T4 = T3 * T, T5 = T4 * T;

            // Rows: p(0)=p0, p'(0)=v0, p''(0)=a0,
            //       p(T)=p1, p'(T)=v1, p''(T)=a1
            A << 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 1, T, T2, T3, T4, T5, 0, 1,
                2 * T, 3 * T2, 4 * T3, 5 * T4, 0, 0, 2, 6 * T, 12 * T2, 20 * T3;

            b << p0, v0, a0, p1, v1, a1;
            return A.fullPivLu().solve(b);
        }
        static QuinticSegment
        build(const GimbalState::State& s0, const GimbalState::State& s1, double T) {
            QuinticSegment seg;
            seg.head = s0;
            seg.tail = s1;
            seg.T = T;
            seg.c = solve1d(s0.p, s0.v, s0.a, s1.p, s1.v, s1.a, T);
            return seg;
        }

        static double evalAcc(const Eigen::Matrix<double, 6, 1>& c, double t) {
            return 2 * c[2] + 6 * c[3] * t + 12 * c[4] * t * t + 20 * c[5] * t * t * t;
        }
        static double maxAbsAcc(const Eigen::Matrix<double, 6, 1>& c, double T) {
            if (T <= 0.0)
                return 0.0;

            auto acc = [&](double t) {
                return 2 * c[2] + 6 * c[3] * t + 12 * c[4] * t * t + 20 * c[5] * t * t * t;
            };

            auto safe_update = [&](double t, double& max_acc) {
                if (t <= 0.0 || t >= T)
                    return;
                double a = acc(t);
                if (std::isfinite(a))
                    max_acc = std::max(max_acc, std::abs(a));
            };

            double max_acc = 0.0;
            safe_update(0.0, max_acc);
            safe_update(T, max_acc);

            // jerk = 6c3 + 24c4 t + 60c5 t^2
            double A = 60.0 * c[5];
            double B = 24.0 * c[4];
            double C = 6.0 * c[3];

            const double eps = 1e-9; // 放宽阈值

            if (std::abs(A) < eps) {
                if (std::abs(B) > eps) {
                    safe_update(-C / B, max_acc);
                }
            } else {
                double D = B * B - 4 * A * C;
                if (D >= 0.0) {
                    double sqrtD = std::sqrt(D);
                    safe_update((-B + sqrtD) / (2 * A), max_acc);
                    safe_update((-B - sqrtD) / (2 * A), max_acc);
                }
            }

            if (!std::isfinite(max_acc))
                return 0.0;

            return max_acc;
        }
        double duration() const {
            return T;
        }

        double MaxAcc(void) const {
            return QuinticSegment::maxAbsAcc(c, T);
        }
        GimbalState::State eval(double t) const {
            GimbalState::State s;
            if (T <= 0.0)
                return s;
            t = std::clamp(t, 0.0, T);
            double t2 = t * t, t3 = t2 * t, t4 = t3 * t, t5 = t4 * t;
            s.p = c[0] + c[1] * t + c[2] * t2 + c[3] * t3 + c[4] * t4 + c[5] * t5;
            s.v = c[1] + 2 * c[2] * t + 3 * c[3] * t2 + 4 * c[4] * t3 + 5 * c[5] * t4;
            s.a = evalAcc(c, t);
            return s;
        }
    };
    class LimitTrajectory: public Trajectory<GimbalState> {
    public:
        struct Traj {
            std::vector<QuinticSegment> segs;
            std::vector<double> seg_dt;
            std::vector<double> seg_prefix_time;
        };

        Traj yaw_traj;
        Traj pitch_traj;
        static double angleDiff(double a, double b) {
            double d = a - b;
            while (d > M_PI)
                d -= 2 * M_PI;
            while (d < -M_PI)
                d += 2 * M_PI;
            return d;
        }

        static double unwrapAngle(double prev, double curr) {
            return prev + angleDiff(curr, prev);
        }

        void unwrapStates(std::vector<GimbalState>& s) {
            for (size_t i = 1; i < s.size(); ++i) {
                s[i].yaw_state.p = unwrapAngle(s[i - 1].yaw_state.p, s[i].yaw_state.p);
                s[i].pitch_state.p = unwrapAngle(s[i - 1].pitch_state.p, s[i].pitch_state.p);
            }
        }

        std::pair<std::vector<GimbalState::State>, std::vector<GimbalState::State>>
        computeNodeStates(const std::vector<GimbalState>& gp, const std::vector<double>& dt) {
            size_t N = gp.size();
            std::vector<GimbalState::State> yaw_states;
            std::vector<GimbalState::State> pitch_states;
            yaw_states.reserve(N);
            pitch_states.reserve(N);
            for (size_t i = 0; i < N; ++i) {
                yaw_states.push_back(gp[i].yaw_state);
                pitch_states.push_back(gp[i].pitch_state);
            }

            if (N < 2) {
                return std::make_pair(yaw_states, pitch_states);
            }

            // 边界速度
            yaw_states.front().v = pitch_states.front().v = 0.0;
            yaw_states.back().v = pitch_states.back().v = 0.0;

            for (size_t i = 1; i + 1 < N; ++i) {
                double dt0 = dt[i - 1];
                double dt1 = dt[i];
                double denom = dt0 + dt1;

                if (denom < 1e-6) {
                    yaw_states[i].v = pitch_states[i].v = 0.0;
                    continue;
                }

                double w0 = dt1 / denom;
                double w1 = dt0 / denom;

                yaw_states[i].v = w0 * (gp[i].yaw_state.p - gp[i - 1].yaw_state.p) / dt0
                    + w1 * (gp[i + 1].yaw_state.p - gp[i].yaw_state.p) / dt1;

                pitch_states[i].v = w0 * (gp[i].pitch_state.p - gp[i - 1].pitch_state.p) / dt0
                    + w1 * (gp[i + 1].pitch_state.p - gp[i].pitch_state.p) / dt1;
            }

            // 边界加速度
            yaw_states.front().a = pitch_states.front().a = 0.0;
            yaw_states.back().a = pitch_states.back().a = 0.0;

            for (size_t i = 1; i + 1 < N; ++i) {
                double dt0 = dt[i - 1];
                double dt1 = dt[i];
                double denom = dt0 + dt1;

                if (denom < 1e-6) {
                    yaw_states[i].a = pitch_states[i].a = 0.0;
                    continue;
                }

                yaw_states[i].a = 2.0
                    * ((gp[i + 1].yaw_state.p - gp[i].yaw_state.p) / dt1
                       - (gp[i].yaw_state.p - gp[i - 1].yaw_state.p) / dt0)
                    / denom;

                pitch_states[i].a = 2.0
                    * ((gp[i + 1].pitch_state.p - gp[i].pitch_state.p) / dt1
                       - (gp[i].pitch_state.p - gp[i - 1].pitch_state.p) / dt0)
                    / denom;
            }

            return std::make_pair(yaw_states, pitch_states);
        }
        void limitTraj(
            Traj& traj,
            const std::vector<GimbalState::State>& s,
            int best_front_idx,
            int best_back_idx,
            double max_acc
        ) const {
            traj.segs.clear();
            traj.seg_dt.clear();
            traj.seg_prefix_time.clear();
            std::vector<int> change_segs_start_idx;
            std::vector<int> change_segs_end_idx;
            std::vector<QuinticSegment> change_segs;
            const int N = static_cast<int>(s.size());
            const auto& _prefix_time = prefix_time;
            const auto& _dt_vec = dt_vec;
            auto push_change_seg = [&](int idx) {
                if (idx < 0)
                    return;

                change_segs_start_idx.push_back(idx);
                change_segs_end_idx.push_back(idx + 1);
                change_segs.push_back(QuinticSegment::build(
                    s[idx],
                    s[idx + 1],
                    _prefix_time[idx + 1] - _prefix_time[idx]
                ));
            };

            push_change_seg(best_front_idx);
            push_change_seg(best_back_idx);

            bool updated = true;
            while (updated) {
                updated = false;

                for (size_t i = 0; i < change_segs.size(); ++i) {
                    if (change_segs[i].MaxAcc() <= max_acc)
                        continue;

                    int old_l = change_segs_start_idx[i];
                    int old_r = change_segs_end_idx[i];

                    int new_l = std::max(0, old_l - 1);
                    int new_r = std::min(N - 1, old_r + 1);

                    if (new_l == old_l && new_r == old_r)
                        continue;

                    change_segs_start_idx[i] = new_l;
                    change_segs_end_idx[i] = new_r;

                    change_segs[i] = QuinticSegment::build(
                        s[new_l],
                        s[new_r],
                        _prefix_time[new_r] - _prefix_time[new_l]
                    );

                    updated = true;
                }

                for (size_t i = 0; i < change_segs.size(); ++i) {
                    for (size_t j = i + 1; j < change_segs.size();) {
                        if (change_segs_start_idx[j] <= change_segs_end_idx[i]
                            && change_segs_start_idx[i] <= change_segs_end_idx[j])
                        {
                            int l = std::min(change_segs_start_idx[i], change_segs_start_idx[j]);
                            int r = std::max(change_segs_end_idx[i], change_segs_end_idx[j]);

                            change_segs_start_idx[i] = l;
                            change_segs_end_idx[i] = r;

                            change_segs[i] = QuinticSegment::build(
                                s[l],
                                s[r],
                                _prefix_time[r] - _prefix_time[l]
                            );

                            change_segs_start_idx.erase(change_segs_start_idx.begin() + j);
                            change_segs_end_idx.erase(change_segs_end_idx.begin() + j);
                            change_segs.erase(change_segs.begin() + j);

                            updated = true;
                        } else {
                            ++j;
                        }
                    }
                }
            }
            std::vector<bool> covered(N - 1, false);
            for (size_t i = 0; i < change_segs.size(); ++i) {
                for (int j = change_segs_start_idx[i]; j < change_segs_end_idx[i]; ++j)
                    covered[j] = true;
            }

            size_t change_idx = 0;
            for (int i = 0; i < N - 1; ++i) {
                if (change_idx < change_segs.size() && i == change_segs_start_idx[change_idx]) {
                    traj.segs.push_back(change_segs[change_idx]);
                    ++change_idx;
                    continue;
                }

                if (!covered[i]) {
                    traj.segs.push_back(QuinticSegment::build(s[i], s[i + 1], _dt_vec[i]));
                }
            }

            for (const auto& seg: traj.segs)
                traj.seg_dt.push_back(seg.duration());

            traj.seg_prefix_time.resize(traj.segs.size() + 1);
            traj.seg_prefix_time[0] = 0.0;
            for (size_t i = 0; i < traj.seg_dt.size(); ++i)
                traj.seg_prefix_time[i + 1] = traj.seg_prefix_time[i] + traj.seg_dt[i];
        }
        void build(double max_yaw_acc, double max_pitch_acc) {
            unwrapStates(cp_vec);
            auto [yaw_states, pitch_states] = computeNodeStates(cp_vec, dt_vec);
            int best_front_idx = -1;
            int best_back_idx = -1;
            double best_front_dist = 1e100;
            double best_back_dist = 1e100;
            const size_t offset = 2;
            const int N = static_cast<int>(cp_vec.size());
            if (N < 2)
                return;
            const double mid_time = 0.5 * total_duration_;
            for (size_t i = offset; i + offset + 1 < cp_vec.size(); ++i) {
                if (cp_vec[i].aim_id == cp_vec[i + 1].aim_id)
                    continue;

                const double seg_mid = 0.5 * (prefix_time[i] + prefix_time[i + 1]);
                const double dist = std::abs(seg_mid - mid_time);

                if (seg_mid <= mid_time) {
                    if (dist < best_front_dist) {
                        best_front_dist = dist;
                        best_front_idx = static_cast<int>(i);
                    }
                } else {
                    if (dist < best_back_dist) {
                        best_back_dist = dist;
                        best_back_idx = static_cast<int>(i);
                    }
                }
            }
            limitTraj(yaw_traj, yaw_states, best_front_idx, best_back_idx, max_yaw_acc);
            limitTraj(pitch_traj, pitch_states, best_front_idx, best_back_idx, max_pitch_acc);
        }
        GimbalState::State getStateAtTime(double t, const Traj& traj) const {
            if (traj.segs.empty())
                return {};

            if (t <= 0.0)
                return traj.segs.front().eval(0.0);

            if (t >= total_duration_)
                return traj.segs.back().eval(traj.segs.back().T);

            const auto it =
                std::upper_bound(traj.seg_prefix_time.begin(), traj.seg_prefix_time.end(), t);

            size_t i = std::distance(traj.seg_prefix_time.begin(), it) - 1;
            i = std::min(i, traj.segs.size() - 1);

            const double t0 = traj.seg_prefix_time[i];
            return traj.segs[i].eval(t - t0);
        }
        GimbalState getStateAtTime(double t) const {
            GimbalState::State yaw = getStateAtTime(t, yaw_traj);
            GimbalState::State pitch = getStateAtTime(t, pitch_traj);
            return GimbalState(yaw, pitch);
        }
    };

    class VerAimerTraj {
    public:
        using Ptr = std::shared_ptr<VerAimerTraj>;
        VerAimerTraj(

        ) {}
        static Ptr create() {
            return std::make_shared<VerAimerTraj>();
        }
        LimitTrajectory target_traj;
        Trajectory<AimPoint> aim_traj;
        ControlPoint cp0;
        AutoAimFsm fsm;
        Eigen::Vector3d fin_aim_pos;
        AimTarget aim_target;
    };

    struct VeryAimerOpt::Impl {
    public:
        Impl(
            const YAML::Node& config,
            std::shared_ptr<TrajectoryCompensator> trajectory_compensator
        ) {
            trajectory_compensator_ = trajectory_compensator;
            config_ = config;
            reset();
        }
        static constexpr int MPC_HORIZON = 300;
        static constexpr double MPC_DT = 1.0 / MPC_HORIZON;
        static constexpr int MPC_HALF_HORIZON = MPC_HORIZON / 2;
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
            max_yaw_acc_ = config_["max_yaw_acc"].as<double>();
        }
        int selectArmor(const Target& target, const AutoAimFsm& auto_aim_fsm) {
            static int lock_id = -1;
            const auto [aim_first, aim_center, aim_pair] = getAimStatus(auto_aim_fsm);
            const auto armor_list = target.getArmorPosAndYaw();
            const int armor_num = static_cast<int>(armor_list.size());
            int i_chosen = 0;

            const double center_yaw = std::atan2(target.position().y(), target.position().x());

            std::vector<double> delta_angles;
            delta_angles.reserve(armor_num);
            for (int i = 0; i < armor_num; ++i) {
                delta_angles.push_back(angles::normalize_angle(armor_list[i][3] - center_yaw));
            }

            const auto pick_best_by_min_abs = [&](const std::vector<int>& idxs) -> int {
                int best = -1;
                double best_val = std::numeric_limits<double>::infinity();
                for (int i: idxs) {
                    const double val = std::abs(delta_angles[i]);
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
                            lock_id =
                                (std::abs(delta_angles[a]) < std::abs(delta_angles[b])) ? a : b;
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
                    const double coming_angle = comming_angle_ * M_PI / 180.0;
                    const double leaving_angle = leaving_angle_ * M_PI / 180.0;

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
        ControlPoint choseAndGetControlPoint(
            const Target& target,
            double bullet_speed,
            const AutoAimFsm& auto_aim_fsm
        ) {
            const auto [aim_first, aim_center, aim_pair] = getAimStatus(auto_aim_fsm);
            const int target_select = selectArmor(target, auto_aim_fsm);
            const auto armors_xyza = target.getArmorPosAndYaw();
            Eigen::Vector3d aim_pos = armors_xyza[target_select].head<3>();

            if (aim_center) {
                const double raw_z = aim_pos.z();
                double c_xy_dis = std::sqrt(
                    target.position().x() * target.position().x()
                    + target.position().y() * target.position().y()
                );
                const double c_yaw = std::atan2(target.position().y(), target.position().x());
                c_xy_dis -= target.getArmor2CenterXYDis(target_select);
                aim_pos.x() = c_xy_dis * std::cos(c_yaw);
                aim_pos.y() = c_xy_dis * std::sin(c_yaw);
                aim_pos.z() = raw_z;
            }
            const double center_yaw = std::atan2(target.position().y(), target.position().x());
            const double d_angle =
                angles::shortest_angular_distance(center_yaw, armors_xyza[target_select][3]);
            ControlPoint cp = getControlPoint(aim_pos, d_angle, bullet_speed);
            cp.id_in_target = target_select;
            return cp;
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
                pt.aim_id = cp.id_in_target;
                traj.push_back(pt, MPC_DT);
                AimPoint aim_pt;
                aim_pt.d_angle = cp.xyza[3];
                aim_pt.pos = cp.xyza.head<3>();
                aim_traj.push_back(aim_pt, MPC_DT);
                cp_last = cp;
                cp = cp_next;
            }
            traj.build(max_yaw_acc_, max_pitch_acc_);
            return { std::move(traj), std::move(aim_traj) };
        }
        ControlPoint
        getControlPoint(Eigen::Vector3d aim_target_pos, double diff_yaw, double bullet_speed) {
            ControlPoint cp;
            double control_yaw = std::atan2(aim_target_pos.y(), aim_target_pos.x());
            double raw_pitch = std::atan2(
                aim_target_pos.z(),
                std::sqrt(
                    aim_target_pos.x() * aim_target_pos.x()
                    + aim_target_pos.y() * aim_target_pos.y()
                )
            );
            try {
                trajectory_compensator_->compensate(aim_target_pos, raw_pitch, bullet_speed);
            } catch (std::exception& e) {
                std::cout << "compensate error: " << e.what() << std::endl;
            }

            double control_pitch = raw_pitch;
            const auto offs = manual_compensator_->angleHardCorrect(
                aim_target_pos.head(2).norm(),
                aim_target_pos.z()
            );
            control_yaw = angles::normalize_angle((control_yaw + offs[1] * M_PI / 180.0));
            control_pitch = (control_pitch + offs[0] * M_PI / 180.0);
            cp.pitch = control_pitch;
            cp.yaw = control_yaw;
            cp.xyza.head<3>() = aim_target_pos;
            cp.xyza[3] = diff_yaw;
            return cp;
        }
        std::tuple<double, double> calEnableDiff(
            Eigen::Vector3d aim_target_pos,
            double diff_yaw,
            const AutoAimFsm& auto_aim_fsm
        ) {
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
            shooting_range_pitch =
                std::max(shooting_range_pitch, min_enable_pitch_deg_ * M_PI / 180);
            shooting_range_yaw *= yaw_factor;
            shooting_range_pitch *= pitch_factor;

            return std::make_tuple(std::abs(shooting_range_yaw), std::abs(shooting_range_pitch));
        }
        inline double rad2deg(double r) {
            return r / M_PI * 180.0;
        }

        struct FireResult {
            bool fire;
            double enable_yaw_diff;
            double enable_pitch_diff;
            FireResult(bool f, double ey, double ep):
                fire(f),
                enable_yaw_diff(ey),
                enable_pitch_diff(ep) {}
        };
        inline FireResult canFireAtTime(const VerAimerTraj::Ptr& traj, double t) {
            const auto target_delay =
                traj->target_traj.LimitTrajectory::getStateAtTime(t + control_delay_);
            const auto control_delay =
                traj->target_traj.Trajectory::getStateAtTime(t + control_delay_);

            if (std::hypot(
                    angles::normalize_angle(target_delay.yaw_state.p + traj->cp0.yaw)
                        - angles::normalize_angle(control_delay.yaw_state.p + traj->cp0.yaw),
                    angles::normalize_angle(target_delay.pitch_state.p)
                        - angles::normalize_angle(control_delay.pitch_state.p)
                )
                >= delay_enable_fire_error_)
            {
                return { false, 0, 0 };
            }

            const auto gimbal = traj->target_traj.LimitTrajectory::getStateAtTime(t);
            const auto control = traj->target_traj.Trajectory::getStateAtTime(t);
            const auto aim = traj->aim_traj.getStateAtTime(t);

            const double target_yaw = angles::normalize_angle(gimbal.yaw_state.p + traj->cp0.yaw);
            const double control_yaw = angles::normalize_angle(control.yaw_state.p + traj->cp0.yaw);

            const auto [enable_yaw, enable_pitch] = calEnableDiff(aim.pos, aim.d_angle, traj->fsm);

            return { std::abs(angles::shortest_angular_distance(target_yaw, control_yaw))
                             <= enable_yaw
                         && std::abs(angles::shortest_angular_distance(
                                gimbal.pitch_state.p,
                                control.pitch_state.p
                            ))
                             <= enable_pitch,
                     enable_yaw,
                     enable_pitch };
        }

        double calTrajectoryScore(const VerAimerTraj::Ptr& traj) {
            const double half_t = traj->target_traj.getPrefixTimeAtIdx(MPC_HALF_HORIZON);

            double score = 0;

            for (int i = -MPC_HALF_HORIZON; i < MPC_HALF_HORIZON; ++i) {
                const double t = i * MPC_DT + half_t;
                auto fire_r = canFireAtTime(traj, t);
                if (fire_r.fire) {
                    score += fire_r.enable_yaw_diff;
                }
            }

            return score;
        }
        std::tuple<bool, bool, bool> getAimStatus(const AutoAimFsm& auto_aim_fsm) {
            const bool aim_first = (auto_aim_fsm == AutoAimFsm::AIM_SINGLE_ARMOR);
            const bool aim_center = (auto_aim_fsm == AutoAimFsm::AIM_WHOLE_CAR_CENTER);
            const bool aim_pair = (auto_aim_fsm == AutoAimFsm::AIM_WHOLE_CAR_PAIR);
            return std::make_tuple(aim_first, aim_center, aim_pair);
        }

        VerAimerTraj::Ptr buildAimAndTrajectory(
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
                build =
                    buildAimAndTrajectory(target, fin_target_select, bullet_speed, auto_aim_fsm);
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

            const auto control_state = build->target_traj.LimitTrajectory::getStateAtTime(half_t);

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

            const auto fire_now = canFireAtTime(build, half_t);
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
        YAML::Node config_;
    };
    VeryAimerOpt::VeryAimerOpt(
        const YAML::Node& config,
        std::shared_ptr<TrajectoryCompensator> trajectory_compensator
    ) {
        _impl = std::make_unique<Impl>(config, trajectory_compensator);
    }
    VeryAimerOpt::~VeryAimerOpt() {
        _impl.reset();
    }
    GimbalCmd
    VeryAimerOpt::veryAim(Target target, double bullet_speed, const AutoAimFsm& auto_aim_fsm) {
        return _impl->veryAim(target, bullet_speed, auto_aim_fsm);
    }
} // namespace very_aimer_opt
} // namespace auto_aim
