#pragma once
#include "tasks/auto_aim/armor_tracker/target.hpp"
#include "tasks/auto_aim/auto_aim_fsm.hpp"
#include "tasks/type_common.hpp"
#include "traj.hpp"
#include "wust_vl/common/utils/manual_compensator.hpp"
namespace wust_vision {
namespace auto_aim {
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
        static inline AimPoint lerp(const AimPoint& p0, const AimPoint& p1, double a) noexcept {
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
        static GimbalState lerp(const GimbalState& s0, const GimbalState& s1, double a) noexcept {
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

        static inline Eigen::Matrix<double, 6, 1> solve1dFullPivLu(
            double p0,
            double v0,
            double a0,
            double p1,
            double v1,
            double a1,
            double T
        ) noexcept {
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
        static inline Eigen::Matrix<double, 6, 1> solve1dClosedForm(
            double p0,
            double v0,
            double a0,
            double p1,
            double v1,
            double a1,
            double T
        ) noexcept {
            Eigen::Matrix<double, 6, 1> c;
            double T2 = T * T;
            double T3 = T2 * T;
            double T4 = T3 * T;
            double T5 = T4 * T;

            // known low-order coefficients
            double c0 = p0;
            double c1 = v0;
            double c2 = a0 * 0.5;

            // closed-form for c3, c4, c5 (derived from boundary conditions at t=T)
            double c3 =
                (-3.0 * T2 * a0 + T2 * a1 - 12.0 * T * v0 - 8.0 * T * v1 - 20.0 * p0 + 20.0 * p1)
                / (2.0 * T3);
            double c4 =
                (1.5 * T2 * a0 - T2 * a1 + 8.0 * T * v0 + 7.0 * T * v1 + 15.0 * p0 - 15.0 * p1)
                / T4;
            double c5 = (-T2 * a0 + T2 * a1 - 6.0 * T * v0 - 6.0 * T * v1 - 12.0 * p0 + 12.0 * p1)
                / (2.0 * T5);

            c << c0, c1, c2, c3, c4, c5;
            return c;
        }

        static inline QuinticSegment
        build(const GimbalState::State& s0, const GimbalState::State& s1, double T) noexcept {
            QuinticSegment seg;
            seg.head = s0;
            seg.tail = s1;
            seg.T = T;
            seg.c = solve1dClosedForm(s0.p, s0.v, s0.a, s1.p, s1.v, s1.a, T);
            return seg;
        }

        static inline double evalAcc(const Eigen::Matrix<double, 6, 1>& c, double t) noexcept {
            return 2 * c[2] + 6 * c[3] * t + 12 * c[4] * t * t + 20 * c[5] * t * t * t;
        }
        static inline double maxAbsAcc(const Eigen::Matrix<double, 6, 1>& c, double T) noexcept {
            if (T <= 0.0)
                return 0.0;

            auto acc = [&](double t) {
                double t2 = t * t;
                return 2 * c[2] + 6 * c[3] * t + 12 * c[4] * t2 + 20 * c[5] * t2 * t;
            };

            double max_acc = std::max(std::abs(acc(0.0)), std::abs(acc(T)));

            // jerk = 6c3 + 24c4 t + 60c5 t^2
            double A = 60.0 * c[5];
            double B = 24.0 * c[4];
            double C = 6.0 * c[3];

            const double eps = 1e-9;

            if (std::abs(A) < eps) {
                if (std::abs(B) > eps) {
                    double t = -C / B;
                    if (t > 0.0 && t < T)
                        max_acc = std::max(max_acc, std::abs(acc(t)));
                }
            } else {
                double D = B * B - 4 * A * C;
                if (D >= 0.0) {
                    double sqrtD = std::sqrt(D);
                    double inv2A = 1.0 / (2 * A);

                    double t1 = (-B + sqrtD) * inv2A;
                    double t2 = (-B - sqrtD) * inv2A;

                    if (t1 > 0.0 && t1 < T)
                        max_acc = std::max(max_acc, std::abs(acc(t1)));
                    if (t2 > 0.0 && t2 < T)
                        max_acc = std::max(max_acc, std::abs(acc(t2)));
                }
            }

            return std::isfinite(max_acc) ? max_acc : 0.0;
        }

        double inline duration() const noexcept {
            return T;
        }

        double inline MaxAcc(void) const noexcept {
            return QuinticSegment::maxAbsAcc(c, T);
        }
        GimbalState::State inline eval(double t) const noexcept {
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
        static inline double angleDiff(double a, double b) noexcept {
            double d = a - b;
            while (d > M_PI)
                d -= 2 * M_PI;
            while (d < -M_PI)
                d += 2 * M_PI;
            return d;
        }

        static inline double unwrapAngle(double prev, double curr) noexcept {
            return prev + angleDiff(curr, prev);
        }

        void unwrapStates(std::vector<GimbalState>& s) const noexcept {
            for (size_t i = 1; i < s.size(); ++i) {
                s[i].yaw_state.p = unwrapAngle(s[i - 1].yaw_state.p, s[i].yaw_state.p);
                s[i].pitch_state.p = unwrapAngle(s[i - 1].pitch_state.p, s[i].pitch_state.p);
            }
        }

        std::pair<std::vector<GimbalState::State>, std::vector<GimbalState::State>>
        computeNodeStates(const std::vector<GimbalState>& gp, const std::vector<double>& dt)
            const noexcept {
            const size_t N = gp.size();
            std::vector<GimbalState::State> yaw(N), pitch(N);
            for (size_t i = 0; i < N; ++i) {
                yaw[i] = gp[i].yaw_state;
                pitch[i] = gp[i].pitch_state;
            }
            if (N < 2)
                return { yaw, pitch };
            auto compute_va = [&](std::vector<GimbalState::State>& s) {
                // 边界
                s.front().v = s.back().v = 0.0;
                s.front().a = s.back().a = 0.0;

                for (size_t i = 1; i + 1 < N; ++i) {
                    const double dt0 = dt[i - 1];
                    const double dt1 = dt[i];
                    const double denom = dt0 + dt1;

                    if (denom < 1e-6) {
                        s[i].v = s[i].a = 0.0;
                        continue;
                    }

                    const double w0 = dt1 / denom;
                    const double w1 = dt0 / denom;

                    s[i].v = w0 * (s[i].p - s[i - 1].p) / dt0 + w1 * (s[i + 1].p - s[i].p) / dt1;

                    s[i].a =
                        2.0 * ((s[i + 1].p - s[i].p) / dt1 - (s[i].p - s[i - 1].p) / dt0) / denom;
                }
            };
            compute_va(yaw);
            compute_va(pitch);

            return { yaw, pitch };
        }

        void limitTraj(
            Traj& traj,
            const std::vector<GimbalState::State>& s,
            int near_change_idx,
            double max_acc
        ) const noexcept {
            traj.segs.clear();
            traj.seg_dt.clear();
            traj.seg_prefix_time.clear();

            const int N = static_cast<int>(s.size());
            if (N <= 1)
                return;

            const auto& _prefix_time = prefix_time;
            const auto& _dt_vec = dt_vec;

            auto buildSeg = [&](int l, int r) -> QuinticSegment {
                double dur = _prefix_time[r] - _prefix_time[l];
                return QuinticSegment::build(s[l], s[r], dur);
            };

            std::optional<std::pair<int, int>> interval;
            if (near_change_idx >= 0) {
                int l = std::clamp(near_change_idx, 0, N - 1);
                int r = std::clamp(near_change_idx + 1, 0, N - 1);
                if (l < r)
                    interval.emplace(l, r);
            }

            if (!interval) {
                traj.segs.reserve(N - 1);
                for (int i = 0; i < N - 1; ++i)
                    traj.segs.push_back(QuinticSegment::build(s[i], s[i + 1], _dt_vec[i]));

                traj.seg_dt.reserve(traj.segs.size());
                for (const auto& seg: traj.segs)
                    traj.seg_dt.push_back(seg.duration());

                traj.seg_prefix_time.resize(traj.segs.size() + 1);
                traj.seg_prefix_time[0] = 0.0;
                for (size_t i = 0; i < traj.seg_dt.size(); ++i)
                    traj.seg_prefix_time[i + 1] = traj.seg_prefix_time[i] + traj.seg_dt[i];
                return;
            }

            {
                int& l = interval->first;
                int& r = interval->second;
                QuinticSegment seg = buildSeg(l, r);

                auto try_candidate = [&](int nl, int nr) -> bool {
                    nl = std::max(0, nl);
                    nr = std::min(N - 1, nr);
                    if (nl == l && nr == r)
                        return false;

                    QuinticSegment cand = buildSeg(nl, nr);
                    if (cand.MaxAcc() <= seg.MaxAcc()) {
                        l = nl;
                        r = nr;
                        seg = std::move(cand);
                        return true;
                    }
                    return false;
                };

                while (seg.MaxAcc() > max_acc) {
                    bool expanded = false;

                    if (l > 0 || r < N - 1)
                        expanded = try_candidate(l - 1, r + 1);

                    if (!expanded && l > 0)
                        expanded = try_candidate(l - 1, r);

                    if (!expanded && r < N - 1)
                        expanded = try_candidate(l, r + 1);

                    if (!expanded && (l > 0 || r < N - 1)) {
                        int nl = std::max(0, l - 1);
                        int nr = std::min(N - 1, r + 1);
                        QuinticSegment forceSeg = buildSeg(nl, nr);

                        if (forceSeg.MaxAcc() < seg.MaxAcc() || (nl == 0 && nr == N - 1)) {
                            l = nl;
                            r = nr;
                            seg = std::move(forceSeg);
                            expanded = true;
                        }
                    }

                    if (!expanded)
                        break;
                    if (l == 0 && r == N - 1 && seg.MaxAcc() > max_acc)
                        break;
                }
            }

            traj.segs.reserve(N - 1);
            for (int i = 0; i < N - 1; ++i) {
                if (interval && i == interval->first) {
                    traj.segs.push_back(buildSeg(interval->first, interval->second));
                    i = interval->second - 1; // skip covered indices
                } else {
                    traj.segs.push_back(QuinticSegment::build(s[i], s[i + 1], _dt_vec[i]));
                }
            }

            traj.seg_dt.reserve(traj.segs.size());
            for (const auto& seg: traj.segs)
                traj.seg_dt.push_back(seg.duration());

            traj.seg_prefix_time.resize(traj.segs.size() + 1);
            traj.seg_prefix_time[0] = 0.0;
            for (size_t i = 0; i < traj.seg_dt.size(); ++i)
                traj.seg_prefix_time[i + 1] = traj.seg_prefix_time[i] + traj.seg_dt[i];
        }

        void buildLimit(double max_yaw_acc, double max_pitch_acc) noexcept {
            unwrapStates(cp_vec);
            auto [yaw_states, pitch_states] = computeNodeStates(cp_vec, dt_vec);
            int best_idx = -1;
            double best_dist = 1e100;
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

                if (dist < best_dist) {
                    best_dist = dist;
                    best_idx = static_cast<int>(i);
                }
            }
            limitTraj(yaw_traj, yaw_states, best_idx, max_yaw_acc);
            limitTraj(pitch_traj, pitch_states, best_idx, max_pitch_acc);
        }
        void simpleTraj(Traj& traj, const std::vector<GimbalState::State>& s) const noexcept {
            traj.segs.clear();
            traj.seg_dt.clear();
            traj.seg_prefix_time.clear();

            const int N = static_cast<int>(s.size());
            if (N <= 1)
                return;
            for (int i = 0; i < N - 1; ++i) {
                traj.segs.push_back(QuinticSegment::build(s[i], s[i + 1], dt_vec[i]));
            }
            traj.seg_dt.reserve(traj.segs.size());
            for (const auto& seg: traj.segs)
                traj.seg_dt.push_back(seg.duration());

            traj.seg_prefix_time.resize(traj.segs.size() + 1);
            traj.seg_prefix_time[0] = 0.0;
            for (size_t i = 0; i < traj.seg_dt.size(); ++i)
                traj.seg_prefix_time[i + 1] = traj.seg_prefix_time[i] + traj.seg_dt[i];
        }
        void buildSimple() {
            unwrapStates(cp_vec);
            auto [yaw_states, pitch_states] = computeNodeStates(cp_vec, dt_vec);
            simpleTraj(yaw_traj, yaw_states);
            simpleTraj(pitch_traj, pitch_states);
        }
        inline GimbalState::State getStateAtTime(double t, const Traj& traj) const noexcept {
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
        inline GimbalState getStateAtTime(double t) const noexcept {
            GimbalState::State yaw = getStateAtTime(t, yaw_traj);
            GimbalState::State pitch = getStateAtTime(t, pitch_traj);
            return GimbalState(yaw, pitch);
        }
    };
    class VeryAimerTrajBase {
    public:
        using Ptr = std::shared_ptr<VeryAimerTrajBase>;
        LimitTrajectory target_traj;
        Trajectory<AimPoint> aim_traj;
        ControlPoint cp0;
        AutoAimFsm fsm;
        Eigen::Vector3d fin_aim_pos;
        AimTarget aim_target;
        virtual GimbalState getTargetState(double t) const = 0;
        virtual GimbalState getControlState(double t) const = 0;
    };
    struct VeryAimerConfig: wust_vl::common::utils::ParamGroup {
        static constexpr const char* Logger = "Config: very_aimer";
        static constexpr const char* kKey = "very_aimer";
        const char* key() const override {
            return kKey;
        }
        using Ptr = std::shared_ptr<VeryAimerConfig>;
        VeryAimerConfig() {
            sample_total_time_param.onChange([this](double o, double n) {
                sample_dt = sample_total_time_param.get() / sample_horizon_param.get();
                sample_half_horizon = sample_horizon_param.get() / 2;
            });
            sample_horizon_param.onChange([this](double o, double n) {
                sample_dt = sample_total_time_param.get() / sample_horizon_param.get();
                sample_half_horizon = sample_horizon_param.get() / 2;
            });
        }
        static Ptr create() {
            return std::make_shared<VeryAimerConfig>();
        }
        std::shared_ptr<wust_vl::common::utils::ManualCompensator> manual_compensator;
        GEN_PARAM(double, sample_total_time);
        GEN_PARAM(int, sample_horizon);
        GEN_PARAM(double, control_delay);
        GEN_PARAM(double, delay_enable_fire_error);
        GEN_PARAM(double, max_yaw_acc);
        GEN_PARAM(double, max_pitch_acc);
        GEN_PARAM(double, prediction_delay);
        GEN_PARAM(double, comming_angle);
        GEN_PARAM(double, leaving_angle);
        GEN_PARAM(double, yaw_limit_deg);
        GEN_PARAM(double, shooting_range_h);
        GEN_PARAM(double, shooting_range_w);
        GEN_PARAM(double, min_enable_pitch_deg);
        GEN_PARAM(double, min_enable_yaw_deg);
        double sample_dt = 0.01;
        int sample_half_horizon = 100;
        bool first_load = false;
        struct Mpc {
            int max_iter;
            std::vector<double> Q_pitch;
            std::vector<double> Q_yaw;
            std::vector<double> R_pitch;
            std::vector<double> R_yaw;
            void load(const YAML::Node& node) {
                max_iter = node["max_iter"].as<int>();
                Q_pitch = node["Q_pitch"].as<std::vector<double>>();
                Q_yaw = node["Q_yaw"].as<std::vector<double>>();
                R_pitch = node["R_pitch"].as<std::vector<double>>();
                R_yaw = node["R_yaw"].as<std::vector<double>>();
            }
        } mpc;
        void loadSelf(const YAML::Node& node) override {
            if (!first_load) {
                manual_compensator = std::make_shared<wust_vl::common::utils::ManualCompensator>();
                std::vector<wust_vl::common::utils::OffsetEntry> entries;

                if (node["trajectory_offset"]) {
                    for (const auto& node: node["trajectory_offset"]) {
                        wust_vl::common::utils::OffsetEntry e;
                        e.d_min = node["d_min"].as<double>();
                        e.d_max = node["d_max"].as<double>();
                        e.h_min = node["h_min"].as<double>();
                        e.h_max = node["h_max"].as<double>();
                        e.pitch_off = node["pitch_off"].as<double>();
                        e.yaw_off = node["yaw_off"].as<double>();
                        entries.push_back(e);
                    }
                    manual_compensator->setBasePitch(node["base_offset"]["pitch"].as<double>());
                    manual_compensator->setBaseYaw(node["base_offset"]["yaw"].as<double>());
                }
                if (!manual_compensator->updateMapFlow(entries) || entries.size() < 1) {
                    std::cout << "Trajectory compensator init failed" << std::endl;
                }
                mpc.load(node);
                first_load = true;
            } else {
            }
            shooting_range_h_param.load(node);
            shooting_range_w_param.load(node);
            yaw_limit_deg_param.load(node);
            min_enable_pitch_deg_param.load(node);
            min_enable_yaw_deg_param.load(node);
            prediction_delay_param.load(node);
            comming_angle_param.load(node);
            leaving_angle_param.load(node);
            control_delay_param.load(node);
            delay_enable_fire_error_param.load(node);
            max_yaw_acc_param.load(node);
            max_pitch_acc_param.load(node);
            sample_total_time_param.load(node);
            sample_horizon_param.load(node);
        }
    };
    class VeryAimerBase {
    public:
        using Ptr = std::unique_ptr<VeryAimerBase>;
        VeryAimerBase(wust_vl::common::utils::Parameter::Ptr auto_aim_config_parameter) {
            auto_aim_config_parameter_ = auto_aim_config_parameter;
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
        [[nodiscard]] inline FireResult
        canFireAtTime(const VeryAimerTrajBase::Ptr& traj, double t) const noexcept;
        void reset();
        [[nodiscard]] int
        selectArmor(const Target& target, const AutoAimFsm& auto_aim_fsm) const noexcept;
        [[nodiscard]] ControlPoint
        getControlPoint(Eigen::Vector3d aim_target_pos, double diff_yaw, double bullet_speed)
            const noexcept;
        [[nodiscard]] std::tuple<double, double> calEnableDiff(
            Eigen::Vector3d aim_target_pos,
            double diff_yaw,
            const AutoAimFsm& auto_aim_fsm
        ) const noexcept;
        [[nodiscard]] inline double rad2deg(double r) const noexcept {
            return r / M_PI * 180.0;
        }
        [[nodiscard]] std::tuple<bool, bool, bool> getAimStatus(const AutoAimFsm& auto_aim_fsm
        ) const noexcept;

        [[nodiscard]] ControlPoint choseAndGetControlPoint(
            const Target& target,
            double bullet_speed,
            const AutoAimFsm& auto_aim_fsm
        ) const noexcept;
        [[nodiscard]] std::pair<LimitTrajectory, Trajectory<AimPoint>> getTrajectory(
            Target& target,
            const ControlPoint& cp0,
            double bullet_speed,
            const AutoAimFsm& auto_aim_fsm
        ) const;
        [[nodiscard]] virtual GimbalCmd
        veryAim(Target target, double bullet_speed, const AutoAimFsm& auto_aim_fsm) = 0;
        TrajectoryCompensatorConfig::Ptr trajectory_compensator_config_;
        wust_vl::common::utils::Parameter::Ptr auto_aim_config_parameter_;
        VeryAimerConfig::Ptr config_;
    };

} // namespace auto_aim
} // namespace wust_vision