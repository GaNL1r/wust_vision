#include "very_aimer.hpp"
#include "tasks/auto_aim/armor_control/tinympc/tiny_api.hpp"
#include "tasks/auto_aim/armor_control/tinympc/types.hpp"
#include "tasks/auto_aim/armor_tracker/target.hpp"
#include "tasks/auto_aim/auto_aim_fsm.hpp"
#include "tasks/type_common.hpp"
#include "traj.hpp"
#include "wust_vl/common/utils/manual_compensator.hpp"
#include "wust_vl/common/utils/parameter.hpp"
#include <wust_vl/common/utils/logger.hpp>
namespace wust_vision::auto_aim {

struct VeryAimer::Impl {
    [[nodiscard]] static inline double lerpAngle(double a0, double a1, double t) noexcept {
        double d = std::remainder(a1 - a0, 2.0 * M_PI);
        return a0 + t * d;
    }
    [[nodiscard]] inline static double rad2deg(double r) noexcept {
        static constexpr double rad_deg = 180.0 / M_PI;
        return r * rad_deg;
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

        [[nodiscard]] static inline QuinticSegment
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

        [[nodiscard]] double inline duration() const noexcept {
            return T;
        }

        [[nodiscard]] double inline MaxAcc(void) const noexcept {
            return QuinticSegment::maxAbsAcc(c, T);
        }
        [[nodiscard]] GimbalState::State inline eval(double t) const noexcept {
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

        [[nodiscard]] std::pair<std::vector<GimbalState::State>, std::vector<GimbalState::State>>
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
        [[nodiscard]] inline GimbalState::State
        getStateAtTime(double t, const Traj& traj) const noexcept {
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
        [[nodiscard]] inline GimbalState getStateAtTime(double t) const noexcept {
            GimbalState::State yaw = getStateAtTime(t, yaw_traj);
            GimbalState::State pitch = getStateAtTime(t, pitch_traj);
            return GimbalState(yaw, pitch);
        }
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
        GEN_PARAM(bool, fuck_test);
        GEN_PARAM(double, fuck_test_thresh);
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
            fuck_test_param.load(node);
            fuck_test_thresh_param.load(node);
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
    class VeryAimerTrajMpc: public VeryAimerTrajBase {
    public:
        using Ptr = std::shared_ptr<VeryAimerTrajMpc>;
        VeryAimerTrajMpc(

        ) {}
        static Ptr create() {
            return std::make_shared<VeryAimerTrajMpc>();
        }
        Trajectory<GimbalState> control_traj;
        GimbalState getTargetState(double t) const override {
            return target_traj.LimitTrajectory::getStateAtTime(t);
        }
        GimbalState getControlState(double t) const override {
            return control_traj.getStateAtTime(t);
        }
    };
    class VeryAimerTrajSeg: public VeryAimerTrajBase {
    public:
        using Ptr = std::shared_ptr<VeryAimerTrajSeg>;
        VeryAimerTrajSeg() {}
        static Ptr create() {
            return std::make_shared<VeryAimerTrajSeg>();
        }
        GimbalState getTargetState(double t) const override {
            return target_traj.Trajectory::getStateAtTime(t);
        }
        GimbalState getControlState(double t) const override {
            return target_traj.LimitTrajectory::getStateAtTime(t);
        }
    };
    enum class Type : int {
        Mpc = 0,
        Seg = 1,
    } type_;
    Impl(wust_vl::common::utils::Parameter::Ptr auto_aim_config_parameter) {
        auto_aim_config_parameter_ = auto_aim_config_parameter;
        reset();
    }

    void reset() {
        config_ = VeryAimerConfig::create();
        trajectory_compensator_config_ = TrajectoryCompensatorConfig::create();
        auto_aim_config_parameter_->registerGroup(*trajectory_compensator_config_);
        auto_aim_config_parameter_->registerGroup(*config_);
        auto_aim_config_parameter_->reloadFromOldPath();
        const auto yaml = auto_aim_config_parameter_->getConfig();
        const auto type = yaml["very_aimer"]["type"].as<std::string>();
        if (type == "mpc" || type == "MPC") {
            type_ = Type::Mpc;
        } else if (type == "seg" || type == "SEG") {
            type_ = Type::Seg;

        } else {
            type_ = Type::Seg;
        }
        if (type_ == Type::Mpc) {
            mpcReset();
        }
    }
    int max_iter_ = 10;
    TinySolver* yaw_solver_;
    TinySolver* pitch_solver_;
    void mpcReset() {
        const int horizon = config_->sample_horizon_param.get();
        const int half_horizon = config_->sample_half_horizon;
        const double dt = config_->sample_dt;
        Eigen::MatrixXd A_pitch { { 1, dt }, { 0, 1 } };
        Eigen::MatrixXd B_pitch { { 0 }, { dt } };
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
            horizon,
            0
        );

        Eigen::MatrixXd x_min_pitch = Eigen::MatrixXd::Constant(2, horizon, -1e17);
        Eigen::MatrixXd x_max_pitch = Eigen::MatrixXd::Constant(2, horizon, 1e17);
        Eigen::MatrixXd u_min_pitch =
            Eigen::MatrixXd::Constant(1, horizon - 1, -config_->max_pitch_acc_param.get());
        Eigen::MatrixXd u_max_pitch =
            Eigen::MatrixXd::Constant(1, horizon - 1, config_->max_pitch_acc_param.get());
        tiny_set_bound_constraints(
            pitch_solver_,
            x_min_pitch,
            x_max_pitch,
            u_min_pitch,
            u_max_pitch
        );
        pitch_solver_->settings->max_iter = max_iter_;
        Eigen::MatrixXd A_yaw { { 1, dt }, { 0, 1 } };
        Eigen::MatrixXd B_yaw { { 0 }, { dt } };
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
            horizon,
            0
        );

        Eigen::MatrixXd x_min_yaw = Eigen::MatrixXd::Constant(2, horizon, -1e17);
        Eigen::MatrixXd x_max_yaw = Eigen::MatrixXd::Constant(2, horizon, 1e17);
        Eigen::MatrixXd u_min_yaw =
            Eigen::MatrixXd::Constant(1, horizon - 1, -config_->max_yaw_acc_param.get());
        Eigen::MatrixXd u_max_yaw =
            Eigen::MatrixXd::Constant(1, horizon - 1, config_->max_yaw_acc_param.get());
        tiny_set_bound_constraints(yaw_solver_, x_min_yaw, x_max_yaw, u_min_yaw, u_max_yaw);
        yaw_solver_->settings->max_iter = max_iter_;
    }
    int selectArmor(const Target& target, const AutoAimFsm& auto_aim_fsm) const noexcept {
        static int lock_id = -1;
        const auto [aim_first, aim_center, aim_pair] = getAimStatus(auto_aim_fsm);
        const auto armor_list = target.getArmorPosAndYaw();
        const int armor_num = static_cast<int>(armor_list.size());
        int i_chosen = 0;

        const double center_yaw = std::atan2(target.target_state_.cy(), target.target_state_.cx());

        std::vector<double> delta_angles;
        delta_angles.reserve(armor_num);
        for (int i = 0; i < armor_num; ++i) {
            delta_angles.push_back(angles::normalize_angle(armor_list[i][3] - center_yaw));
        }

        const auto pick_best_by_min_delta = [&](const std::vector<int>& idxs) -> int {
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

        if (aim_first && target.tracked_id_ != ArmorNumber::OUTPOST && armor_num > 0) {
            std::vector<int> candidates;
            for (int i = 0; i < armor_num; ++i)
                if (std::abs(delta_angles[i]) <= 90.0 / 57.3)
                    candidates.push_back(i);

            if (!candidates.empty()) {
                if (candidates.size() > 1) {
                    int a = candidates[0], b = candidates[1];
                    if (lock_id != a && lock_id != b) {
                        lock_id = (std::abs(delta_angles[a]) < std::abs(delta_angles[b])) ? a : b;
                    }
                    int pick = (lock_id >= 0 && lock_id < armor_num)
                        ? lock_id
                        : pick_best_by_min_delta(candidates);
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

            if (target.tracked_id_ != ArmorNumber::OUTPOST) {
                const double coming_angle = config_->comming_angle_param.get() * M_PI / 180.0;
                const double leaving_angle = config_->leaving_angle_param.get() * M_PI / 180.0;

                for (int i = 0; i < armor_num; ++i) {
                    if (std::abs(delta_angles[i]) > coming_angle)
                        continue;

                    if (target.target_state_.vyaw() > 0 && delta_angles[i] < leaving_angle)
                        best_idx = i;
                    if (target.target_state_.vyaw() < 0 && delta_angles[i] > -leaving_angle)
                        best_idx = i;
                }
            }

            if (best_idx < 0) {
                std::vector<int> all(armor_num);
                std::iota(all.begin(), all.end(), 0);
                best_idx = pick_best_by_min_delta(all);
            }
            if (aim_pair && target.tracked_id_ != ArmorNumber::OUTPOST) {
                std::vector<int> all;
                if (target.target_state_.h() > 0) {
                    all.push_back(1);
                    all.push_back(3);
                } else {
                    all.push_back(0);
                    all.push_back(2);
                }
                all.push_back(0);
                best_idx = pick_best_by_min_delta(all);
            }

            i_chosen = best_idx;
        }

        return i_chosen;
    }
    ControlPoint
    getControlPoint(Eigen::Vector3d aim_target_pos, double diff_yaw, double bullet_speed) const {
        ControlPoint cp;
        double control_yaw = std::atan2(aim_target_pos.y(), aim_target_pos.x());
        double raw_pitch = std::atan2(
            aim_target_pos.z(),
            std::sqrt(
                aim_target_pos.x() * aim_target_pos.x() + aim_target_pos.y() * aim_target_pos.y()
            )
        );
        if (!trajectory_compensator_config_->trajectory_compensator
                 ->compensate(aim_target_pos, raw_pitch, bullet_speed))
        {
            WUST_ERROR("very_aimer") << " traj compense error";

            break_this_ = true;
            return cp;
        }

        double control_pitch = raw_pitch;
        const auto offs = config_->manual_compensator->angleHardCorrect(
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
    std::tuple<double, double>
    calEnableDiff(Eigen::Vector3d aim_target_pos, double diff_yaw, const AutoAimFsm& auto_aim_fsm)
        const noexcept {
        const double distance = aim_target_pos.norm();
        double shooting_range_yaw =
            std::abs(atan2(config_->shooting_range_w_param.get() / 2, distance));
        double shooting_range_pitch =
            std::abs(atan2(config_->shooting_range_h_param.get() / 2, distance));
        double yaw_factor = 0.0;

        const double yaw_rad = diff_yaw;
        if (auto_aim_fsm != AutoAimFsm::AIM_SINGLE_ARMOR) {
            if (std::abs(yaw_rad) <= config_->yaw_limit_deg_param.get() / 180.0 * M_PI) {
                yaw_factor = std::cos(yaw_rad);
            }
        } else {
            yaw_factor = std::cos(yaw_rad);
        }

        const double pitch_factor = std::cos(15.0 * M_PI / 180);

        shooting_range_yaw =
            std::max(shooting_range_yaw, config_->min_enable_yaw_deg_param.get() * M_PI / 180);
        shooting_range_pitch =
            std::max(shooting_range_pitch, config_->min_enable_pitch_deg_param.get() * M_PI / 180);
        shooting_range_yaw *= yaw_factor;
        shooting_range_pitch *= pitch_factor;

        return std::make_tuple(std::abs(shooting_range_yaw), std::abs(shooting_range_pitch));
    }
    std::tuple<bool, bool, bool> getAimStatus(const AutoAimFsm& auto_aim_fsm) const noexcept {
        const bool aim_first = (auto_aim_fsm == AutoAimFsm::AIM_SINGLE_ARMOR);
        const bool aim_center = (auto_aim_fsm == AutoAimFsm::AIM_WHOLE_CAR_CENTER);
        const bool aim_pair = (auto_aim_fsm == AutoAimFsm::AIM_WHOLE_CAR_PAIR);
        return std::make_tuple(aim_first, aim_center, aim_pair);
    }

    ControlPoint choseAndGetControlPoint(
        const Target& target,
        double bullet_speed,
        const AutoAimFsm& auto_aim_fsm
    ) const noexcept {
        const auto [aim_first, aim_center, aim_pair] = getAimStatus(auto_aim_fsm);
        const int target_select = selectArmor(target, auto_aim_fsm);
        const auto armors_xyza = target.getArmorPosAndYaw();
        Eigen::Vector3d aim_pos = armors_xyza[target_select].head<3>();

        if (aim_center) {
            const double raw_z = aim_pos.z();
            double c_xy_dis = std::sqrt(
                target.target_state_.cx() * target.target_state_.cx()
                + target.target_state_.cy() * target.target_state_.cy()
            );
            const double c_yaw = std::atan2(target.target_state_.cy(), target.target_state_.cx());
            c_xy_dis -= target.getArmor2CenterXYDis(target_select);
            aim_pos.x() = c_xy_dis * std::cos(c_yaw);
            aim_pos.y() = c_xy_dis * std::sin(c_yaw);
            aim_pos.z() = raw_z;
        }
        const double center_yaw = std::atan2(target.target_state_.cy(), target.target_state_.cx());
        const double d_angle =
            angles::shortest_angular_distance(center_yaw, armors_xyza[target_select][3]);
        ControlPoint cp = getControlPoint(aim_pos, d_angle, bullet_speed);
        cp.id_in_target = target_select;
        return cp;
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
    inline FireResult canFireAtTime(const VeryAimerTrajBase::Ptr& traj, double t) const noexcept {
        auto cal_delay = [&](double _t) {
            const auto target_delay = traj->getTargetState(_t + config_->control_delay_param.get());
            const auto control_delay =
                traj->getControlState(_t + config_->control_delay_param.get());

            if (std::hypot(
                    angles::normalize_angle(target_delay.yaw_state.p + traj->cp0.yaw)
                        - angles::normalize_angle(control_delay.yaw_state.p + traj->cp0.yaw),
                    angles::normalize_angle(target_delay.pitch_state.p)
                        - angles::normalize_angle(control_delay.pitch_state.p)
                )
                >= config_->delay_enable_fire_error_param.get())
            {
                return false;
            }
            return true;
        };
        if (!cal_delay(t)) {
            return { false, 0, 0 };
        }
        if (!cal_delay(-t)) {
            return { false, 0, 0 };
        }
        const auto target = traj->getTargetState(t);
        const auto control = traj->getControlState(t);
        const auto aim = traj->aim_traj.getStateAtTime(t);

        const double target_yaw = angles::normalize_angle(target.yaw_state.p + traj->cp0.yaw);
        const double control_yaw = angles::normalize_angle(control.yaw_state.p + traj->cp0.yaw);

        const auto [enable_yaw, enable_pitch] = calEnableDiff(aim.pos, aim.d_angle, traj->fsm);

        return { std::abs(angles::shortest_angular_distance(target_yaw, control_yaw)) <= enable_yaw
                     && std::abs(angles::shortest_angular_distance(
                            target.pitch_state.p,
                            control.pitch_state.p
                        ))
                         <= enable_pitch,
                 enable_yaw,
                 enable_pitch };
    }
    std::pair<LimitTrajectory, Trajectory<AimPoint>> getTrajectory(
        Target& target,
        const ControlPoint& cp0,
        double bullet_speed,
        const AutoAimFsm& auto_aim_fsm
    ) const {
        LimitTrajectory traj;
        Trajectory<AimPoint> aim_traj;
        const int horizon = config_->sample_horizon_param.get();
        const int half_horizon = config_->sample_half_horizon;
        const double dt = config_->sample_dt;

        traj.reserve(horizon);
        aim_traj.reserve(horizon);

        // prepare: roll the target back so we start from same relative time as original impl
        target.predictSimple(-dt * (half_horizon + 1));

        // compute first two cps (target state is mutated between calls but choseAndGetControlPoint takes const&)
        auto cp_last = choseAndGetControlPoint(target, bullet_speed, auto_aim_fsm);
        target.predictSimple(dt);
        auto cp = choseAndGetControlPoint(target, bullet_speed, auto_aim_fsm);

        for (int i = 0; i < horizon; ++i) {
            target.predictSimple(dt);
            const auto cp_next = choseAndGetControlPoint(target, bullet_speed, auto_aim_fsm);
            GimbalState pt;
            pt.yaw_state.p = angles::normalize_angle(cp.yaw - cp0.yaw);
            pt.pitch_state.p = cp.pitch;
            pt.aim_id = cp.id_in_target;
            traj.push_back(pt, dt);
            AimPoint aim_pt;
            aim_pt.d_angle = cp.xyza[3];
            aim_pt.pos = cp.xyza.head<3>();
            aim_traj.push_back(aim_pt, dt);
            cp_last = cp;
            cp = cp_next;
        }

        return { std::move(traj), std::move(aim_traj) };
    }
    template<typename VeryATraj, typename FinalizeFn>
    std::shared_ptr<VeryATraj> buildVeryAimerCommon(
        Target& target,
        int fin_target_select,
        double bullet_speed,
        const AutoAimFsm& auto_aim_fsm,
        FinalizeFn finalize
    ) {
        auto res = std::make_shared<VeryATraj>();

        const auto fin_armors_xyza = target.getArmorPosAndYaw();
        res->fin_aim_pos = fin_armors_xyza[fin_target_select].head<3>();

        const auto [aim_first, aim_center, aim_pair] = getAimStatus(auto_aim_fsm);

        if (aim_center) {
            const double raw_z = res->fin_aim_pos.z();
            double c_xy_dis = std::hypot(target.target_state_.cx(), target.target_state_.cy());
            const double c_yaw = std::atan2(target.target_state_.cy(), target.target_state_.cx());

            c_xy_dis -= target.getArmor2CenterXYDis(fin_target_select);

            res->fin_aim_pos.x() = c_xy_dis * std::cos(c_yaw);
            res->fin_aim_pos.y() = c_xy_dis * std::sin(c_yaw);
            res->fin_aim_pos.z() = raw_z;
        }

        // AimTarget
        {
            AimTarget at;
            at.pos = res->fin_aim_pos;
            at.valid = true;

            Eigen::Vector3d euler;
            euler.x() = M_PI / 2.0;
            euler.y() = (target.tracked_id_ == ArmorNumber::OUTPOST) ? -0.2618 : 0.2618;
            euler.z() = target.target_state_.yaw();

            at.ori = utils::eulerToQuat(euler, utils::EulerOrder::ZYX);
            res->aim_target = at;
        }

        const double center_yaw = std::atan2(target.target_state_.cy(), target.target_state_.cx());

        const double d_angle =
            angles::shortest_angular_distance(center_yaw, fin_armors_xyza[fin_target_select][3]);

        res->cp0 = getControlPoint(res->fin_aim_pos, d_angle, bullet_speed);
        res->cp0.id_in_target = fin_target_select;
        res->cp0.xyza = fin_armors_xyza[fin_target_select];

        auto traj = getTrajectory(target, res->cp0, bullet_speed, auto_aim_fsm);

        finalize(res, traj); // 差异点交给外部

        return res;
    }
    VeryAimerTrajSeg::Ptr buildVerAimerTrajSeg(
        Target& target,
        int fin_target_select,
        double bullet_speed,
        const AutoAimFsm& auto_aim_fsm
    ) {
        return buildVeryAimerCommon<VeryAimerTrajSeg>(
            target,
            fin_target_select,
            bullet_speed,
            auto_aim_fsm,
            [this](auto& res, auto& traj) {
                traj.first.buildLimit(
                    config_->max_yaw_acc_param.get(),
                    config_->max_pitch_acc_param.get()
                );
                res->target_traj = traj.first;
                res->aim_traj = traj.second;
            }
        );
    }

    VeryAimerTrajMpc::Ptr buildVerAimerTrajMpc(
        Target& target,
        int fin_target_select,
        double bullet_speed,
        const AutoAimFsm& auto_aim_fsm
    ) {
        return buildVeryAimerCommon<VeryAimerTrajMpc>(
            target,
            fin_target_select,
            bullet_speed,
            auto_aim_fsm,
            [this](auto& res, auto& traj) {
                traj.first.buildSimple();
                res->target_traj = traj.first;
                res->aim_traj = traj.second;
                res->control_traj = solveTrajectoryMpc(res->target_traj);
            }
        );
    }

    Trajectory<GimbalState> solveTrajectoryMpc(const Trajectory<GimbalState>& traj) {
        const double total_time = traj.getTotalDuration();
        const int horizon = config_->sample_horizon_param.get();
        const int half_horizon = config_->sample_half_horizon;
        const double dt = config_->sample_dt;
        const auto trajVecToEigen = [&](const Trajectory<GimbalState>& traj) {
            Eigen::Matrix<double, 4, Eigen::Dynamic> mat(4, horizon);

            const double half_t = total_time * 0.5;
            for (int k = 0; k < horizon; ++k) {
                int i = k - half_horizon;
                double t = i * dt + half_t;
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
        yaw_solver_->work->Xref = traj_eigen.block(0, 0, 2, horizon);
        tiny_solve(yaw_solver_);

        x0 << traj_eigen(2, 0), traj_eigen(3, 0);
        tiny_set_x0(pitch_solver_, x0);
        pitch_solver_->work->Xref = traj_eigen.block(2, 0, 2, horizon);
        tiny_solve(pitch_solver_);
        Trajectory<GimbalState> control_traj;
        control_traj.reserve(horizon);
        for (int i = 0; i < horizon; i++) {
            GimbalState tp;
            tp.yaw_state.p = yaw_solver_->work->x(0, i);
            tp.yaw_state.v = yaw_solver_->work->x(1, i);
            tp.pitch_state.p = pitch_solver_->work->x(0, i);
            tp.pitch_state.v = pitch_solver_->work->x(1, i);
            tp.yaw_state.a = yaw_solver_->work->u(0, i);
            tp.pitch_state.a = pitch_solver_->work->u(0, i);
            control_traj.push_back(tp, dt);
        }
        return control_traj;
    }
    struct PredictResult {
        double fly_time;
        int fin_target_select;
        std::vector<Eigen::Vector4d> fin_armors_xyza;
    };

    PredictResult
    predictAndSelect(Target& target, double bullet_speed, const AutoAimFsm& auto_aim_fsm) {
        const int roughly_select = selectArmor(target, auto_aim_fsm);

        const auto now = wust_vl::common::utils::time_utils::now();
        target.predictSimple(now);

        const auto ap = target.getArmorPositions();
        double prev_fly_time = trajectory_compensator_config_->trajectory_compensator
                                   ->getFlyingTime(ap[roughly_select].head<3>(), bullet_speed);

        std::vector<Target> iteration_target(10, target);

        for (int iter = 0; iter < 10; ++iter) {
            iteration_target[iter].predictSimple(prev_fly_time);

            const int iter_select = selectArmor(iteration_target[iter], auto_aim_fsm);

            const auto iter_poss = iteration_target[iter].getArmorPositions();

            const double iter_fly_time =
                trajectory_compensator_config_->trajectory_compensator->getFlyingTime(
                    iter_poss[roughly_select],
                    bullet_speed
                );

            if (std::abs(iter_fly_time - prev_fly_time) < 1e-3)
                break;

            prev_fly_time = iter_fly_time;
        }

        const double predict_time = prev_fly_time + config_->prediction_delay_param.get();

        target.predictSimple(predict_time);

        PredictResult res;
        res.fly_time = prev_fly_time;
        res.fin_armors_xyza = target.getArmorPosAndYaw();
        res.fin_target_select = selectArmor(target, auto_aim_fsm);
        return res;
    }

    template<typename VeryAimerPtr, typename BuildFn>
    GimbalCmd veryAimImpl(
        Target target,
        double bullet_speed,
        const AutoAimFsm& auto_aim_fsm,
        BuildFn build_fn
    ) {
        GimbalCmd cmd;
        if (!trajectory_compensator_config_->trajectory_compensator) {
            cmd.appera = false;
            return cmd;
        }

        const auto [aim_first, aim_center, aim_pair] = getAimStatus(auto_aim_fsm);

        auto predict = predictAndSelect(target, bullet_speed, auto_aim_fsm);

        VeryAimerPtr build;
        try {
            build = build_fn(target, predict.fin_target_select, bullet_speed, auto_aim_fsm);
        } catch (...) {
            WUST_WARN("very_aimer") << "build failed";
            cmd.appera = false;
            return cmd;
        }

        cmd.aim_target = build->aim_target;

        double target_yaw = build->cp0.yaw;
        double target_pitch = build->cp0.pitch;

        if (aim_center) {
            const double center_yaw =
                std::atan2(target.target_state_.cy(), target.target_state_.cx());

            const double d_angle = angles::shortest_angular_distance(
                center_yaw,
                predict.fin_armors_xyza[predict.fin_target_select][3]
            );

            auto cp = getControlPoint(
                predict.fin_armors_xyza[predict.fin_target_select].head<3>(),
                d_angle,
                bullet_speed
            );

            target_yaw = cp.yaw;
            target_pitch = cp.pitch;
        }

        const int half_horizon = config_->sample_half_horizon;
        const double half_t = build->target_traj.getPrefixTimeAtIdx(half_horizon);

        const auto cs = build->getControlState(half_t);
        // const auto cs = build->getControlState(wait_t);
        const double yaw = angles::normalize_angle(cs.yaw_state.p + build->cp0.yaw);

        cmd.yaw = rad2deg(yaw);
        cmd.v_yaw = rad2deg(cs.yaw_state.v);
        cmd.a_yaw = rad2deg(cs.yaw_state.a);

        cmd.pitch = rad2deg(cs.pitch_state.p);
        cmd.v_pitch = rad2deg(cs.pitch_state.v);
        cmd.a_pitch = rad2deg(cs.pitch_state.a);

        cmd.target_yaw = rad2deg(target_yaw);
        cmd.target_pitch = rad2deg(target_pitch);

        cmd.raw_yaw = cmd.target_yaw;
        cmd.raw_pitch = cmd.target_pitch;

        cmd.distance = build->fin_aim_pos.norm();
        cmd.fly_time = predict.fly_time;

        auto fire = canFireAtTime(build, half_t);
        if (config_->fuck_test_param.get()) {
            double center_yaw = std::atan2(target.target_state_.cy(), target.target_state_.cx());
            Eigen::Vector3d vel = target.target_state_.vel();
            double vx_center = std::cos(center_yaw) * vel.x() + std::sin(center_yaw) * vel.y();
            double vy_center = -std::sin(center_yaw) * vel.x() + std::cos(center_yaw) * vel.y();

            double thresh = config_->fuck_test_thresh_param.get();
            bool no_shoot = (target.target_state_.vyaw() > 0 && vy_center < thresh)
                || (target.target_state_.vyaw() <= 0 && vy_center > -thresh);

            if (no_shoot) {
                fire.fire = false;
                fire.enable_pitch_diff = 0.0;
                fire.enable_yaw_diff = 0.0;
            }
        }

        cmd.enable_yaw_diff = rad2deg(fire.enable_yaw_diff);
        cmd.enable_pitch_diff = rad2deg(fire.enable_pitch_diff);
        cmd.fire_advice = fire.fire;

        cmd.appera = cmd.isValid();
        if (break_this_) {
            cmd.appera = false;
        }

        return cmd;
    }
    GimbalCmd veryAimSeg(Target target, double bullet_speed, const AutoAimFsm& auto_aim_fsm) {
        return veryAimImpl<VeryAimerTrajSeg::Ptr>(
            std::move(target),
            bullet_speed,
            auto_aim_fsm,
            [this](auto& t, int sel, double bs, const AutoAimFsm& fsm) {
                return buildVerAimerTrajSeg(t, sel, bs, fsm);
            }
        );
    }

    GimbalCmd veryAimMpc(Target target, double bullet_speed, const AutoAimFsm& auto_aim_fsm) {
        return veryAimImpl<VeryAimerTrajMpc::Ptr>(
            std::move(target),
            bullet_speed,
            auto_aim_fsm,
            [this](auto& t, int sel, double bs, const AutoAimFsm& fsm) {
                return buildVerAimerTrajMpc(t, sel, bs, fsm);
            }
        );
    }

    GimbalCmd veryAim(Target target, double bullet_speed, const AutoAimFsm& auto_aim_fsm) {
        break_this_ = false;
        if (type_ == Type::Mpc) {
            return veryAimMpc(target, bullet_speed, auto_aim_fsm);
        } else if (type_ == Type::Seg) {
            return veryAimSeg(target, bullet_speed, auto_aim_fsm);
        } else {
            return veryAimSeg(target, bullet_speed, auto_aim_fsm);
        }
    }
    TrajectoryCompensatorConfig::Ptr trajectory_compensator_config_;
    wust_vl::common::utils::Parameter::Ptr auto_aim_config_parameter_;
    VeryAimerConfig::Ptr config_;
    mutable bool break_this_ = false;
};
VeryAimer::VeryAimer(wust_vl::common::utils::Parameter::Ptr auto_aim_config_parameter) {
    _impl = std::make_unique<Impl>(auto_aim_config_parameter);
}
VeryAimer::~VeryAimer() {
    _impl.reset();
}
GimbalCmd VeryAimer::veryAim(Target target, double bullet_speed, const AutoAimFsm& auto_aim_fsm)

{
    return _impl->veryAim(target, bullet_speed, auto_aim_fsm);
}
} // namespace wust_vision::auto_aim