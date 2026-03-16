#include "aimer.hpp"
#include "tasks/auto_buff/rune_tracker/rune_target.hpp"
#include "wust_vl/common/utils/manual_compensator.hpp"
namespace wust_vision {

namespace auto_buff {

    struct Aimer::Impl {
    public:
        struct AimerConfig: wust_vl::common::utils::ParamGroup {
            static constexpr const char* Logger = "Config: auto_buff::aimer";
            static constexpr const char* kKey = "aimer";
            const char* key() const override {
                return kKey;
            }
            using Ptr = std::shared_ptr<AimerConfig>;
            AimerConfig() {}
            static Ptr create() {
                return std::make_shared<AimerConfig>();
            }
            std::shared_ptr<wust_vl::common::utils::ManualCompensator> manual_compensator;
            GEN_PARAM(double, prediction_delay);
            GEN_PARAM(double, shooting_range_h);
            GEN_PARAM(double, shooting_range_w);
            GEN_PARAM(double, min_enable_pitch_deg);
            GEN_PARAM(double, min_enable_yaw_deg);
            bool first_load = false;
            using OffsetEntry = wust_vl::common::utils::OffsetEntry;

            static std::vector<OffsetEntry>
            parseTrajectoryOffset(const YAML::Node& node, double& base_pitch, double& base_yaw) {
                std::vector<OffsetEntry> entries;

                if (!node || !node["trajectory_offset"]) {
                    return entries;
                }

                for (const auto& n: node["trajectory_offset"]) {
                    entries.push_back(OffsetEntry { .d_min = n["d_min"].as<double>(),
                                                    .d_max = n["d_max"].as<double>(),
                                                    .h_min = n["h_min"].as<double>(),
                                                    .h_max = n["h_max"].as<double>(),
                                                    .pitch_off = n["pitch_off"].as<double>(),
                                                    .yaw_off = n["yaw_off"].as<double>() });
                }

                base_pitch = node["base_offset"]["pitch"].as<double>();
                base_yaw = node["base_offset"]["yaw"].as<double>();

                return entries;
            }
            std::vector<OffsetEntry> last_entries_;
            double last_base_pitch_ = 0.0;
            double last_base_yaw_ = 0.0;
            bool first_load_ = true;
            void loadSelf(const YAML::Node& node) override {
                double base_pitch = 0.0;
                double base_yaw = 0.0;

                auto entries = parseTrajectoryOffset(node, base_pitch, base_yaw);

                const bool trajectory_changed = first_load_ || entries != last_entries_
                    || base_pitch != last_base_pitch_ || base_yaw != last_base_yaw_;
                if (trajectory_changed) {
                    manual_compensator =
                        std::make_shared<wust_vl::common::utils::ManualCompensator>();

                    manual_compensator->setBasePitch(base_pitch);
                    manual_compensator->setBaseYaw(base_yaw);

                    if (!manual_compensator->updateMapFlow(entries) || entries.empty()) {
                        std::cout << "manual_compensator init failed" << std::endl;
                    }

                    last_entries_ = entries;
                    last_base_pitch_ = base_pitch;
                    last_base_yaw_ = base_yaw;
                    first_load_ = false;
                }
                shooting_range_h_param.load(node);
                shooting_range_w_param.load(node);
                min_enable_pitch_deg_param.load(node);
                min_enable_yaw_deg_param.load(node);
                prediction_delay_param.load(node);
            }
        };
        Impl(wust_vl::common::utils::Parameter::Ptr auto_buff_config_parameter) {
            aimer_config_ = AimerConfig::create();
            trajectory_compensator_config_ = TrajectoryCompensatorConfig::create();
            auto_buff_config_parameter->registerGroup(*aimer_config_);
            auto_buff_config_parameter->registerGroup(*trajectory_compensator_config_);
            auto_buff_config_parameter->reloadFromOldPath();
        }
        std::tuple<double, double> calEnableDiff(Eigen::Vector3d aim_target_pos) const noexcept {
            const double distance = aim_target_pos.norm();
            double shooting_range_yaw =
                std::abs(atan2(aimer_config_->shooting_range_w_param.get() / 2, distance));
            double shooting_range_pitch =
                std::abs(atan2(aimer_config_->shooting_range_h_param.get() / 2, distance));
            constexpr double yaw_factor = 1.0;
            constexpr double pitch_factor = 1.0;
            shooting_range_yaw = std::max(
                shooting_range_yaw,
                aimer_config_->min_enable_yaw_deg_param.get() * M_PI / 180
            );
            shooting_range_pitch = std::max(
                shooting_range_pitch,
                aimer_config_->min_enable_pitch_deg_param.get() * M_PI / 180
            );
            shooting_range_yaw *= yaw_factor;
            shooting_range_pitch *= pitch_factor;

            return std::make_tuple(std::abs(shooting_range_yaw), std::abs(shooting_range_pitch));
        }
        struct ControlPoint {
            double yaw;
            double pitch;
            Eigen::Vector3d aim_pos;
        };
        ControlPoint getControlPoint(RuneTarget target, double bullet_speed) const noexcept {
            auto [aim_target_pos, _] = target.getHitPoint();
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
                trajectory_compensator_config_->trajectory_compensator
                    ->compensate(aim_target_pos, raw_pitch, bullet_speed);
            } catch (std::exception& e) {
                std::cout << "compensate error: " << e.what() << std::endl;
            }

            double control_pitch = raw_pitch;
            const auto offs = aimer_config_->manual_compensator->angleHardCorrect(
                aim_target_pos.head(2).norm(),
                aim_target_pos.z()
            );
            control_yaw = angles::normalize_angle((control_yaw + offs[1] * M_PI / 180.0));
            control_pitch = (control_pitch + offs[0] * M_PI / 180.0);
            cp.pitch = control_pitch;
            cp.yaw = control_yaw;
            cp.aim_pos = aim_target_pos;
            return cp;
        }
        GimbalCmd aim(RuneTarget target, double bullet_speed) {
            GimbalCmd cmd;

            // 当前时间
            const auto now = wust_vl::common::utils::time_utils::now();

            // 首次预测目标位置
            target.predictWithFitter(now);
            auto [p0, q0] = target.getHitPoint();

            // 迭代计算飞行时间
            bool converged = false;
            double prev_fly_time = trajectory_compensator_config_->trajectory_compensator
                                       ->getFlyingTime(p0, bullet_speed);

            std::vector<RuneTarget> iteration_target(10, target);
            for (int iter = 0; iter < 10; ++iter) {
                iteration_target[iter].predictWithFitter(prev_fly_time);
                auto [pb, qb] = iteration_target[iter].getHitPoint();
                double iter_fly_time = trajectory_compensator_config_->trajectory_compensator
                                           ->getFlyingTime(pb, bullet_speed);
                if (std::abs(iter_fly_time - prev_fly_time) < 0.001) {
                    converged = true;
                    break;
                }
                prev_fly_time = iter_fly_time;
            }

            // 预测最终位置，考虑系统延迟
            const double predict_time = prev_fly_time + aimer_config_->prediction_delay_param.get();
            target.predictWithFitter(predict_time);
            const auto cp = getControlPoint(target, bullet_speed);
            RuneTarget target_prev, target_next = target;
            // 离散控制点
            const double dt = 0.01; // 10ms 前后差分
            target_prev.predictWithFitter(-dt);
            auto cp_prev = getControlPoint(target_prev, bullet_speed);

            target_next.predictWithFitter(dt);
            auto cp_next = getControlPoint(target_next, bullet_speed);

            double yaw_speed = (cp_next.yaw - cp_prev.yaw) / (2.0 * dt);
            double pitch_speed = (cp_next.pitch - cp_prev.pitch) / (2.0 * dt);
            double yaw_acc = (cp_next.yaw - 2.0 * cp.yaw + cp_prev.yaw) / (dt * dt);
            double pitch_acc = (cp_next.pitch - 2.0 * cp.pitch + cp_prev.pitch) / (dt * dt);

            AimTarget aim_target;
            aim_target.pos = cp.aim_pos;

            // 填充 GimbalCmd
            cmd.distance = cp.aim_pos.norm();
            cmd.aim_target = aim_target;
            cmd.yaw = cp.yaw * 180.0 / M_PI;
            cmd.pitch = cp.pitch * 180.0 / M_PI;
            cmd.v_yaw = yaw_speed * 180.0 / M_PI; // 转为度/秒
            cmd.v_pitch = pitch_speed * 180.0 / M_PI;
            cmd.a_yaw = yaw_acc * 180.0 / M_PI; // 转为度/秒²
            cmd.a_pitch = pitch_acc * 180.0 / M_PI;
            const auto [enable_yaw, enable_pitch] = calEnableDiff(cp.aim_pos);
            cmd.fire_advice = true;
            cmd.enable_yaw_diff = enable_yaw;
            cmd.enable_pitch_diff = enable_pitch;
            cmd.target_yaw = cp.yaw * 180.0 / M_PI;
            cmd.target_pitch = cp.pitch * 180.0 / M_PI;
            cmd.fly_time = prev_fly_time;
            cmd.appera = true;

            return cmd;
        }
        TrajectoryCompensatorConfig::Ptr trajectory_compensator_config_;
        AimerConfig::Ptr aimer_config_;
    };
    Aimer::Aimer(wust_vl::common::utils::Parameter::Ptr auto_buff_config_parameter) {
        _impl = std::make_unique<Impl>(auto_buff_config_parameter);
    }
    Aimer::~Aimer() {
        _impl.reset();
    }
    GimbalCmd Aimer::aim(const auto_buff::RuneTarget& target, double bullet_speed) {
        return _impl->aim(target, bullet_speed);
    }
} // namespace auto_buff
} // namespace wust_vision