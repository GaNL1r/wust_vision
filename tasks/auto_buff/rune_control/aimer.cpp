#include "aimer.hpp"
#include "wust_vl/common/utils/manual_compensator.hpp"
namespace wust_vision {

namespace auto_buff {
    struct Aimer::Impl {
    public:
        Impl(
            const YAML::Node& config,
            std::shared_ptr<wust_vl::common::utils::TrajectoryCompensator> trajectory_compensator
        ) {
            trajectory_compensator_ = trajectory_compensator;
            prediction_delay_ = config["prediction_delay"].as<double>();
            shooting_range_w_ = config["shooting_range_w"].as<double>(0.12);
            shooting_range_h_ = config["shooting_range_h"].as<double>(0.12);
            min_enable_pitch_deg_ = config["min_enable_pitch_deg"].as<double>(0.0);
            min_enable_yaw_deg_ = config["min_enable_yaw_deg"].as<double>(0.0);
            manual_compensator_ = std::make_unique<wust_vl::common::utils::ManualCompensator>();
            std::vector<wust_vl::common::utils::OffsetEntry> entries;

            if (config["trajectory_offset"]) {
                for (const auto& node: config["trajectory_offset"]) {
                    wust_vl::common::utils::OffsetEntry e;
                    e.d_min = node["d_min"].as<double>();
                    e.d_max = node["d_max"].as<double>();
                    e.h_min = node["h_min"].as<double>();
                    e.h_max = node["h_max"].as<double>();
                    e.pitch_off = node["pitch_off"].as<double>();
                    e.yaw_off = node["yaw_off"].as<double>();
                    entries.push_back(e);
                }
                manual_compensator_->setBasePitch(config["base_offset"]["pitch"].as<double>());
                manual_compensator_->setBaseYaw(config["base_offset"]["yaw"].as<double>());
            }
            manual_compensator_->updateMapFlow(entries);
        }
        std::tuple<double, double> calEnableDiff(Eigen::Vector3d aim_target_pos) const noexcept {
            const double distance = aim_target_pos.norm();
            double shooting_range_yaw = std::abs(atan2(shooting_range_w_ / 2, distance));
            double shooting_range_pitch = std::abs(atan2(shooting_range_h_ / 2, distance));
            constexpr double yaw_factor = 1.0;
            constexpr double pitch_factor = 1.0;
            shooting_range_yaw = std::max(shooting_range_yaw, min_enable_yaw_deg_ * M_PI / 180);
            shooting_range_pitch =
                std::max(shooting_range_pitch, min_enable_pitch_deg_ * M_PI / 180);
            shooting_range_yaw *= yaw_factor;
            shooting_range_pitch *= pitch_factor;

            return std::make_tuple(std::abs(shooting_range_yaw), std::abs(shooting_range_pitch));
        }
        struct ControlPoint {
            double yaw;
            double pitch;
        };
        ControlPoint
        getControlPoint(Eigen::Vector3d aim_target_pos, double bullet_speed) const noexcept {
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
            return cp;
        }
        GimbalCmd aim(RuneTarget target, double bullet_speed) {
            GimbalCmd cmd;

            const auto now = wust_vl::common::utils::time_utils::now();
            const double dt0 =
                wust_vl::common::utils::time_utils::durationSec(target.timestamp_, now);
            target.predictWithFitter(dt0);
            auto [p0, q0] = target.getHitPoint();
            bool converged = false;
            const double fly_time = trajectory_compensator_->getFlyingTime(p0, bullet_speed);
            double prev_fly_time = fly_time;
            std::vector<RuneTarget> iteration_target(10, target);
            Eigen::Vector3d pbest;
            for (int iter = 0; iter < 10; ++iter) {
                iteration_target[iter].predictWithFitter(prev_fly_time);
                auto [pb, qb] = iteration_target[iter].getHitPoint();
                pbest = pb;
                double iter_fly_time = trajectory_compensator_->getFlyingTime(pb, bullet_speed);
                if (std::abs(iter_fly_time - prev_fly_time) < 0.001) {
                    converged = true;
                    break;
                }
                prev_fly_time = iter_fly_time;
            }
            const double predict_time = prev_fly_time + prediction_delay_;
            target.predictWithFitter(predict_time);
            cmd.distance = pbest.norm();
            const auto cp = getControlPoint(pbest, bullet_speed);
            AimTarget aim_target;
            aim_target.pos = pbest;
            cmd.aim_target = aim_target;
            cmd.yaw = cp.yaw * 180.0 / M_PI;
            cmd.pitch = cp.pitch * 180.0 / M_PI;
            const auto [enable_yaw, enable_pitch] = calEnableDiff(pbest);
            cmd.fire_advice = true;
            cmd.enable_yaw_diff = enable_yaw;
            cmd.enable_pitch_diff = enable_pitch;
            cmd.target_yaw = cp.yaw * 180.0 / M_PI;
            cmd.target_pitch = cp.pitch * 180.0 / M_PI;
            cmd.fly_time = prev_fly_time;
            cmd.appera = true;
            return cmd;
        }
        std::shared_ptr<wust_vl::common::utils::TrajectoryCompensator> trajectory_compensator_;
        std::unique_ptr<wust_vl::common::utils::ManualCompensator> manual_compensator_;
        double prediction_delay_ = 0.0;
        double shooting_range_w_ = 0.2;
        double shooting_range_h_ = 0.2;
        double min_enable_yaw_deg_ = 0.5;
        double min_enable_pitch_deg_ = 0.5;
    };
    Aimer::Aimer(
        const YAML::Node& config,
        std::shared_ptr<wust_vl::common::utils::TrajectoryCompensator> trajectory_compensator
    ) {
        _impl = std::make_unique<Impl>(config, trajectory_compensator);
    }
    Aimer::~Aimer() {
        _impl.reset();
    }
    GimbalCmd Aimer::aim(const auto_buff::RuneTarget& target, double bullet_speed) {
        return _impl->aim(target, bullet_speed);
    }
} // namespace auto_buff
} // namespace wust_vision