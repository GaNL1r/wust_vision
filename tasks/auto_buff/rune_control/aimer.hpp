#pragma once
#include "tasks/auto_buff/rune_tracker/rune_target.hpp"
#include "tasks/type_common.hpp"
#include <wust_vl/common/utils/trajectory_compensator.hpp>
namespace rune {
class Aimer {
public:
    Aimer(const YAML::Node& config, std::shared_ptr<TrajectoryCompensator> trajectory_compensator);
    GimbalCmd
    aim(const rune::RuneTarget& target,
        double bullet_speed,
        std::chrono::steady_clock::time_point time);
    std::tuple<bool, double, double, double, double> checkHit(
        Eigen::Vector4d maybe_hit,
        const double current_yaw,
        const double current_pitch,
        const double bullet_speed,
        bool use_off_fire,
        double gun_yaw_speed = 0.0
    );
    double getFlyingTime(const Eigen::Vector3d& target_position, const double bullet_speed) {
        return trajectory_compensator_->getFlyingTime(target_position, bullet_speed);
    }
    std::shared_ptr<TrajectoryCompensator> trajectory_compensator_;
    std::unique_ptr<ManualCompensator> manual_compensator_;
    double prediction_delay_ = 0.0;
    double shooting_range_w_ = 0.2;
    double shooting_range_h_ = 0.2;
};

} // namespace rune