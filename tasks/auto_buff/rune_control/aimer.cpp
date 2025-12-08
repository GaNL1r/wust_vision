#include "aimer.hpp"
namespace rune {
Aimer::Aimer(
    const YAML::Node& config,
    std::shared_ptr<TrajectoryCompensator> trajectory_compensator
) {
    trajectory_compensator_ = trajectory_compensator;
    prediction_delay_ = config["prediction_delay"].as<double>();
    manual_compensator_ = std::make_unique<ManualCompensator>();
    std::vector<OffsetEntry> entries;

    if (config["trajectory_offset"]) {
        for (const auto& node: config["trajectory_offset"]) {
            OffsetEntry e;
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
std::tuple<bool, double, double, double, double> Aimer::checkHit(
    Eigen::Vector4d maybe_hit,
    const double current_yaw,
    const double current_pitch,
    const double bullet_speed,
    bool use_off_fire,
    double gun_yaw_speed
) {
    Eigen::Vector3d target_pos = maybe_hit.head<3>();
    double dx = target_pos.x();
    double dy = target_pos.y();
    double dz = target_pos.z();

    double distance = target_pos.norm();
    double shooting_range_yaw = std::abs(atan2(shooting_range_w_ / 2, distance));
    double shooting_range_pitch = std::abs(atan2(shooting_range_h_ / 2, distance));
    double yaw_factor = 1.0 * std::cos(maybe_hit[3]);
    double pitch_factor = 1.0;

    shooting_range_yaw *= yaw_factor;
    shooting_range_pitch *= pitch_factor;
    shooting_range_yaw = std::max(shooting_range_yaw, 0.5 * M_PI / 180);
    shooting_range_pitch = std::max(shooting_range_pitch, 0.5 * M_PI / 180);

    double target_yaw = angles::normalize_angle(std::atan2(dy, dx));
    double tmp_pitch = std::atan2(dz, std::sqrt(dx * dx + dy * dy));
    double target_pitch = tmp_pitch;

    if (!trajectory_compensator_->compensate(target_pos, target_pitch, bullet_speed))
        return { false,
                 shooting_range_yaw / M_PI * 180.0,
                 shooting_range_pitch / M_PI * 180.0,
                 target_yaw / M_PI * 180.0,
                 target_pitch / M_PI * 180.0 };

    if (use_off_fire) {
        auto offs =
            manual_compensator_->angleHardCorrect(target_pos.head<2>().norm(), target_pos.z());
        target_yaw += offs[1] * M_PI / 180.0;
        target_pitch += offs[0] * M_PI / 180.0;
    }

    double t_f = distance / bullet_speed;
    target_yaw -= gun_yaw_speed * t_f;

    double yaw_diff = std::abs(angles::shortest_angular_distance(current_yaw, target_yaw));
    double pitch_diff = std::abs(angles::shortest_angular_distance(current_pitch, target_pitch));

    if (yaw_diff < shooting_range_yaw && pitch_diff < shooting_range_pitch) {
        return { true,
                 std::abs(shooting_range_yaw) / M_PI * 180.0,
                 std::abs(shooting_range_pitch) / M_PI * 180.0,
                 target_yaw / M_PI * 180.0,
                 target_pitch / M_PI * 180.0 };
    }

    return { false,
             std::abs(shooting_range_yaw) / M_PI * 180.0,
             std::abs(shooting_range_pitch) / M_PI * 180.0,
             target_yaw / M_PI * 180.0,
             target_pitch / M_PI * 180.0 };
}
GimbalCmd Aimer::aim(
    const RuneTarget& target,
    double bullet_speed,
    std::chrono::steady_clock::time_point time
) {
    GimbalCmd cmd;
    RuneTarget tmp_target = target;

    double dt0 = time_utils::durationSec(target.timestamp_, time) + prediction_delay_;
    std::chrono::steady_clock::time_point future = time + std::chrono::microseconds(int(dt0 * 1e6));
    tmp_target.predictWithFitter(future);
    auto [p0, q0] = tmp_target.getHitPoint();
    bool converged = false;
    double prev_fly_time = trajectory_compensator_->getFlyingTime(p0, bullet_speed);
    auto pre_time0 = future + std::chrono::microseconds(static_cast<int>(prev_fly_time * 1e6));
    tmp_target.predict(pre_time0);
    std::vector<RuneTarget> iteration_target(10, tmp_target);
    Eigen::Vector3d pbest;
    for (int iter = 0; iter < 10; ++iter) {
        auto predict_time =
            future + std::chrono::microseconds(static_cast<int>(prev_fly_time * 1e6));
        iteration_target[iter].predict(predict_time);
        auto [pb, qb] = iteration_target[iter].getHitPoint();
        pbest = pb;
        double iter_fly_time = trajectory_compensator_->getFlyingTime(pb, bullet_speed);
        if (std::abs(iter_fly_time - prev_fly_time) < 0.001) {
            converged = true;
            break;
        }
        prev_fly_time = iter_fly_time;
    }
    cmd.distance = pbest.norm();
    double raw_yaw = angles::normalize_angle(std::atan2(pbest.y(), pbest.x()));
    double raw_pitch =
        std::atan2(pbest.z(), std::sqrt(pbest.x() * pbest.x() + pbest.y() * pbest.y()));
    trajectory_compensator_->compensate(pbest, raw_pitch, bullet_speed);
    AimTarget aim_target;
    aim_target.pos = pbest;
    cmd.aim_target = aim_target;
    cmd.yaw = raw_yaw * 180.0 / M_PI;
    cmd.pitch = raw_pitch * 180.0 / M_PI;
    auto offs = manual_compensator_->angleHardCorrect(
        cmd.aim_target.pos.head(2).norm(),
        cmd.aim_target.pos.z()
    );
    cmd.yaw = angles::normalize_angle((cmd.yaw + offs[1]) * M_PI / 180.0) * 180.0 / M_PI;
    cmd.pitch = cmd.pitch + offs[0];
    auto check = checkHit(
        Eigen::Vector4d(pbest.x(), pbest.y(), pbest.z(), 0),
        cmd.yaw * M_PI / 180.0,
        cmd.pitch * M_PI / 180.0,
        bullet_speed,
        true,
        0
    );
    cmd.fire_advice = true;
    cmd.enable_yaw_diff = std::get<1>(check);
    cmd.enable_pitch_diff = std::get<2>(check);
    cmd.target_yaw = std::get<3>(check);
    cmd.target_pitch = std::get<4>(check);
    return cmd;
}
} // namespace rune
