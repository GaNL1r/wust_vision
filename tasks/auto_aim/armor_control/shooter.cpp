#include "shooter.hpp"
#include "3rdparty/angles.h"
#include <wust_vl/common/utils/manual_compensator.hpp>
struct Shooter::Impl {
    void
    init(const YAML::Node& config, std::shared_ptr<TrajectoryCompensator> trajectory_compensator) {
        trajectory_compensator_ = trajectory_compensator;
        shooting_range_w_ = config["shooter"]["shooting_range_w"].as<double>(0.12);
        shooting_range_h_ = config["shooter"]["shooting_range_h"].as<double>(0.12);
        manual_compensator_ = std::make_unique<ManualCompensator>();
        std::vector<OffsetEntry> entries;

        if (config["shooter"]["trajectory_offset"]) {
            for (const auto& node: config["shooter"]["trajectory_offset"]) {
                OffsetEntry e;
                e.d_min = node["d_min"].as<double>();
                e.d_max = node["d_max"].as<double>();
                e.h_min = node["h_min"].as<double>();
                e.h_max = node["h_max"].as<double>();
                e.pitch_off = node["pitch_off"].as<double>();
                e.yaw_off = node["yaw_off"].as<double>();
                entries.push_back(e);
            }
        }
        manual_compensator_->updateMapFlow(entries);
        last_cmd_.yaw = 0;
    }
    GimbalCmd shoot(
        const GimbalCmd& cmd,
        const double current_yaw,
        const double current_pitch,
        const double bullet_speed,
        bool use_off_fire,
        double gun_yaw_speed
    ) {
        GimbalCmd gimbal_cmd = cmd;
        if (!cmd.appera) {
            gimbal_cmd.appera = false;
            return gimbal_cmd;
        }
        std::vector<Eigen::Vector4d> raw_xyza_list = cmd.armor_posandyaw;
        std::vector<Eigen::Vector4d> maybe_hit_list;
        for (int i = 0; i < raw_xyza_list.size(); i++) {
            double d_angle = angles::shortest_angular_distance(current_yaw, raw_xyza_list[i][3]);
            if (std::abs(d_angle) < M_PI / 2.0) {
                raw_xyza_list[i][3] = std::abs(d_angle);
                maybe_hit_list.emplace_back(raw_xyza_list[i]);
            }
        }
        bool fire = false;
        for (auto maybe_hit: maybe_hit_list) {
            auto check = checkHit(
                maybe_hit,
                current_yaw,
                current_pitch,
                bullet_speed,
                use_off_fire,
                gun_yaw_speed
            );
            gimbal_cmd.enable_yaw_diff = std::get<1>(check);
            gimbal_cmd.enable_pitch_diff = std::get<2>(check);
            gimbal_cmd.target_yaw = std::get<3>(check);
            gimbal_cmd.target_pitch = std::get<4>(check);
            if (std::get<0>(check)) {
                fire = true;
                break;
            }
        }

        auto offs = manual_compensator_->angleHardCorrect(
            cmd.aim_target.pos.head(2).norm(),
            cmd.aim_target.pos.z()
        );
        gimbal_cmd.yaw = angles::normalize_angle((cmd.yaw + offs[1]) * M_PI / 180.0) * 180.0 / M_PI;
        gimbal_cmd.pitch = cmd.pitch + offs[0];

        gimbal_cmd.fire_advice = fire;
        return gimbal_cmd;
    }
    std::tuple<bool, double, double, double, double> checkHit(
        Eigen::Vector4d maybe_hit,
        const double current_yaw,
        const double current_pitch,
        const double bullet_speed,
        bool use_off_fire,
        double gun_yaw_speed = 0.0
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
        double pitch_diff =
            std::abs(angles::shortest_angular_distance(current_pitch, target_pitch));

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

    GimbalCmd returnDefaultCmd() {
        last_cmd_.fire_advice = false;
        last_cmd_.appera = false;
        return last_cmd_;
    }
    std::unique_ptr<ManualCompensator> manual_compensator_;
    double shooting_range_w_ = 0.135;
    double shooting_range_h_ = 0.135;
    GimbalCmd last_cmd_;
    std::shared_ptr<TrajectoryCompensator> trajectory_compensator_;
};
Shooter::Shooter(): _impl(std::make_unique<Impl>()) {}
Shooter::~Shooter() {
    _impl.reset();
}
void Shooter::init(
    const YAML::Node& config,
    std::shared_ptr<TrajectoryCompensator> trajectory_compensator
) {
    _impl->init(config, trajectory_compensator);
}
GimbalCmd Shooter::shoot(
    const GimbalCmd& cmd,
    const double current_yaw,
    const double current_pitch,
    const double bullet_speed,
    bool use_off_fire,
    double gun_yaw_speed
) {
    return _impl->shoot(cmd, current_yaw, current_pitch, bullet_speed, use_off_fire, gun_yaw_speed);
}

GimbalCmd Shooter::shoot(const GimbalCmd& cmd, const double bullet_speed) {
    return _impl->shoot(cmd, 0, 0, bullet_speed, false, 0.0);
}