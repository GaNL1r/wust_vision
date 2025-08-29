#include "shooter.hpp"
#include "3rdparty/angles.h"
#include <wust_vl/common/utils/manual_compensator.hpp>
struct Shooter::Impl {
    void init(const YAML::Node& config) {
        small_shooting_range_w_ = config["shooter"]["small_shooting_range_w"].as<double>(0.12);
        small_shooting_range_h_ = config["shooter"]["small_shooting_range_h"].as<double>(0.12);
        big_shooting_range_w_ = config["shooter"]["big_shooting_range_w"].as<double>(0.12);
        big_shooting_range_h_ = config["shooter"]["big_shooting_range_h"].as<double>(0.12);
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
    GimbalCmd calYawPitch(const AimTarget& aim_target, const bool shoot_center) {
        if (shoot_center) {
            return calShootCenter(aim_target, 0, 0, 0);
        } else {
            return calShootArmor(aim_target, 0, 0, 0);
        }
    }
    GimbalCmd shoot(
        const AimTarget& aim_target,
        const double current_yaw,
        const double current_pitch,
        const double controller_delay,
        const bool shoot_center
    ) {
        if (shoot_center) {
            return calShootCenter(aim_target, current_yaw, current_pitch, controller_delay);
        } else {
            return calShootArmor(aim_target, current_yaw, current_pitch, controller_delay);
        }
    }
    GimbalCmd calShootArmor(
        const AimTarget& aim_target,
        const double current_yaw,
        const double current_pitch,
        const double controller_delay
    ) {
        GimbalCmd cmd;
        AimTarget tmp_aim_target = aim_target;
        tmp_aim_target.predictSelf(controller_delay);
        bool fire_advice = false;
        double fake_yaw = 0.0, fake_pitch = 0.0;
        double distance = tmp_aim_target.distance();
        double fire_yaw = 0.0, fire_pitch = 0.0;
        double control_yaw = 0.0, control_pitch = 0.0;
        double v_yaw = 0.0, v_pitch = 0.0;
        std::tie(fire_yaw, fire_pitch) = manual_compensator_->applyManualCompensator(
            tmp_aim_target.distance(),
            tmp_aim_target.pos[2],
            tmp_aim_target.calYaw(),
            tmp_aim_target.shoot_pitch
        );
        std::tie(control_yaw, control_pitch) = manual_compensator_->applyManualCompensator(
            aim_target.distance(),
            aim_target.pos[2],
            aim_target.calYaw(),
            aim_target.shoot_pitch
        );
        fire_advice = aim_target.is_old ? false
                                        : isOnTarget(
                                            current_yaw,
                                            current_pitch,
                                            fire_yaw,
                                            fire_pitch,
                                            distance,
                                            aim_target.is_big_armor
                                        );
        v_yaw = tmp_aim_target.calVYaw();
        v_pitch = tmp_aim_target.calVPitch();
        cmd.fire_advice = fire_advice;
        cmd.timestamp = std::chrono::steady_clock::now();
        cmd.distance = distance;
        cmd.yaw = angles::normalize_angle(control_yaw) * 180.0 / M_PI;
        cmd.pitch = control_pitch * 180.0 / M_PI;
        cmd.yaw_diff = angles::normalize_angle(control_yaw - current_yaw) * 180.0 / M_PI;
        cmd.pitch_diff = (control_pitch - current_pitch) * 180.0 / M_PI;
        cmd.v_yaw = v_yaw * 180.0 / M_PI;
        cmd.v_pitch = v_pitch * 180.0 / M_PI;
        last_cmd_ = cmd;
        return cmd;
    }
    GimbalCmd calShootCenter(
        const const AimTarget& aim_target,
        const double current_yaw,
        const double current_pitch,
        const double controller_delay
    ) {
        GimbalCmd cmd;
        AimTarget tmp_aim_target = aim_target;
        tmp_aim_target.predictSelf(controller_delay);
        bool fire_advice = false;
        double fake_yaw = 0.0, fake_pitch = 0.0;
        double distance = tmp_aim_target.distance();
        double fire_yaw = 0.0, fire_pitch = 0.0;
        double control_yaw = 0.0, control_pitch = 0.0;
        double v_yaw = 0.0, v_pitch = 0.0;
        auto host_pos = aim_target.host_pos + aim_target.host_vel * controller_delay;
        control_yaw = std::atan2(host_pos.y(), host_pos.x());
        std::tie(fire_yaw, fire_pitch) = manual_compensator_->applyManualCompensator(
            tmp_aim_target.distance(),
            tmp_aim_target.pos[2],
            tmp_aim_target.calYaw(),
            tmp_aim_target.shoot_pitch
        );
        std::tie(fake_yaw, control_pitch) = manual_compensator_->applyManualCompensator(
            aim_target.distance(),
            aim_target.pos[2],
            aim_target.calYaw(),
            aim_target.shoot_pitch
        );
        fire_advice = aim_target.is_old ? false
                                        : isOnTarget(
                                            current_yaw,
                                            current_pitch,
                                            fire_yaw,
                                            fire_pitch,
                                            distance,
                                            aim_target.is_big_armor
                                        );
        std::tie(control_yaw, fake_pitch) = manual_compensator_->applyManualCompensator(
            distance,
            tmp_aim_target.pos[2],
            control_yaw,
            tmp_aim_target.shoot_pitch
        );

        v_yaw = tmp_aim_target.calHostVYaw();
        v_pitch = tmp_aim_target.calVPitch();
        cmd.fire_advice = fire_advice;
        cmd.timestamp = std::chrono::steady_clock::now();
        cmd.distance = distance;
        cmd.yaw = angles::normalize_angle(control_yaw) * 180.0 / M_PI;
        cmd.pitch = control_pitch * 180.0 / M_PI;
        cmd.yaw_diff = angles::normalize_angle(control_yaw - current_yaw) * 180.0 / M_PI;
        cmd.pitch_diff = (control_pitch - current_pitch) * 180.0 / M_PI;
        cmd.v_yaw = v_yaw * 180.0 / M_PI;
        cmd.v_pitch = v_pitch * 180.0 / M_PI;
        last_cmd_ = cmd;
        return cmd;
    }
    bool isOnTarget(
        const double cur_yaw,
        const double cur_pitch,
        const double target_yaw,
        const double target_pitch,
        const double distance,
        const bool is_large_armor
    ) const noexcept {
        double yaw_diff =
            angles::shortest_angular_distance(std::abs(target_yaw), std::abs(cur_yaw));

        double shooting_range_yaw, shooting_range_pitch;
        if (is_large_armor) {
            double dynamic_w = big_shooting_range_w_;
            shooting_range_yaw = std::abs(std::atan2(big_shooting_range_w_ / 2.0, distance));
            shooting_range_pitch = std::abs(std::atan2(big_shooting_range_h_ / 2.0, distance));
        } else {
            double dynamic_w = small_shooting_range_w_;
            shooting_range_yaw = std::abs(std::atan2(small_shooting_range_w_ / 2.0, distance));
            shooting_range_pitch = std::abs(std::atan2(small_shooting_range_h_ / 2.0, distance));
        }

        constexpr double min_angle_rad = 1.0 * M_PI / 180.0;
        shooting_range_yaw = std::max(shooting_range_yaw, min_angle_rad);
        shooting_range_pitch = std::max(shooting_range_pitch, min_angle_rad);
        if (std::abs(yaw_diff) < shooting_range_yaw
            && std::abs(cur_pitch - target_pitch) < shooting_range_pitch)
        {
            return true;
        }

        return false;
    }
    GimbalCmd returnDefaultCmd() {
        last_cmd_.fire_advice = false;
        return last_cmd_;
    }
    std::unique_ptr<ManualCompensator> manual_compensator_;
    double small_shooting_range_w_ = 0.135;
    double small_shooting_range_h_ = 0.135;
    double big_shooting_range_w_ = 0.135;
    double big_shooting_range_h_ = 0.135;
    GimbalCmd last_cmd_;
};
Shooter::Shooter(): _impl(std::make_unique<Impl>()) {}
Shooter::~Shooter() {
    _impl.reset();
}
void Shooter::init(const YAML::Node& config) {
    _impl->init(config);
}
GimbalCmd Shooter::shoot(
    const AimTarget& aim_target,
    const double current_yaw,
    const double current_pitch,
    const double controller_delay,
    const bool shoot_center
) {
    return _impl->shoot(aim_target, current_yaw, current_pitch, controller_delay, shoot_center);
}
GimbalCmd Shooter::returnDefaultCmd() {
    return _impl->returnDefaultCmd();
}
GimbalCmd Shooter::calYawPitch(const AimTarget& aim_target, const bool shoot_center) {
    return _impl->calYawPitch(aim_target, shoot_center);
}