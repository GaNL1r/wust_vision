#pragma once
#include "tasks/auto_aim/type.hpp"
#include <wust_vl/common/utils/trajectory_compensator.hpp>
#include <yaml-cpp/yaml.h>

class Shooter {
public:
    Shooter();
    ~Shooter();
    void
    init(const YAML::Node& config, std::shared_ptr<TrajectoryCompensator> trajectory_compensator);
    GimbalCmd shoot(
        const GimbalCmd& cmd,
        const double current_yaw,
        const double current_pitch,
        const double bullet_speed,
        bool use_off_fire,
        double gun_yaw_speed = 0.0
    );
    GimbalCmd shoot(const GimbalCmd& cmd, const double bullet_speed);

private:
    struct Impl;
    std::unique_ptr<Impl> _impl;
};