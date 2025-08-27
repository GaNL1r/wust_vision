#pragma once
#include "tasks/auto_aim/type.hpp"
#include <yaml-cpp/yaml.h>

class Shooter {
public:
    Shooter();
    ~Shooter();
    void init(const YAML::Node& config);
    GimbalCmd shoot(
        const AimTarget& aim_target,
        const double current_yaw,
        const double current_pitch,
        const double controller_delay,
        const bool shoot_center
    );
    GimbalCmd returnDefaultCmd();

private:
    struct Impl;
    std::unique_ptr<Impl> _impl;
};