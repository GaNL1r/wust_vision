#pragma once
#include "tasks/auto_aim/armor_tracker/tracker_manager.hpp"

class Aimer {
public:
    Aimer();
    ~Aimer();
    void init(const YAML::Node& config);
    AimTarget aimTarget(
        const Target& armor_slover_target,
        std::chrono::steady_clock::time_point time,
        const double bullet_speed,
        const AutoAimFsm& auto_aim_fsm
    );

private:
    struct Impl;
    std::unique_ptr<Impl> _impl;
};
