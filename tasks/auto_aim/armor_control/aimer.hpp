#pragma once
#include "tasks/auto_aim/armor_tracker/tracker_manager.hpp"
#include <wust_vl/common/utils/trajectory_compensator.hpp>

class Aimer {
public:
    Aimer();
    ~Aimer();
    void
    init(const YAML::Node& config, std::shared_ptr<TrajectoryCompensator> trajectory_compensator);
    AimTarget aimTarget(
        const Target& armor_slover_target,
        std::chrono::steady_clock::time_point time,
        const double bullet_speed,
        const AutoAimFsm& auto_aim_fsm
    );
    AimTarget aimTargetNoPre(const Target& target, const double bullet_speed, bool aim_first);
    double getFlyingTime(const Eigen::Vector3d& target_position, const double bullet_speed);
    int selectBestArmor(const Target& target, bool aim_first);
    AimTarget aimTargetNoPreWithIdx(const Target& target, const double bullet_speed, int idx);
    bool
    compensate(const Eigen::Vector3d& target_position, double& pitch, const double bullet_speed);
    GimbalCmd
    aim(const Target& target,
        std::chrono::steady_clock::time_point time,
        const double bullet_speed,
        const AutoAimFsm& auto_aim_fsm);
    std::pair<double, double> getCommingLeaving();
    double getPredelay();

private:
    struct Impl;
    std::unique_ptr<Impl> _impl;
};
