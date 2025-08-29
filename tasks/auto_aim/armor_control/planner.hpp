#pragma once
#include <Eigen/Dense>
#include <list>
#include <optional>

#include "aimer.hpp"
#include "shooter.hpp"
#include "tasks/auto_aim/armor_tracker/target.hpp"
#include "tinympc/tiny_api.hpp"

constexpr double DT = 0.01;
constexpr int HALF_HORIZON = 50;
constexpr int HORIZON = HALF_HORIZON * 2;

using Trajectory = Eigen::Matrix<double, 4, HORIZON>; // yaw, yaw_vel, pitch, pitch_vel

struct Plan {
    bool control;
    bool fire;
    float target_yaw;
    float target_pitch;
    float yaw;
    float yaw_vel;
    float yaw_acc;
    float pitch;
    float pitch_vel;
    float pitch_acc;
};

class Planner {
public:
    Eigen::Vector4d debug_xyza;
    Planner(
        const YAML::Node& config,
        std::shared_ptr<Aimer> aimer,
        std::shared_ptr<Shooter> shooter
    );

    Plan plan(Target target, double bullet_speed, const AutoAimFsm& auto_aim_fsm);

private:
    double yaw_offset_;
    double pitch_offset_;
    double fire_thresh_;
    double low_speed_delay_time_, high_speed_delay_time_, decision_speed_;
    bool has_changed_;
    int chenged_idx_;
    TinySolver* yaw_solver_;
    TinySolver* pitch_solver_;

    void setup_yaw_solver(const YAML::Node& config);
    void setup_pitch_solver(const YAML::Node& config);
    Eigen::Matrix<double, 2, 1>
    aim(const Target& target, double bullet_speed, int idx, bool aim_center, bool aim_first);
    Trajectory get_trajectory(
        Target& target,
        double yaw0,
        double bullet_speed,
        int idx,
        bool aim_center,
        bool aim_first
    );
    std::shared_ptr<Aimer> aimer_;
    std::shared_ptr<Shooter> shooter_;
};