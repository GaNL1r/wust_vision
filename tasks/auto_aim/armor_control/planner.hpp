#pragma once
#include "aimer.hpp"
#include "shooter.hpp"
#include "tasks/auto_aim/armor_tracker/target.hpp"
#include "tinympc/tiny_api.hpp"
#include <Eigen/Dense>
constexpr int MPC_HORIZON = 300;
constexpr double MPC_DT = 1.0 / MPC_HORIZON;
constexpr int MPC_HALF_HORIZON = MPC_HORIZON / 2;
constexpr int CAL_HORIZON = 100;
constexpr double CAL_DT = 1.0 / CAL_HORIZON;
constexpr int CAL_HALF_HORIZON = CAL_HORIZON / 2;

using Trajectory = Eigen::Matrix<double, 4, MPC_HORIZON>; // yaw, yaw_vel, pitch, pitch_vel

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
    double distance;
    std::vector<Eigen::Vector4d> armor_posandyaw;
    AimTarget aim_target;
};

class Planner {
public:
    Eigen::Vector4d debug_xyza;
    Planner(
        const YAML::Node& config,
        std::shared_ptr<Aimer> aimer,
        std::shared_ptr<Shooter> shooter
    );

    Plan plan(
        Target target,
        double bullet_speed,
        const AutoAimFsm& auto_aim_fsm,
        double self_v_yaw = 0.0
    );

private:
    double yaw_offset_;
    double pitch_offset_;
    double fire_thresh_;
    double low_speed_delay_time_, high_speed_delay_time_, decision_speed_;
    double max_yaw_acc_, max_pitch_acc_;
    TinySolver* yaw_solver_;
    TinySolver* pitch_solver_;

    void setup_yaw_solver(const YAML::Node& config);
    void setup_pitch_solver(const YAML::Node& config);
    Eigen::Matrix<double, 2, 1>
    aim(const Target& target,
        double bullet_speed,
        bool aim_center,
        bool aim_first,
        double self_v_yaw,
        double dt);
    std::tuple<double, double, std::vector<Eigen::Vector4d>, AimTarget> calaim(
        const Target& target,
        double bullet_speed,
        bool aim_center,
        bool aim_first,
        double self_v_yaw,
        double dt
    );
    std::pair<Eigen::Vector3d, double> selectArmor(const Target& target, bool aim_first);
    Trajectory get_trajectory(
        Target& target,
        double yaw0,
        double bullet_speed,
        bool aim_center,
        bool aim_first,
        double self_v_yaw,
        double dt
    );
    std::shared_ptr<Aimer> aimer_;
    std::shared_ptr<Shooter> shooter_;
    int max_iter_;

};