// Maintained by Chengfu Zou, Labor
// Copyright (C) FYT Vision Group. All rights reserved.
// Copyright 2025 Xiaojian Wu
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#pragma once

// std
#include <array> // std::array :contentReference[oaicite:1]{index=1}
#include <memory> // std::unique_ptr, std::shared_ptr :contentReference[oaicite:0]{index=0}
#include <string>
#include <utility> // std::pair :contentReference[oaicite:3]{index=3}
#include <vector> // std::vector :contentReference[oaicite:2]{index=2}

// 3rd-party
#include <Eigen/Dense> // Eigen::Vector3d :contentReference[oaicite:4]{index=4}
#include <yaml-cpp/node/node.h>

// project
#include "common/gobal.hpp"
#include "common/tf.hpp" // Transform, TfTree, eulerToQuaternion
#include "control/manual_compensator.hpp"
#include "control/trajectory_compensator.hpp"
#include "type/type.hpp" // Position, Target, GimbalCmd

// Normalize to (–π, +π]
inline double normalize_angle(double a) noexcept {
    while (a > M_PI)
        a -= 2 * M_PI;
    while (a <= -M_PI)
        a += 2 * M_PI;
    return a;
}

class ArmorSolver {
public:
    ArmorSolver() = default;
    ArmorSolver(const YAML::Node& config);

    ~ArmorSolver() = default;

    // Solve a new gimbal command; timestamp in seconds
    GimbalCmd solve(const Target& target, std::chrono::steady_clock::time_point current_time);
    GimbalCmd solve(
        const Target& target,
        std::vector<OneTarget> one_targets_,
        std::chrono::steady_clock::time_point current_time
    );
    std::vector<GimbalCmd> solve_vector(
        const Target& target,
        std::vector<OneTarget> one_targets_,
        std::chrono::steady_clock::time_point current_time,
        int step,
        double dt
    );

    // Retrieve a compensated trajectory (same API as before)
    std::vector<std::pair<double, double>> getTrajectory() const noexcept;

    enum State { TRACKING_ARMOR = 0, TRACKING_CENTER = 1 };

private:
    void init(const YAML::Node& config);
    // Core helper methods (same logic as original)
    bool isOnTarget(
        double cur_yaw,
        double cur_pitch,
        double target_yaw,
        double target_pitch,
        double distance,
        bool is_large_armor
    ) const noexcept;
    void calcYawAndPitch(
        const Eigen::Vector3d& p,
        const std::array<double, 3>& rpy,
        double& yaw,
        double& pitch
    ) const noexcept;
    void calcVYawAndVPitch(
        const Eigen::Vector3d& v,
        const std::array<double, 3>& rpy,
        double& vyaw,
        double& vpitch
    ) const noexcept;

    std::vector<Eigen::Vector3d> getArmorPositions(
        const Eigen::Vector3d& center,
        double yaw,
        double r1,
        double r2,
        double d_zc,
        double d_za,
        size_t num
    ) const noexcept;
    std::vector<Eigen::Vector3d> getArmorVelocities(
        const Eigen::Vector3d& target_center,
        const double target_yaw,
        const Eigen::Vector3d& target_velocity,
        const double target_yaw_rate,
        const double r1,
        const double r2,
        const double d_zc,
        const double d_za,
        const size_t armors_num
    ) const noexcept;
    int selectBestArmor(
        const std::vector<Eigen::Vector3d>& armors,
        const Eigen::Vector3d& center,
        double yaw,
        double v_yaw,
        size_t num
    ) const noexcept;
    int selectBestTarget(const std::vector<OneTarget>& targets) const noexcept;

    double small_shooting_range_w = 0.135;
    double small_shooting_range_h = 0.135;
    double big_shooting_range_w = 0.135;
    double big_shooting_range_h = 0.135;
    double max_tracking_v_yaw = 6.0;
    double prediction_delay = 0.0;
    double side_angle = 15.0; // degrees
    double min_switching_v_yaw = 1.0;
    int transfer_thresh = 5;
    int iteration_times = 20;
    double bullet_speed = 20.0;
    double gravity = 9.8;
    double resistance = 0.001;
    std::vector<std::pair<double, double>> manual_angle_offset; // distance→{pitch°, yaw°}

    std::unique_ptr<TrajectoryCompensator> trajectory_compensator_;
    std::unique_ptr<ManualCompensator> manual_compensator_;
    State state_ = TRACKING_ARMOR;
    int overflow_count_ = 0;
    std::array<double, 3> rpy_ { 0, 0, 0 }; // roll, pitch, yaw
    std::string solver_logger = "armor_solver";
    mutable OneTarget last_target_;
    mutable bool has_last_target_ = false;

    mutable OneTarget pending_target_;
    mutable std::chrono::steady_clock::time_point pending_target_start_time_;
    mutable bool has_pending_target_ = false;

    double oneswitch_position_thres_ = 0.3;
    double oneswitch_angle_thres_ = 30.0;
    double oneswitch_hold_time_ = 1.5;
};
