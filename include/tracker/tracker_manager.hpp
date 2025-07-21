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
#include "common/3rdparty/angles.h"
#include "tracker/motion_models/acc_model.hpp"
#include "tracker/one_ca_tracker.hpp"
#include "tracker/one_tracker.hpp"
#include "tracker/tracker.hpp"
#include "type/type.hpp"
#include "yaml-cpp/yaml.h"
class TrackerManager {
public:
    explicit TrackerManager(const YAML::Node& config);
    void update(
        armor::Target& target_,
        std::vector<armor::OneTarget>& one_targets_,
        armor::Armors armors_,
        std::chrono::steady_clock::time_point time,
        const Eigen::Matrix3d& R_gimbal2odom,
        const Eigen::Vector3d& v
    );
    void updateAttackState(const double& v_yaw_abs);
    void updateTracker(
        armor::Target& target_,
        armor::Armors armors_,
        std::chrono::steady_clock::time_point time,
        const Eigen::Vector3d& v
    );
    void updateOneTrackers(
        std::vector<armor::OneTarget>& one_targets_,
        armor::Armors armors_,
        std::chrono::steady_clock::time_point time,
        const Eigen::Vector3d& v
    );

    enum State {
        LOST,
        DETECTING,
        TRACKING,
        TEMP_LOST,
    } tracker_state;
    Eigen::Matrix3d R_gimbal2odom_;

    bool use_ypd_tracker_ = false;
    double dt_;
    std::unique_ptr<Tracker> tracker_;
    int track_one_num_;
    std::vector<std::unique_ptr<OneTracker>> one_trackers_;

    double ys2qx_, ys2qy_, ys2qz_, ys2qyaw_, ys2qr_, ys2qd_zc_;
    double oys2qx_, oys2qy_, oys2qz_, oys2qyaw_;
    double yr_y_, yr_p_, yr_d_front_, yr_d_side_, yr_yaw_front_, yr_yaw_side_;
    double oyr_y_, oyr_p_, oyr_d_front_, oyr_d_side_, oyr_yaw_front_, oyr_yaw_side_;
    double r_v, q_a, q_v;
    double lost_time_thres_;
    double one_lost_time_thres_;
    std::chrono::steady_clock::time_point last_time_;
    double v_yaw_to_one_thres_high_;
    double v_yaw_to_one_thres_low_;
    int iteration_num_ = 1;
};
