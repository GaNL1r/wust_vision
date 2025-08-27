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
#include "3rdparty/angles.h"
#include "tasks/auto_aim/armor_tracker/motion_models/acc_model.hpp"
#include "tasks/auto_aim/armor_tracker/one_ca_tracker.hpp"
#include "tasks/auto_aim/armor_tracker/one_tracker.hpp"
#include "tasks/auto_aim/armor_tracker/tracker.hpp"
#include "tasks/auto_aim/type.hpp"
#include "wust_vl/common/utils/config_binder.hpp"
#include "yaml-cpp/yaml.h"
class TrackerManager {
public:
    explicit TrackerManager(const YAML::Node& config, std::shared_ptr<ConfigBinder> config_binder);
    void update(
        armor::Target& target_,
        const armor::Armors& armors_,
        const std::chrono::steady_clock::time_point& time,
        AutoAimFsmController& auto_aim_fsm_cl
    );
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

    double ys2qx_a_, ys2qy_a_, ys2qz_a_, ys2qyaw_a_, ys2qr_a_, ys2qd_zc_a_;
    double yr_y_a_, yr_p_a_, yr_d_front_a_, yr_d_side_a_, yr_yaw_front_a_, yr_yaw_side_a_;
    double ys2qx_c_, ys2qy_c_, ys2qz_c_, ys2qyaw_c_, ys2qr_c_, ys2qd_zc_c_;
    double yr_y_c_, yr_p_c_, yr_d_front_c_, yr_d_side_c_, yr_yaw_front_c_, yr_yaw_side_c_;
    double oys2qx_, oys2qy_, oys2qz_, oys2qyaw_;
    double oyr_y_, oyr_p_, oyr_d_front_, oyr_d_side_, oyr_yaw_front_, oyr_yaw_side_;
    double r_v_, q_a_, q_v_;
    double lost_time_thres_;
    double one_lost_time_thres_;
    std::chrono::steady_clock::time_point last_time_;
    int iteration_num_ = 1;
    std::shared_ptr<ConfigBinder> config_binder_;
    AutoAimFsm auto_aim_fsm_ = AutoAimFsm::AIM_WHOLE_CAR_ARMOR;
};
