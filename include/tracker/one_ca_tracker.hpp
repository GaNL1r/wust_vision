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
#include <memory>
#include <string>

// third party
#include <Eigen/Eigen>
#include <vector>

// project
#include "tracker/extended_kalman_filter.hpp"
#include "tracker/motion_modeloneca.hpp"
#include "type/type.hpp"

inline double onormalizeAnglem(double angle) {
  while (angle > M_PI)
    angle -= 2 * M_PI;
  while (angle < -M_PI)
    angle += 2 * M_PI;
  return angle;
}

class OneCaTracker {
public:
  OneCaTracker(double max_match_distance, double max_match_yaw,
               double max_match_z_diff_, double max_match_x_diff,
               double jump_thresh);

  void init(const Armors &armors_msg) noexcept;
  void update(const Armors &armors_msg) noexcept;
  void init(const Armor &armor_msg) noexcept;
  void update(const Armor &armor_msg) noexcept;

  enum State {
    LOST,
    DETECTING,
    TRACKING,
    TEMP_LOST,
  } tracker_state;
  static Eigen::Vector3d
  getArmorPositionFromState(const Eigen::VectorXd &x) noexcept;

  std::unique_ptr<onecaarmor_motion_model::RobotStateEKF> ekf;

  int tracking_thres;
  int lost_thres;

  Armor tracked_armor;

  ArmorNumber tracked_id;
  std::string type;
  int retype;
  Eigen::VectorXd measurement;
  Eigen::VectorXd target_state;

  double distance_to_image_center;

  double d_za, another_r;
  double d_zc;
  float yaw_diff_;
  float position_diff_;
  double jump_thresh = 0.4;
  double max_match_distance_;
  double max_match_yaw_diff_;
  double max_match_z_diff_;
  double max_match_x_diff_;

private:
  void initEKF(const Armor &a) noexcept;
  void handleArmorJump(const Armor &a) noexcept;

  double orientationToYaw(const tf::Quaternion &q) noexcept;

  int detect_count_;
  int lost_count_;

  double last_yaw_;

  std::chrono::steady_clock::time_point last_track_time_;
  std::deque<float> yaw_velocity_buffer_;

  int track_update_count_ = 0;
  bool if_have_last_track_ = false;
  double last_track_yaw_;

  int rotation_inconsistent_count_ = 0;

  int rotation_inconsistent_cooldown_ = 0;

  std::string tracker_logger = "onetracker";
  std::deque<std::chrono::steady_clock::time_point> armor_jump_timestamps_;
};
