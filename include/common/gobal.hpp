#pragma once

#include "common/tf.hpp"
#include "detect/mono_measure_tool.hpp"
#include "yaml-cpp/yaml.h"
#include <optional>

namespace gobal {
extern tf::TfTree tf_tree_;
extern std::unique_ptr<MonoMeasureTool> measure_tool_;
extern int detect_color_;
extern bool debug_mode_;

extern double controller_delay;
extern double velocity;
extern bool if_manual_reset;
extern int control_rate;
extern double last_roll;
extern double last_pitch;
extern double last_yaw;
extern double gimbal2camera_yaw, gimbal2camera_roll, gimbal2camera_pitch;

extern bool is_inited_;
extern YAML::Node config;
extern bool use_calculation_;
extern bool use_serial;
extern int attack_mode;
extern double communication_delay_μs;

extern AttitudeBuffer attitude_buffer;

extern cv::Mat camera_intrinsic_;
extern cv::Mat camera_distortion_;

} // namespace gobal
