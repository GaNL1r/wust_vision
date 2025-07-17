#pragma once

#include "common/tf.hpp"
#include "detect/mono_measure_tool.hpp"
#include "yaml-cpp/yaml.h"
#include <optional>

namespace gobal {
extern std::atomic<bool> exit_flag;
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
extern double last_v_x;
extern double last_v_y;
extern double last_v_z;
extern double gimbal2camera_yaw, gimbal2camera_roll, gimbal2camera_pitch;

extern bool is_inited_;
extern YAML::Node config;
extern bool use_calculation_;
extern bool use_serial;
extern int attack_mode;
extern double communication_delay_μs;

extern MotionBuffer attitude_buffer;

extern cv::Mat camera_intrinsic_;
extern cv::Mat camera_distortion_;
extern std::atomic<bool> ncnn_gpu_destroyed;

extern enum AttackState { ATTACKONE, ATTACKWHOLECAR } attack_state;

extern int use_detect_ncnn_count;
extern std::vector<OneTarget> omni_targets;
extern GimbalCmd last_cmd_;

} // namespace gobal
