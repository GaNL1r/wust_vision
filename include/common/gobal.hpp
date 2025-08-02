#pragma once

#include "common/ThreadPool.h"
#include "common/tf.hpp"
#include "detect/mono_measure_tool.hpp"
#include "yaml-cpp/yaml.h"
#include <optional>
#define ROOT_CONFIG "/home/hy/wust_vision/config/config_common.yaml"
#define OPENVINO_CONFIG "/home/hy/wust_vision/config/detect_openvino.yaml"
#define TENSORRT_CONFIG "/home/hy/wust_vision/config/detect_trt.yaml"
#define NCNN_CONFIG "/home/hy/wust_vision/config/detect_ncnn.yaml"
#define ONNXRUNTIME_CONFIG "/home/hy/wust_vision/config/detect_ort.yaml"
#define OPENCV_CONFIG "/home/hy/wust_vision/config/detect_opencv.yaml"
#define OMNI_CONFIG "/home/hy/wust_vision/config/omni_config.yaml"
namespace gobal {
extern std::atomic<bool> exit_flag;
extern std::unique_ptr<MonoMeasureTool> measure_tool;
extern int detect_color;
extern bool debug_mode;
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
extern bool use_serial;
extern int attack_mode;
extern double communication_delay_μs;
extern MotionBuffer motion_buffer;
extern cv::Mat camera_intrinsic;
extern cv::Mat camera_distortion;
extern enum AttackState { ATTACKONE, ATTACKWHOLECAR } attack_state;
extern enum ArmorSloveState { TRACKING_ARMOR = 0, TRACKING_CENTER = 1 } armor_slove_state;
extern int use_detect_ncnn_count;
extern std::vector<armor::OneTarget> omni_targets;
extern GimbalCmd last_cmd;
extern std::unique_ptr<ThreadPool> thread_pool;
} // namespace gobal
