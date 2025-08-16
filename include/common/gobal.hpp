#pragma once

#include "common/tf.hpp"
#include "detect/mono_measure_tool.hpp"
#include "wust_vl/common/ThreadPool.h"
#include "wust_vl/common/stringanything.hpp"
#include "yaml-cpp/yaml.h"
#include <optional>
#define ROOT_CONFIG "/home/hy/wust_vision/config/config_common.yaml"
#define OPENVINO_CONFIG "/home/hy/wust_vision/config/detect_openvino.yaml"
#define TENSORRT_CONFIG "/home/hy/wust_vision/config/detect_trt.yaml"
#define NCNN_CONFIG "/home/hy/wust_vision/config/detect_ncnn.yaml"
#define ONNXRUNTIME_CONFIG "/home/hy/wust_vision/config/detect_ort.yaml"
#define OPENCV_CONFIG "/home/hy/wust_vision/config/detect_opencv.yaml"
#define OMNI_CONFIG "/home/hy/wust_vision/config/omni_config.yaml"
#define FUN_CONFIG "/home/hy/wust_vision/config/fun.yaml"

struct CommonInfo {
    bool debug_mode;
    bool if_manual_reset;
    bool use_serial;
    int use_detect_ncnn_count;
};
struct GobalState {
    int attack_mode;
    enum AttackState { ATTACKONE, ATTACKWHOLECAR } attack_state;
    enum ArmorSloveState { TRACKING_ARMOR = 0, TRACKING_CENTER = 1 } armor_slove_state;
};

namespace gobal {
extern YAML::Node config;
extern bool is_inited_;
extern stringanything::Manager stringanything;
} // namespace gobal
