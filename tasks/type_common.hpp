#pragma once
#include "Eigen/Dense"
#include <opencv2/opencv.hpp>
#define ROOT_CONFIG "/home/hy/wust_vision/config/config_common.yaml"
#define OPENVINO_CONFIG "/home/hy/wust_vision/config/detect_openvino.yaml"
#define TENSORRT_CONFIG "/home/hy/wust_vision/config/detect_trt.yaml"
#define NCNN_CONFIG "/home/hy/wust_vision/config/detect_ncnn.yaml"
#define ONNXRUNTIME_CONFIG "/home/hy/wust_vision/config/detect_ort.yaml"
#define OPENCV_CONFIG "/home/hy/wust_vision/config/detect_opencv.yaml"
#define OMNI_CONFIG "/home/hy/wust_vision/config/omni_config.yaml"
#define FUN_CONFIG "/home/hy/wust_vision/config/fun.yaml"

struct CommonFrame {
    cv::Mat src_img;
    std::chrono::steady_clock::time_point timestamp;
    Eigen::Matrix4d T_camera_to_odom;
    Eigen::Vector3d v;
    int id;
    int detect_color;
};
enum class EnemyColor {
    RED = 0,
    BLUE = 1,
    WHITE = 2,
};
inline std::string enemyColorToString(EnemyColor color) {
    switch (color) {
        case EnemyColor::RED:
            return "RED";
             break;
        case EnemyColor::BLUE:
            return "BLUE";
             break;
        case EnemyColor::WHITE:
            return "WHITE";
             break;
        default:
            return "UNKNOWN";
    }
}
struct GridAndStride {
    int grid0;
    int grid1;
    int stride;
};
struct imgframe {
    cv::Mat img;
    std::chrono::steady_clock::time_point timestamp;
};
struct GimbalCmd {
    std::chrono::steady_clock::time_point timestamp;
    float pitch = 0;
    float yaw = 0;
    float yaw_diff = 0;
    float pitch_diff = 0;
    float v_yaw = 0;
    float v_pitch = 0;
    float distance = -1;
    bool fire_advice = false;
    int select_id = -1;
};
enum class AttackMode { ARMOR = 0, SMALL_RUNE, BIG_RUNE, UNKNOWN };
inline AttackMode toAttackMode(int value) {
    switch (value) {
        case 0:
            return AttackMode::ARMOR;
        case 1:
            return AttackMode::SMALL_RUNE;
        case 2:
            return AttackMode::BIG_RUNE;
        default:
            return AttackMode::UNKNOWN;
    }
}