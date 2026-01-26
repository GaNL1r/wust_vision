#pragma once
#include "tasks/auto_aim/armor_tracker/target.hpp"
#include "tasks/auto_aim/auto_aim_fsm.hpp"
#include "tasks/auto_aim/type.hpp"
#include "tasks/auto_buff/rune_tracker/rune_target.hpp"
#include "tasks/auto_buff/type.hpp"
#include "tasks/packet_typedef.hpp"
#include <nlohmann/json.hpp>
namespace wust_vision {
struct DebugArmor {
    imgframe src_img;
    auto_aim::Armors armors;
    auto_aim::Target target;
    GimbalCmd gimbal_cmd;
    auto_aim::AutoAimFsm fsm;
    AimTarget aim_target;
    double latency_ms;
    Eigen::Matrix4d T_camera_to_odom;
    std::vector<auto_aim::ArmorObject> armor_objs;
    int detect_color = 0;
    cv::Rect expanded;
};
struct DebugRune {
    imgframe src_img;
    auto_buff::RuneTarget target;
    AimTarget aim_target;
    auto_buff::PowerRune power_rune;
    double predict_angle;
    GimbalCmd gimbal_cmd;
    Eigen::Matrix4d T_camera_to_odom;
    std::string debug_text;
    double latency_ms;
    double obs_angle;
    double pre_angle;
    double fitter_v;
    double obs_v;
    cv::Rect expanded;
    double pnp_distance;
};


void drawDebugOverlayShm(
    const DebugArmor& dbg,
    std::pair<cv::Mat, cv::Mat> camera_info,
    bool auto_fps
);
void drawDebugOverlayWrite(
    const DebugArmor& dbg,
    std::pair<cv::Mat, cv::Mat> camera_info,
    bool auto_fps
);
void drawDebugOverlayShow(
    const DebugArmor& dbg,
    std::pair<cv::Mat, cv::Mat> camera_info,
    bool auto_fps
);
cv::Mat drawDebugOverlayMat(const DebugArmor& dbg, std::pair<cv::Mat, cv::Mat> camera_info);
void drawDebugArmorContent(
    cv::Mat& debug_img,
    const DebugArmor& dbg,
    std::pair<cv::Mat, cv::Mat> camera_info
);
void drawDebugOverlayShm(
    const DebugRune& dbg,
    std::pair<cv::Mat, cv::Mat> camera_info,
    bool auto_fps
);
void drawDebugOverlayWrite(
    const DebugRune& dbg,
    std::pair<cv::Mat, cv::Mat> camera_info,
    bool auto_fps
);
void drawDebugOverlayShow(
    const DebugRune& dbg,
    std::pair<cv::Mat, cv::Mat> camera_info,
    bool auto_fps
);
cv::Mat drawDebugOverlayMat(const DebugRune& dbg, std::pair<cv::Mat, cv::Mat> camera_info);
void drawDebugRuneContent(
    cv::Mat& debug_img,
    const DebugRune& dbg,
    std::pair<cv::Mat, cv::Mat> camera_info
);
void debuglog(
    const DebugArmor& dbg_armor,
    const DebugRune& dbg_rune,
    const GimbalCmd& gimbal_cmd,
    const std::pair<double, double>& gimbal_py
);
void writeSerialLogToJson(const ReceiveAimINFO& aim);
} // namespace wust_vision