#pragma once
#include "tasks/auto_aim/armor_tracker/target.hpp"
#include "tasks/auto_aim/auto_aim_fsm.hpp"
#include "tasks/auto_aim/type.hpp"
#include "tasks/auto_buff/rune_tracker/rune_target.hpp"
#include "tasks/auto_buff/type.hpp"
#include "tasks/packet_typedef.hpp"
struct DebugArmor {
    imgframe src_img;
    armor::Armors armors;
    Target target;
    GimbalCmd gimbal_cmd;
    AutoAimFsm fsm;
    AimTarget aim_target;
    double latency_ms;
    Eigen::Matrix4d T_camera_to_odom;
    std::vector<armor::ArmorObject> armor_objs;
    int detect_color = 0;
    cv::Rect expanded;
};
struct DebugRune {
    imgframe src_img;
    rune::RuneTarget target;
    AimTarget aim_target;
    rune::PowerRune power_rune;
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
struct DebugLogs {
    std::vector<double> time_log;
    std::vector<double> raw_yaw_log;
    std::vector<double> raw_pitch_log;
    std::vector<double> cmd_yaw_log;
    std::vector<double> cmd_pitch_log;
    std::vector<double> armor_dis_log;
    std::vector<double> armor_x_log;
    std::vector<double> armor_y_log;
    std::vector<double> armor_z_log;
    std::vector<double> armor_yaw_log;
    std::vector<double> ypd_y_log;
    std::vector<double> ypd_p_log;
    std::vector<double> rune_obs_log;
    std::vector<double> rune_pre_log;
    std::vector<double> rune_obsv_log;
    std::vector<double> rune_fitv_log;
    std::vector<double> gimbal_yaw_log;
    std::vector<double> gimbal_pitch_log;
    std::vector<double> target_v_yaw_log;
    std::vector<double> control_v_yaw_log;
    std::vector<double> control_v_pitch_log;
    std::vector<double> yaw_diff_log;
    std::vector<double> fire_log;
    std::vector<double> rune_dis_log;
    std::vector<double> fly_time_log;
    void clear() {
        time_log.clear();
        raw_yaw_log.clear();
        raw_pitch_log.clear();
        cmd_yaw_log.clear();
        cmd_pitch_log.clear();
        armor_dis_log.clear();
        armor_x_log.clear();
        armor_y_log.clear();
        armor_z_log.clear();
        armor_yaw_log.clear();
        ypd_y_log.clear();
        ypd_p_log.clear();
        rune_obs_log.clear();
        rune_pre_log.clear();
        rune_obsv_log.clear();
        rune_fitv_log.clear();
        gimbal_yaw_log.clear();
        gimbal_pitch_log.clear();
        target_v_yaw_log.clear();
        control_v_yaw_log.clear();
        yaw_diff_log.clear();
        fire_log.clear();
        rune_dis_log.clear();
        fly_time_log.clear();
    }
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