#pragma once

#include "common/tf.hpp"
#include "detect/mono_measure_tool.hpp"
#include "tracker/tracker.hpp"
#include <thread>

struct DebugLog {
    double time_log = 0.0;
    double cmd_yaw_log = 0.0;
    double cmd_pitch_log = 0.0;
    double armor_dis_log = 0.0;
    double armor_x_log = 0.0;
    double armor_y_log = 0.0;
    double armor_z_log = 0.0;
    double armor_yaw_log = 0.0;
    double ypd_y_log = 0.0;
    double ypd_p_log = 0.0;
    double rune_obs_log = 0.0;
    double rune_pre_log = 0.0;

    void clear() {
        time_log = 0.0;
        cmd_yaw_log = 0.0;
        cmd_pitch_log = 0.0;
        armor_dis_log = 0.0;
        armor_x_log = 0.0;
        armor_y_log = 0.0;
        armor_z_log = 0.0;
        armor_yaw_log = 0.0;
        ypd_y_log = 0.0;
        ypd_p_log = 0.0;
        rune_obs_log = 0.0;
        rune_pre_log = 0.0;
    }
};
struct DebugArmor {
    std::optional<imgframe> src_img; // 图像帧
    std::optional<Armors> armors; // 装甲板列表
    std::optional<Target_info> target_info; // 目标信息
    std::optional<Target> target; // 当前目标
    std::optional<Tracker::State> tracker_state; // 跟踪状态
    std::optional<GimbalCmd> gimbal_cmd; // 云台指令
};
struct DebugRune {
    std::optional<imgframe> src_img; // 图像帧
    std::optional<std::vector<RuneObject>> objs; // 目标对象列表
    std::optional<double> predict_angle; // 预测角度
    std::optional<GimbalCmd> gimbal_cmd; // 云台指令
    std::optional<std::vector<cv::Point2f>> manual_r_box; // 手动标注框点集
};

namespace toolsgobal {

extern std::chrono::steady_clock::time_point start_time_;

extern std::thread robot_cmd_plot_thread_;

extern std::mutex robot_cmd_mutex_;
extern std::vector<DebugLog> debug_logs_;

extern int debug_w;
extern int debug_h;

extern double latency_ms;
extern double debug_fps;
} // namespace toolsgobal