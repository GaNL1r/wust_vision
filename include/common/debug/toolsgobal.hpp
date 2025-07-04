#pragma once

#include "common/tf.hpp"
#include "detect/mono_measure_tool.hpp"
#include <thread>
namespace toolsgobal {

extern std::chrono::steady_clock::time_point start_time_;

extern std::thread robot_cmd_plot_thread_;

extern std::mutex robot_cmd_mutex_;
extern std::vector<double> time_log_;
extern std::vector<double> cmd_yaw_log_;
extern std::vector<double> cmd_pitch_log_;
extern std::vector<double> armor_dis_log_;
extern std::vector<double> armor_x_log_;
extern std::vector<double> armor_y_log_;
extern std::vector<double> armor_z_log_;
extern std::vector<double> armor_yaw_log_;
extern std::vector<double> ypd_y_log_;
extern std::vector<double> ypd_p_log_;
extern std::vector<double> rune_obs_log_;
extern std::vector<double> rune_pre_log_;

extern int debug_w;
extern int debug_h;

extern double latency_ms;
extern double debug_fps;
} // namespace toolsgobal