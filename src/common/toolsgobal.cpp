#include "common/toolsgobal.hpp"
namespace toolsgobal {

std::chrono::steady_clock::time_point start_time_ = std::chrono::steady_clock::now();

std::thread robot_cmd_plot_thread_;
std::mutex robot_cmd_mutex_;
std::vector<double> time_log_;
std::vector<double> cmd_yaw_log_;
std::vector<double> cmd_pitch_log_;
std::vector<double> armor_dis_log_;
std::vector<double> armor_x_log_;
std::vector<double> armor_y_log_;
std::vector<double> armor_z_log_;
std::vector<double> armor_yaw_log_;
std::vector<double> ypd_y_log_;
std::vector<double> ypd_p_log_;
std::vector<double> rune_obs_log_;
std::vector<double> rune_pre_log_;
double latency_ms;
int debug_w;
int debug_h;

} // namespace toolsgobal
