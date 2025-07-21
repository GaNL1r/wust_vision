#include "common/debug/toolsgobal.hpp"
namespace toolsgobal {

std::chrono::steady_clock::time_point start_time_ = std::chrono::steady_clock::now();

std::thread robot_cmd_plot_thread_;
std::mutex robot_cmd_mutex_;
std::vector<DebugLog> debug_logs_;
double latency_ms;
int debug_w;
int debug_h;
double debug_fps;
} // namespace toolsgobal
