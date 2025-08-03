#include "common/debug/toolsgobal.hpp"
namespace toolsgobal {

std::chrono::steady_clock::time_point start_time_ = std::chrono::steady_clock::now();
std::thread debug_thread_;
std::mutex robot_cmd_mutex_;
DebugLogs debug_logs_;
double latency_ms;
int debug_w;
int debug_h;
double debug_fps;
LatencyAveragerDeque latency_averager(100);
} // namespace toolsgobal
