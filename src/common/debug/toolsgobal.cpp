#include "common/debug/toolsgobal.hpp"
namespace toolsgobal {
std::mutex robot_cmd_mutex_;
DebugLogs debug_logs_;
int debug_w;
int debug_h;
double debug_fps;
double latency_ms;
} // namespace toolsgobal
