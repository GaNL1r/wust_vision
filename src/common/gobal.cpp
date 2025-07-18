#include "common/gobal.hpp"

namespace gobal {
std::atomic<bool> exit_flag(false);
std::unique_ptr<MonoMeasureTool> measure_tool_;
int detect_color_;
bool debug_mode_ = false;
double controller_delay = 0.0;
double velocity = 25.0;
bool if_manual_reset = false;
int control_rate;
double last_roll = 0;
double last_pitch = 0;
double last_yaw = 0;
double last_v_x = 0;
double last_v_y = 0;
double last_v_z = 0;
double gimbal2camera_yaw = 0;
double gimbal2camera_roll = 0;
double gimbal2camera_pitch = 0;
bool is_inited_ = false;
YAML::Node config;
bool use_calculation_ = false;
bool use_serial = false;
int attack_mode = 0;
double communication_delay_μs;
MotionBuffer motion_buffer;
cv::Mat camera_intrinsic_;
cv::Mat camera_distortion_;
AttackState attack_state;
int use_detect_ncnn_count = 0;
std::vector<OneTarget> omni_targets;
GimbalCmd last_cmd_;
} // namespace gobal
