#include "common/gobal.hpp"

namespace gobal {
std::atomic<bool> exit_flag(false);
tf::TfTree tf_tree_;
std::unique_ptr<MonoMeasureTool> measure_tool_;
int detect_color_;
bool debug_mode_ = false;

double controller_delay = 0.0;
double velocity = 15.0;
bool if_manual_reset = false;
int control_rate;
double last_roll;
double last_pitch;
double last_yaw;
double gimbal2camera_yaw, gimbal2camera_roll, gimbal2camera_pitch;
bool is_inited_ = false;
YAML::Node config;
bool use_calculation_ = false;
bool use_serial = false;
int attack_mode = 0;
double communication_delay_μs;
AttitudeBuffer attitude_buffer;
cv::Mat camera_intrinsic_;
cv::Mat camera_distortion_;
std::atomic<bool> ncnn_gpu_destroyed = false;
} // namespace gobal
