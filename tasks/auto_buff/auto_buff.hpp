#pragma once
#include "tasks/debug.hpp"
#include "tasks/type_common.hpp"
#include "wust_vl/common/concurrency/queues.hpp"
#include "wust_vl/common/utils/timer.hpp"
#include "yaml-cpp/yaml.h"

namespace auto_buff {
struct AutoBuffShared {
    std::shared_ptr<MotionBufferGeneric<Motion, 1024>> motion_buffer;
    double bullet_speed;
    double controller_delay;
    bool is_rune_big;
    Eigen::Matrix3d R_camera2gimbal;
    Eigen::Vector3d t_camera2gimbal;
    double communication_delay_μs;
    AutoBuffShared(std::shared_ptr<MotionBufferGeneric<Motion, 1024>> mb) {
        motion_buffer = mb;
    }
};
class AutoBuff {
public:
    AutoBuff();
    ~AutoBuff();
    bool init(
        const YAML::Node& config,
        int& use_detect_ncnn_count,
        const Eigen::Matrix3d& R_camera2gimbal,
        const Eigen::Vector3d& t_camera2gimbal,
        const std::pair<cv::Mat, cv::Mat>& camera_info
    );
    void start();
    void pushInput(CommonFrame& frame,bool is_big);
    void setDebug(bool debug);
    DebugRune getDebugFrame();
    GimbalCmd solve();
    void setShared(std::shared_ptr<AutoBuffShared> shared);
    bool isActive();
    void processingWait();
    void processingUp();
    struct Impl;
    std::unique_ptr<Impl> _impl;
};
} // namespace auto_buff