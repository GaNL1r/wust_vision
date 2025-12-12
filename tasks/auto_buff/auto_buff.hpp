#pragma once
#include "tasks/debug.hpp"
#include "tasks/type_common.hpp"
#include "wust_vl/video/camera.hpp"

namespace auto_buff {
struct AutoBuffShared {
    std::shared_ptr<MotionBufferGeneric<Motion, 1024>> motion_buffer;
    double bullet_speed;
    bool is_rune_big;
    double communication_delay_μs;
    AutoBuffShared(
        std::shared_ptr<MotionBufferGeneric<Motion, 1024>> mb,
        double bullet_speed,
        double communication_delay_μs
    ) {
        motion_buffer = mb;
        this->bullet_speed = bullet_speed;
        this->communication_delay_μs = communication_delay_μs;
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
        const std::pair<cv::Mat, cv::Mat>& camera_info,
        int max_detect_running
    );
    void start();
    void pushInput(CommonFrame& frame, bool is_big);
    void setDebug(bool debug);
    DebugRune getDebugFrame();
    GimbalCmd solve();
    void setShared(std::shared_ptr<AutoBuffShared> shared);
    bool isActive();
    void processingWait();
    void processingUp();
    void autoExposureControl(const cv::Mat& frame, std::shared_ptr<wust_vl_video::Camera> camera);
    struct Impl;
    std::unique_ptr<Impl> _impl;
};
} // namespace auto_buff