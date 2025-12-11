#pragma once
#include "tasks/auto_aim/armor_detect/armor_pose_estimator.hpp"
#include "tasks/auto_aim/armor_detect/detector_factory.hpp"
#include "tasks/auto_aim/armor_tracker/tracker_manager.hpp"
#include "tasks/debug.hpp"
#include "tasks/type_common.hpp"
#include "wust_vl/common/concurrency/queues.hpp"
#include "wust_vl/common/utils/config_binder.hpp"
#include "wust_vl/common/utils/timer.hpp"
#include "wust_vl/video/camera.hpp"
#include "yaml-cpp/yaml.h"
namespace auto_aim {
struct AutoAimShared {
    std::shared_ptr<MotionBufferGeneric<Motion, 1024>> motion_buffer;
    double bullet_speed;
    double communication_delay_μs;
    AutoAimShared(
        std::shared_ptr<MotionBufferGeneric<Motion, 1024>> mb,
        double bullet_speed,
        double communication_delay_μs
    ) {
        motion_buffer = mb;
        this->bullet_speed = bullet_speed;
        this->communication_delay_μs = communication_delay_μs;
    }
};
class AutoAim {
public:
    AutoAim();
    ~AutoAim();
    bool init(
        const YAML::Node& config,
        int& use_detect_ncnn_count,
        const Eigen::Matrix3d& R_camera2gimbal,
        const Eigen::Vector3d& t_camera2gimbal,
        const std::pair<cv::Mat, cv::Mat>& camera_info,
        std::shared_ptr<wust_vl_utils::ConfigBinder> config_binder
    );
    void start();
    void pushInput(CommonFrame& frame);
    void setDebug(bool debug);
    DebugArmor getDebugFrame();
    GimbalCmd solve(double dt_ms);
    void setShared(std::shared_ptr<AutoAimShared> shared);
    bool isActive();
    void processingWait();
    void processingUp();
    void autoExposureControl(const cv::Mat& frame, std::shared_ptr<wust_vl_video::Camera> camera);
    struct Impl;
    std::unique_ptr<Impl> _impl;
};
} // namespace auto_aim
