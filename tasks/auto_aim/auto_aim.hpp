#pragma once
#include "tasks/auto_aim/armor_detect/armor_pose_estimator.hpp"
#include "tasks/auto_aim/armor_detect/detector_factory.hpp"
#include "tasks/auto_aim/armor_tracker/tracker_manager.hpp"
#include "tasks/debug.hpp"
#include "tasks/type_common.hpp"
#include "wust_vl/common/concurrency/queues.hpp"
#include "wust_vl/common/utils/config_binder.hpp"
#include "wust_vl/common/utils/timer.hpp"
#include "yaml-cpp/yaml.h"
namespace auto_aim {
struct AutoAimShared {
    std::shared_ptr<MotionBuffer> motion_buffer;
    double bullet_speed;
    double controller_delay;
    AutoAimShared(std::shared_ptr<MotionBuffer> mb) {
        motion_buffer = mb;
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
        const Eigen::Vector3d& t_gimbal_to_camera,
        const std::pair<cv::Mat, cv::Mat>& camera_info,
        std::shared_ptr<ConfigBinder> config_binder
    );
    void start();
    void pushInput(CommonFrame& frame);
    void setDebug(bool debug);
    DebugArmor getDebugFrame();
    GimbalCmd solve();
    void setShared(std::shared_ptr<AutoAimShared> shared);
    struct Impl;
    std::unique_ptr<Impl> _impl;
};
} // namespace auto_aim
