#pragma once

#include "common/gobal.hpp"
#include "control/armor_solver.hpp"
#include "control/rune_solver.hpp"
#include "detect/armor_detect/armor_pose_estimator.hpp"
#include "detect/detector_factory.hpp"
#include "driver/hik.hpp"
#include "driver/serial.hpp"
#include "driver/tools/labeler.hpp"
#include "driver/tools/recorder.hpp"
#include "driver/tools/video_player.hpp"
#include "tracker/tracker_manager.hpp"
#include "type/type.hpp"
#include "yaml-cpp/yaml.h"
#include <opencv2/core/mat.hpp>
/*目前的思路： 一个OmniVision 只拥有一个相机示例，由OmniManager管理多个OmniVision实例并管理识别，将识别目标直接构建为一个低优先级的OneTarget,在主控制器中注入给onetargets，时间戳过远丢弃*/

class OmniVision {
public:
    OmniVision();
    ~OmniVision();
    void stop();
    bool init(
        const YAML::Node& config,
        std::function<void(const ImageFrame&, const Eigen::Matrix3d&, const Eigen::Vector3d&)>
            on_frame_callback_,
        size_t index
    );
    void initTF(const YAML::Node& config);
    void run();

    std::unique_ptr<HikCamera> camera_;
    std::unique_ptr<VideoPlayer> video_player_;

    bool use_video = false;
    bool is_inited_ = false;
    size_t index;
    cv::Mat camera_intrinsic_;
    cv::Mat camera_distortion_;
    double gimbal2camera_roll_, gimbal2camera_pitch_, gimbal2camera_yaw_;
    std::function<void(const ImageFrame&, const Eigen::Matrix3d&, const Eigen::Vector3d&)> callback;
};

class OmniManager {
public:
    OmniManager(const YAML::Node& config);
    ~OmniManager();
    void stop();
    void initdetector();
    void run();
    void processImage(
        const ImageFrame& frame,
        const Eigen::Matrix3d& R_gimbal2odom,
        const Eigen::Vector3d& v
    );
    void ArmorDetectCallback(
        const std::vector<ArmorObject>& objs,
        std::chrono::steady_clock::time_point timestamp,
        const cv::Mat& src_img,
        const Eigen::Matrix4d& T_camera_to_odom,
        const Eigen::Vector3d& v
    );
    std::vector<OneTarget> buildOneTargetsfromOmni(const Armors& armors);
    void startTimer();
    void stopTimer();
    void timerCallback(double dt_ms);
    std::atomic<bool> timer_running_ { false };
    std::thread timer_thread_;
    std::mutex timer_mtx_;
    std::condition_variable timer_cv_;
    int total_fps_ = 0;
    YAML::Node config_;
    std::unique_ptr<ThreadPool> thread_pool_;
    std::unique_ptr<ArmorDetectorBase> armor_detector_;
    std::vector<std::unique_ptr<OmniVision>> omni_visions_;
    std::unique_ptr<MonoMeasureTool> measure_tool_;
    size_t img_recv_count_ = 0;
    size_t detect_finish_count_ = 0;
    int max_infer_running_;
    int max_detect_armors_;
    std::mutex callback_mutex_;
    std::atomic<int> infer_running_count_ { 0 };
    std::string vision_logger = "omni_vision";
    size_t omni_num_ = 0;
    int count_ = 0;
    std::function<void(const ImageFrame&, const Eigen::Matrix3d&, const Eigen::Vector3d&)> callback;
};
