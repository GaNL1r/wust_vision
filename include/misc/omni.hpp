#pragma once

#include "common/gobal.hpp"
#include "common/timer.hpp"
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
        std::function<void(const ImageFrame&)> on_frame_callback_,
        size_t index
    );
    void initTF(const YAML::Node& config);
    void run();

    std::unique_ptr<HikCamera> camera_;
    std::unique_ptr<VideoPlayer> video_player_;

    bool use_video = false;
    bool is_inited_ = false;
    double video_alpha = 1.0;
    double video_beta = 0.0;
    size_t index_;
    cv::Mat camera_intrinsic_;
    cv::Mat camera_distortion_;
    double gimbal2camera_roll_, gimbal2camera_pitch_, gimbal2camera_yaw_;
    std::function<void(const ImageFrame&)> callback_;
};

class OmniManager {
public:
    OmniManager(const YAML::Node& config);
    ~OmniManager();
    void stop();
    void initDetector();
    void start();
    void processImage(const ImageFrame& frame);
    void ArmorDetectCallback(const std::vector<armor::ArmorObject>& objs, const CommonFrame& frame);
    std::vector<armor::OneTarget> buildOneTargetsfromOmni(const armor::Armors& armors);
    void timerCallback(double dt_ms);
    std::unique_ptr<Timer> timer_;
    int total_fps_ = 0;
    YAML::Node config_;

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
};
