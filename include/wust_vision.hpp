#pragma once

#include "common/gobal.hpp"
#include "control/armor_solver.hpp"
#include "control/rune_solver.hpp"
#include "detect/armor_pose_estimator.hpp"
#include "detect/detector_factory.hpp"
#include "driver/hik.hpp"
#include "driver/image_capturer.hpp"
#include "driver/labeler.hpp"
#include "driver/recorder.hpp"
#include "driver/serial.hpp"
#include "driver/video_player.hpp"
#include "tracker/tracker_manager.hpp"
#include "type/type.hpp"
#include "yaml-cpp/yaml.h"
#include <opencv2/core/mat.hpp>
class WustVision {
public:
    WustVision();
    ~WustVision();

    void init();
    void run();
    void processImage(const ImageFrame& frame, Eigen::Matrix3d R_gimbal2odom);
    void processImage(const cv::Mat& frame, std::chrono::steady_clock::time_point timestamp);
    void captureLoop();

    void printStats();
    void ArmorDetectCallback(
        const std::vector<ArmorObject>& objs,
        std::chrono::steady_clock::time_point timestamp,
        const cv::Mat& src_img,
        Eigen::Matrix4d T_camera_to_odom
    );
    void RuneDetectCallback(
        std::vector<RuneObject>& rune_objects,
        std::chrono::steady_clock::time_point timestamp,
        const cv::Mat& img,
        Eigen::Matrix4d T_camera_to_odom
    );
    void stop();
    void armorsCallback(Armors armors_, const cv::Mat& src_img);
    void initTF();
    void initSerial();
    void initTracker(const YAML::Node& config);
    void timerCallback();
    void startTimer();
    void stopTimer();
    void restartTimerThread();
    void transformArmorData(Armors& armors);
    void transformArmorData(Armors& armors, Eigen::Matrix4d T_camera_to_odom);
    void runeTargetCallback(const Rune rune_target, Eigen::Matrix4d T_camera_to_odom);
    void update();
    void initRune(const std::string& camera_info_path);
    void calculation_manual_r(const cv::Mat& src_img);
    static void onMouse(int event, int x, int y, int, void* userdata);
    // Armors visualizeTargetProjection(Target armor_target_);
    Armors
    visualizeTargetProjection(Target armor_target_, std::vector<OneTarget> one_armor_targets_);

    std::thread image_thread_;
    std::unique_ptr<ThreadPool> thread_pool_;

    std::unique_ptr<HikCamera> camera_;
    std::unique_ptr<VideoPlayer> video_player_;
    int max_infer_running_;
    std::mutex callback_mutex_;
    std::atomic<int> infer_running_count_ { 0 };

#ifdef USE_OPENVINO
    std::string vision_logger = "openvino_vision";
#elif defined(USE_TRT)
    std::string vision_logger = "trt_vision";
#else
    static_assert(false, "No backend defined: USE_OPENVINO or USE_TRT");
#endif
    std::atomic<bool> run_loop_ { false };
    std::string target_frame_;
    std::atomic<bool> timer_running_ { false };
    std::thread timer_thread_;

    std::unique_ptr<TrackerManager> tracker_manager_;
    double gimbal2camera_x_, gimbal2camera_y_, gimbal2camera_z_;
    double gimbal2camera_yaw_, gimbal2camera_roll_, gimbal2camera_pitch_;

    std::unique_ptr<serial::Serial> serial_;
    std::unique_ptr<ArmorSolver> armor_solver_;
    int max_detect_armors_;
    std::unique_ptr<ArmorPoseEstimator> armor_pose_estimator_;
    Eigen::Matrix3d imu_to_camera_;
    bool only_nav_enable;
    std::unique_ptr<hikcamera::ImageCapturer> capturer_;
    std::unique_ptr<std::thread> capture_thread_;
    std::atomic<bool> capture_running_;
    Target armor_target;
    std::vector<OneTarget> one_armor_targets;
    Armors armors_gobal;
    Rune rune_gobal;
    std::mutex img_mutex_;
    imgframe imgframe_;

    std::unique_ptr<RuneSolver> rune_solver_;
    bool detect_r_tag_;
    int rune_binary_thresh_;
    Rune last_rune_target_;
    std::unique_ptr<Labeler> auto_labeler_;
    bool use_auto_labeler;
    bool use_video;
    std::vector<RuneObject> rune_objects_;
    Eigen::Vector3d t_gimbal_to_camera;
    Eigen::Matrix4d T_camera_to_odom_;
    Eigen::Matrix3d R_gimbal_camera;

    int timer_count_ = 0;
    std::mutex timer_mtx_;
    std::condition_variable timer_cv_;

    size_t img_recv_count_ = 0;
    size_t detect_finish_count_ = 0;
    size_t fire_count_ = 0;
    std::chrono::steady_clock::time_point last_stat_time_steady_;
    double debug_show_dt_;
    GimbalCmd last_cmd_;
    Armor last_armor_;
    double last_distance;
    double last_ypd_y;
    double last_ypd_p;
    double last_armor_yaw;

    std::unique_ptr<ArmorDetectorBase> armor_detector_;
    std::unique_ptr<RuneDetectorBase> rune_detector_;

    static std::vector<cv::Point2f> clicked_points_;
    bool manual_r_init = false;
    cv::Point2f manual_r_center;
    std::vector<cv::Point2f> manual_r_box;
    bool use_manual_r = false;
    std::atomic<bool> manual_r_runing = false;
    Eigen::Matrix4d T_r;
};