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
#include "omni.hpp"
#include "tracker/tracker_manager.hpp"
#include "type/type.hpp"
#include "yaml-cpp/yaml.h"
#include <Eigen/Dense>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <mutex>
#include <opencv2/core/mat.hpp>
#include <thread>
#include <vector>

class WustVision {
public:
    // Constructor & Destructor
    WustVision();
    ~WustVision();

    // Lifecycle
    bool init();
    void start();
    void stop();

    // Image processing
    void processImage(const ImageFrame& frame);

    // Callbacks
    void ArmorDetectCallback(const std::vector<armor::ArmorObject>& objs, const CommonFrame& frame);
    void RuneDetectCallback(std::vector<rune::RuneObject>& rune_objects, const CommonFrame& frame);
    void armorsCallback(
        armor::Armors armors_,
        const cv::Mat& src_img,
        const Eigen::Matrix3d& R_gimbal2odom,
        const Eigen::Vector3d& v
    );
    void runeTargetCallback(const rune::Rune rune_target, Eigen::Matrix4d T_camera_to_odom);

    // Solvers
    GimbalCmd solveByMode(
        AttackMode mode,
        const ArmorSloverTarget& armor_slover_target,
        const std::chrono::steady_clock::time_point& now
    );

    // Initialization helpers
    void initDetector();
    void initTF();
    void initSerial();
    void initTracker(const YAML::Node& config);
    void initRune(const std::string& camera_info_path);

    // Timer control
    void startTimer();
    void stopTimer();
    void restartTimerThread();
    void timerCallback(double dt_ms);

    // Utilities
    void update();
    void visualizeAndLog(bool auto_fps = true);
    void printStats();
    void debugThread();
    void saveAutoLabelData(const std::vector<armor::ArmorObject>& objs, const CommonFrame& frame);
    void calculationManualR(const cv::Mat& src_img);
    static void onMouse(int event, int x, int y, int, void* userdata);
    armor::Armors visualizeTargetProjection(
        armor::Target armor_target_,
        std::vector<armor::OneTarget> one_armor_targets_
    );
    void reloadConfig();

private:
    // Camera & Video
    std::unique_ptr<HikCamera> camera_;
    std::unique_ptr<VideoPlayer> video_player_;
    double video_alpha_ = 1.0;
    double video_beta_ = 0.0;
    bool use_video_ = false;

    // Serial & Omni
    std::unique_ptr<serial::Serial> serial_;
    bool use_omni_ = false;
    std::unique_ptr<OmniManager> omni_manager_;

    // Detection & Tracking
    std::unique_ptr<ArmorDetectorBase> armor_detector_;
    std::unique_ptr<RuneDetectorBase> rune_detector_;
    std::unique_ptr<ArmorSolver> armor_solver_;
    std::unique_ptr<RuneSolver> rune_solver_;
    std::unique_ptr<TrackerManager> tracker_manager_;
    std::unique_ptr<ArmorPoseEstimator> armor_pose_estimator_;

    // Configuration
    YAML::Node rune_detect_config_;
    int max_detect_armors_ = 0;
    int max_infer_running_;
    bool only_nav_enable_ = false;
    bool detect_r_tag_ = false;
    int rune_binary_thresh_ = 0;
    bool use_auto_labeler_ = false;
    std::unique_ptr<Labeler> auto_labeler_;

    // Transformation parameters
    double gimbal2camera_x_ = 0.0, gimbal2camera_y_ = 0.0, gimbal2camera_z_ = 0.0;
    double gimbal2camera_yaw_ = 0.0, gimbal2camera_roll_ = 0.0, gimbal2camera_pitch_ = 0.0;
    Eigen::Vector3d t_gimbal_to_camera_;
    Eigen::Matrix4d T_camera_to_odom_;
    Eigen::Matrix3d R_camera2gimbal_;

    // Frame & Image data
    imgframe imgframe_;
    std::mutex img_mutex_;

    // Threading & synchronization
    std::mutex callback_mutex_;
    std::atomic<int> infer_running_count_ { 0 };
    std::atomic<bool> timer_running_ { false };
    std::thread timer_thread_;
    int timer_count_ = 0;
    std::mutex timer_mtx_;
    std::condition_variable timer_cv_;

    // Statistics
    std::string vision_logger_ = "wust_vision";
    size_t img_recv_count_ = 0;
    size_t detect_finish_count_ = 0;
    size_t fire_count_ = 0;
    std::chrono::steady_clock::time_point last_stat_time_steady_;
    double debug_show_dt_ = 0.0;

    // Target data
    armor::Target armor_target_;
    std::vector<armor::OneTarget> one_armor_targets_;
    armor::Armors armors_gobal_;
    rune::Rune rune_gobal_;

    std::vector<rune::RuneObject> rune_objects_;

    // Last target & tracking
    armor::Armor last_armor_;
    double last_distance_ = 0.0;
    double last_ypd_y_ = 0.0, last_ypd_p_ = 0.0;
    double last_armor_yaw_ = 0.0;
    double jump_yaw = 0.0;
    double receive_omni_dt_ = 0.0, hit_omni_dt_ = 0.0;
    std::chrono::steady_clock::time_point last_track_target_;

    // Manual calibration
    static std::vector<cv::Point2f> clicked_points_;
    bool manual_r_init_ = false;
    bool use_calculation_ = false;
    cv::Point2f manual_r_center_;
    std::vector<cv::Point2f> manual_r_box_;
    bool use_manual_r_ = false;
    std::atomic<bool> manual_r_runing_ { false };
    Eigen::Matrix4d T_r_;
};
