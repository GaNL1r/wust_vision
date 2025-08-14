#pragma once

#include "common/gobal.hpp"
#include "common/ordered_queue.hpp"
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
#include "fun/have_fun.hpp"
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
    void armorsCallback(armor::Armors armors);
    void processingLoop();
    void runeTargetCallback(const rune::Rune rune_target);

    // Solvers
    GimbalCmd solveByMode(
        AttackMode mode,
        const ArmorSolverTarget& armor_slover_target,
        const std::chrono::steady_clock::time_point& now
    );

    // Initialization helpers
    void initDetector();
    void initTF();
    void initSerial();
    void initTracker(const YAML::Node& config);
    void initRune(const std::string& camera_info_path);

    // Timer control
    void timerCallback(double dt_ms);

    // Utilities
    void debugvisualize(bool auto_fps = true);
    void debuglog();
    void printStats();
    void debugThread();
    void saveAutoLabelData(const std::vector<armor::ArmorObject>& objs, const CommonFrame& frame);
    void calculationManualR(const cv::Point2f center);
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
    YAML::Node armor_detect_config_;
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

    struct DebugGobal {
        imgframe imgframe;
        armor::Target armor_target;
        std::vector<armor::OneTarget> one_armor_targets;
        armor::Armors armors_gobal;
        std::vector<rune::RuneObject> rune_objects;
    } debug_gobal_frame_;
    std::mutex dbg_mutex_;

    // Threading & synchronization
    std::unique_ptr<Timer> timer_;
    std::mutex callback_mutex_;
    int timer_count_ = 0;

    // Statistics
    std::string vision_logger_ = "wust_vision";
    std::atomic<int> infer_running_count_ { 0 };
    size_t img_recv_count_ = 0;
    size_t detect_finish_count_ = 0;
    size_t finish_count_ = 0;
    size_t fire_count_ = 0;
    std::chrono::steady_clock::time_point last_stat_time_steady_;
    double debug_show_dt_ = 0.0;

    // Target data
    int current_id_ = 0;
    std::thread processing_thread_;
    std::unique_ptr<OrderedQueue<armor::Armors>> armor_queue_;
    std::unique_ptr<OrderedQueue<rune::Rune>> rune_queue_;
    std::chrono::steady_clock::time_point last_rune_target_time_;
    ArmorSolverTarget armor_solver_target_;

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

    //have fun
    std::unique_ptr<HaveFun> have_fun_;
};
