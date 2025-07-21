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
//#include "common/debug/crow.hpp"
#include <opencv2/core/mat.hpp>
class WustVision {
public:
    WustVision();
    ~WustVision();

    bool init();
    void run();
    void processImage(const ImageFrame& frame);

    void printStats();
    void ArmorDetectCallback(const std::vector<ArmorObject>& objs, const CommonFrame& frame);
    void RuneDetectCallback(std::vector<RuneObject>& rune_objects, const CommonFrame& frame);
    void stop();
    void armorsCallback(
        Armors armors_,
        const cv::Mat& src_img,
        const Eigen::Matrix3d& R_gimbal2odom,
        const Eigen::Vector3d& v
    );
    void runeTargetCallback(const Rune rune_target, Eigen::Matrix4d T_camera_to_odom);
    void initDetector();
    void initTF();
    void initSerial();
    void initTracker(const YAML::Node& config);
    void timerCallback(double dt_ms);
    void startTimer();
    void stopTimer();
    void restartTimerThread();
    void update();
    void initRune(const std::string& camera_info_path);
    void calculationManualR(const cv::Mat& src_img);
    static void onMouse(int event, int x, int y, int, void* userdata);
    Armors
    visualizeTargetProjection(Target armor_target_, std::vector<OneTarget> one_armor_targets_);
    GimbalCmd solveByMode(
        AttackMode mode,
        const Target& target,
        const std::vector<OneTarget>& one_targets,
        const std::chrono::steady_clock::time_point& now
    );
    void visualizeAndLog(
        AttackMode mode,
        const Target& target,
        const std::vector<OneTarget>& one_targets,
        const GimbalCmd& gimbal_cmd,
        Tracker::State state,
        const std::chrono::steady_clock::time_point& now
    );

    std::unique_ptr<HikCamera> camera_;
    std::unique_ptr<VideoPlayer> video_player_;
    double video_alpha_;
    double video_beta_;
    int max_infer_running_;
    std::mutex callback_mutex_;
    std::atomic<int> infer_running_count_ { 0 };

    std::string vision_logger_ = "wust_vision";

    std::atomic<bool> timer_running_ { false };
    std::thread timer_thread_;
    bool use_omni_ = false;
    std::unique_ptr<OmniManager> omni_manager_;
    std::unique_ptr<TrackerManager> tracker_manager_;
    double gimbal2camera_x_, gimbal2camera_y_, gimbal2camera_z_;
    double gimbal2camera_yaw_, gimbal2camera_roll_, gimbal2camera_pitch_;

    std::unique_ptr<serial::Serial> serial_;
    std::unique_ptr<ArmorSolver> armor_solver_;
    int max_detect_armors_;
    std::unique_ptr<ArmorPoseEstimator> armor_pose_estimator_;
    bool only_nav_enable_;

    Target armor_target_;
    std::vector<OneTarget> one_armor_targets_;
    Armors armors_gobal_;
    Rune rune_gobal_;
    std::mutex img_mutex_;
    imgframe imgframe_;
    YAML::Node rune_detect_config_;
    std::unique_ptr<RuneSolver> rune_solver_;
    bool detect_r_tag_;
    int rune_binary_thresh_;
    Rune last_rune_target_;
    std::unique_ptr<Labeler> auto_labeler_;
    bool use_auto_labeler_;
    bool use_video_;
    std::vector<RuneObject> rune_objects_;
    Eigen::Vector3d t_gimbal_to_camera;
    Eigen::Matrix4d T_camera_to_odom_;
    Eigen::Matrix3d R_camera2gimbal_;

    int timer_count_ = 0;
    std::mutex timer_mtx_;
    std::condition_variable timer_cv_;

    size_t img_recv_count_ = 0;
    size_t detect_finish_count_ = 0;
    size_t fire_count_ = 0;
    std::chrono::steady_clock::time_point last_stat_time_steady_;
    double debug_show_dt_;

    Armor last_armor_;
    double last_distance_;
    double last_ypd_y_;
    double last_ypd_p_;
    double last_armor_yaw_;

    std::unique_ptr<ArmorDetectorBase> armor_detector_;
    std::unique_ptr<RuneDetectorBase> rune_detector_;

    static std::vector<cv::Point2f> clicked_points_;
    bool manual_r_init_ = false;
    cv::Point2f manual_r_center_;
    std::vector<cv::Point2f> manual_r_box_;
    bool use_manual_r_ = false;
    std::atomic<bool> manual_r_runing_ = false;
    Eigen::Matrix4d T_r_;

    double jump_yaw;
    double receive_omni_dt_;
    double hit_omni_dt_;
    std::chrono::steady_clock::time_point last_track_target_;
    //std::unique_ptr<DebugWebServer> debug_server_;
};