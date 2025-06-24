#include "common/gobal.hpp"
#include "control/armor_solver.hpp"
#include "control/bspline.hpp"
#include "control/rune_solver.hpp"
#include "detect/armor_detector_openvino.hpp"
#include "detect/armor_pose_estimator.hpp"
#include "detect/rune_detector_openvino.hpp"
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
  void processImage(const ImageFrame &frame, Eigen::Matrix3d R_gimbal2odom);
  void processImage(const cv::Mat &frame,
                    std::chrono::steady_clock::time_point timestamp);
  void captureLoop();

  void printStats();
  void DetectCallback(const std::vector<ArmorObject> &objs,
                      std::chrono::steady_clock::time_point timestamp,
                      const cv::Mat &src_img, Eigen::Matrix4d T_camera_to_odom);
  void inferResultCallback(std::vector<RuneObject> &rune_objects,
                           std::chrono::steady_clock::time_point timestamp,
                           const cv::Mat &img,
                           Eigen::Matrix4d T_camera_to_odom);
  void stop();
  void armorsCallback(Armors armors_, const cv::Mat &src_img);
  void initTF();
  void initSerial();
  void initTracker(const YAML::Node &config);
  std::unique_ptr<RuneDetectorOpenvino> initRuneDetectorOpenvino();
  void timerCallback();
  void startTimer();
  void stopTimer();
  void transformArmorData(Armors &armors);
  void transformArmorData(Armors &armors, Eigen::Matrix4d T_camera_to_odom);
  void runeTargetCallback(const Rune rune_target,
                          Eigen::Matrix4d T_camera_to_odom);
  void update();
  void initRune(const std::string &camera_info_path);
  // Armors visualizeTargetProjection(Target armor_target_);
  Armors visualizeTargetProjection(Target armor_target_,
                                   std::vector<OneTarget> one_armor_targets_);

  std::thread image_thread_;
  std::unique_ptr<ThreadPool> thread_pool_;
  std::unique_ptr<OpenVino> detector_;
  std::unique_ptr<HikCamera> camera_;
  std::unique_ptr<VideoPlayer> video_player_;
  int max_infer_running_;
  std::mutex callback_mutex_;
  std::atomic<int> infer_running_count_{0};

  std::string vision_logger = "openvino_vision";
  std::atomic<bool> run_loop_{false};
  std::string target_frame_;
  std::atomic<bool> timer_running_{false};
  std::thread timer_thread_;
  std::unique_ptr<TrackerManager> tracker_manager_;
  double gimbal2camera_x_, gimbal2camera_y_, gimbal2camera_z_;
  double gimbal2camera_yaw_, gimbal2camera_roll_, gimbal2camera_pitch_;

  serial::Serial serial_;
  std::unique_ptr<Solver> solver_;

  std::unique_ptr<ArmorPoseEstimator> armor_pose_estimator_;
  Eigen::Matrix3d imu_to_camera_;
  bool only_nav_enable;
  std::unique_ptr<hikcamera::ImageCapturer> capturer_;
  std::unique_ptr<std::thread> capture_thread_;
  std::atomic<bool> capture_running_;
  Target armor_target;
  // OneTarget one_armor_target;
  std::vector<OneTarget> one_armor_targets;
  Armors armors_gobal;
  Rune rune_gobal;
  std::mutex img_mutex_;
  imgframe imgframe_;
  std::unique_ptr<RuneDetectorOpenvino> rune_detector_;
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
  int timer_count;

  std::unique_ptr<RealtimeBSplineSegment> spline;
};