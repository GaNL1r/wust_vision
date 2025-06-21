#include "wust_vision_trt.hpp"
#include "common/calculation.hpp"
#include "common/gobal.hpp"
#include "common/logger.hpp"
#include "common/matplottools.hpp"
#include "common/tf.hpp"
#include "common/tools.hpp"
#include "detect/mono_measure_tool.hpp"
#include "string"
#include "type/type.hpp"
#include <csignal>
#include <iostream>
#include <vector>
#include <yaml-cpp/yaml.h>
WustVision::WustVision() { init(); }
WustVision::~WustVision() {}
void WustVision::stop() {
  is_inited_ = false;

  if (!only_nav_enable) {
    auto_labeler_.reset();
    if (use_video) {
      video_player_->stop();
    } else {
      // capture_running_ = false;
      // if (capture_thread_ && capture_thread_->joinable()) {
      //   capture_thread_->join();
      // }
      camera_->stopCamera();
      camera_.reset();
    }

    stopTimer();
    std::this_thread::sleep_for(std::chrono::seconds(1));

    detector_.reset();
    measure_tool_.reset();

    if (thread_pool_) {
      thread_pool_->waitUntilEmpty();
      thread_pool_.reset();
    }
    if (robot_cmd_plot_thread_.joinable()) {
      robot_cmd_plot_thread_.join();
    }
    if (thread_pool_) {
      thread_pool_->waitUntilEmpty();
    }
  }
  serial_.stopThread();

  WUST_INFO(vision_logger) << "WustVision shutdown complete.";
}
void WustVision::stopTimer() {
  timer_running_ = false;
  if (timer_thread_.joinable()) {
   
    timer_thread_.detach();
  }
}

void WustVision::init() {
  config = YAML::LoadFile("/home/hy/wust_vision/config/config_trt.yaml");
  debug_mode_ = config["debug"]["debug_mode"].as<bool>();
  debug_w = config["debug"]["debug_w"].as<int>(640);
  debug_h = config["debug"]["debug_h"].as<int>(480);
  std::string log_level_ =
      config["logger"]["log_level"].as<std::string>("INFO");
  std::string log_path_ =
      config["logger"]["log_path"].as<std::string>("wust_log");
  bool use_logcli = config["logger"]["use_logcli"].as<bool>();
  bool use_logfile = config["logger"]["use_logfile"].as<bool>();
  bool use_simplelog = config["logger"]["use_simplelog"].as<bool>();
  initLogger(log_level_, log_path_, use_logcli, use_logfile, use_simplelog);
  control_rate = config["control"]["control_rate"].as<int>();
  only_nav_enable = config["only_nav_enable"].as<bool>();
  if (!only_nav_enable) {
    debug_show_dt_ = config["debug"]["debug_show_dt"].as<double>(0.05);
    use_calculation_ = config["use_calculation"].as<bool>();
    // 模型参数
    const std::string model_path = config["model_path"].as<std::string>();
    auto classify_model_path = config["classify_model_path"].as<std::string>();
    auto classify_label_path = config["classify_label_path"].as<std::string>();
    AdaptedTRTModule::Params params;
    params.input_w = config["model"]["input_w"].as<int>();
    params.input_h = config["model"]["input_h"].as<int>();
    params.num_classes = config["model"]["num_classes"].as<int>();
    params.num_colors = config["model"]["num_colors"].as<int>();
    params.conf_threshold = config["model"]["conf_threshold"].as<float>();
    params.nms_threshold = config["model"]["nms_threshold"].as<float>();
    params.top_k = config["model"]["top_k"].as<int>();
    float expand_ratio_w = config["light"]["expand_ratio_w"].as<float>();
    float expand_ratio_h = config["light"]["expand_ratio_h"].as<float>();
    int binary_thres = config["light"]["binary_thres"].as<int>();
    gimbal2camera_x_ = config["tf"]["gimbal2camera_x"].as<double>();
    gimbal2camera_y_ = config["tf"]["gimbal2camera_y"].as<double>();
    gimbal2camera_z_ = config["tf"]["gimbal2camera_z"].as<double>();
    gimbal2camera_roll_ = config["tf"]["gimbal2camera_roll"].as<double>();
    gimbal2camera_pitch_ = config["tf"]["gimbal2camera_pitch"].as<double>();
    gimbal2camera_yaw_ = config["tf"]["gimbal2camera_yaw"].as<double>();
    odom2gimbal_pitch = config["tf"]["odom2gimbal_pitch"].as<double>();
    odom2gimbal_roll = config["tf"]["odom2gimbal_roll"].as<double>();
    odom2gimbal_yaw = config["tf"]["odom2gimbal_yaw"].as<double>();

    LightParams l_params = {
        .min_ratio = config["light"]["min_ratio"].as<double>(),
        .max_ratio = config["light"]["max_ratio"].as<double>(),
        .max_angle = config["light"]["max_angle"].as<double>()};

    // 相机参数
    const std::string camera_info_path =
        config["camera"]["camera_info_path"].as<std::string>();
    measure_tool_ = std::make_unique<MonoMeasureTool>(camera_info_path);
    initTF();

    armor_pose_estimator_ =
        std::make_unique<ArmorPoseEstimator>(camera_info_path);

    use_serial = config["control"]["use_serial"].as<bool>();

    initTracker(config["tracker"]);

    detect_color_ = config["detect_color"].as<int>(0);
    max_infer_running_ = config["max_infer_running"].as<int>(4);

    if (model_path.empty()) {
      WUST_ERROR(vision_logger) << "Model path is empty.";
      return;
    }
    use_auto_labeler = config["use_auto_labeler"].as<bool>(false);
    if (use_auto_labeler) {
      auto_labeler_ = std::make_unique<Labeler>();
    }
    detector_ = std::make_unique<AdaptedTRTModule>(
      model_path, params, expand_ratio_h, expand_ratio_w, binary_thres,
      l_params, classify_model_path, classify_label_path);
    detector_->setCallback(
        std::bind(&WustVision::DetectCallback, this, std::placeholders::_1,
                  std::placeholders::_2, std::placeholders::_3,std::placeholders::_4));

    thread_pool_ =
        std::make_unique<ThreadPool>(std::thread::hardware_concurrency(), 100);

    solver_ = std::make_unique<Solver>(config);
    use_video = config["camera"]["video_player"]["use"].as<bool>(false);
    if (use_video) {
      std::string video_play_path =
          config["camera"]["video_player"]["path"].as<std::string>("");
      int video_play_fps = config["camera"]["video_player"]["fps"].as<int>(30);
      int start_frame =
          config["camera"]["video_player"]["start_frame"].as<int>(0);
      bool loop = config["camera"]["video_player"]["loop"].as<bool>(false);
      video_player_ = std::make_unique<VideoPlayer>(
          video_play_path, video_play_fps, start_frame, loop);
          video_player_ = std::make_unique<VideoPlayer>(
            video_play_path, video_play_fps, start_frame, loop);
            video_player_->setCallback([this](const ImageFrame &frame) {
              static bool first_is_inited = false;
      
              if (is_inited_) {
                Eigen::Matrix3d R_gimbal2odom;
                R_gimbal2odom = Eigen::AngleAxisd(last_yaw,   Eigen::Vector3d::UnitZ()) *
                Eigen::AngleAxisd(last_pitch, Eigen::Vector3d::UnitY()) *
                Eigen::AngleAxisd(last_roll,  Eigen::Vector3d::UnitX());
                thread_pool_->enqueue(
                    [frame = std::move(frame), R_gimbal2odom,this]() { processImage(frame,R_gimbal2odom); });
              } else {
                return;
              }
            });
            video_player_->start();
          } else {
            camera_ = std::make_unique<HikCamera>();
            if (!camera_->initializeCamera()) {
              WUST_ERROR(vision_logger) << "Camera initialization failed.";
              return;
            }
      
            camera_->setParameters(
                config["camera"]["acquisition_frame_rate"].as<int>(),
                config["camera"]["exposure_time"].as<int>(),
                config["camera"]["gain"].as<double>(),
                config["camera"]["adc_bit_depth"].as<std::string>(),
                config["camera"]["pixel_format"].as<std::string>());
            camera_->setFrameCallback([this](const ImageFrame &frame,Eigen::Matrix3d R_gimbal2odom) {
              static bool first_is_inited = false;
      
              if (is_inited_) {
                thread_pool_->enqueue(
                    [frame = std::move(frame),R_gimbal2odom, this]() { processImage(frame,R_gimbal2odom); });
              } else {
                return;
              }
            });
      bool if_recorder = config["camera"]["recorder"].as<bool>(false);

      camera_->startCamera(if_recorder);
      // bool trigger_mode = config["camera"]["trigger_mode"].as<bool>(false);
      // bool invert_image = config["camera"]["invert_image"].as<bool>(false);
      // int exposure_time_us =
      // config["camera"]["exposure_time"].as<int>(3500); float gain =
      // config["camera"]["gain"].as<float>(7.0f);

      // hikcamera::ImageCapturer::CameraProfile profile;
      // profile.trigger_mode = trigger_mode;
      // profile.invert_image = invert_image;
      // profile.exposure_time = std::chrono::microseconds(exposure_time_us);
      // profile.gain = gain;

      // capturer_ = std::make_unique<hikcamera::ImageCapturer>(
      //     profile, nullptr, hikcamera::SyncMode::NONE);
      // capture_running_ = true;
      // capture_thread_ =
      //     std::make_unique<std::thread>(&WustVision::captureLoop, this);
    }
    startTimer();
    robot_cmd_plot_thread_ = std::thread(&robotCmdLoggerThread);
  } else {
    WUST_INFO(vision_logger) << "only nav mode";
  }

  initSerial();
  is_inited_ = true;
}
// void WustVision::captureLoop() {
//   while (capture_running_ && is_inited_) {
//     // auto start = std::chrono::high_resolution_clock::now();
//     using namespace std::chrono_literals;

//     auto frame = capturer_->read();
//     auto now =  std::chrono::steady_clock::now();

//     if (!frame.empty()) {

//       thread_pool_->enqueue(
//           [frame = std::move(frame), this, now]() {
//             processImage(frame, now);
//           });
//       auto end =  std::chrono::steady_clock::now();
//       //  WUST_DEBUG(vision_logger) << "process time: "
//       //                          <<
//       std::chrono::duration_cast<std::chrono::milliseconds>(end -
//       now).count() << "ms";
//     }
//   }
// }
void WustVision::startTimer() {
  if (timer_running_) {
    return;
  }
  WUST_INFO(vision_logger) << "starting timer";
  timer_running_ = true;

  int ms_interval = 1000 / control_rate;

  timer_thread_ = std::thread([this, ms_interval]() {
    const auto interval = std::chrono::milliseconds(ms_interval);
    auto next_time = std::chrono::steady_clock::now() + interval;

    while (timer_running_) {
      std::this_thread::sleep_until(next_time);
      if (!timer_running_)
        break;

      // auto future = std::async(std::launch::async, [this]() {
      this->timerCallback();
      // });

      next_time += interval;
    }
  });
}

void WustVision::initTF() {
  // odom 是世界坐标系的根节点
  tf_tree_.setTransform("", "odom",
                        createTf(0, 0, 0, tf2::Quaternion(0, 0, 0, 1)), true);

  // camera 相对于 odom，设置 odom -> camera 的变换
  tf_tree_.setTransform("odom", "gimbal_odom",
                        createTf(0, 0, 0, tf2::Quaternion(0, 0, 0, 1)), true);
  double odom2gimbal_roll_ = odom2gimbal_roll * M_PI / 180;
  double odom2gimbal_pitch_ = odom2gimbal_pitch * M_PI / 180;
  double odom2gimbal_yaw_ = odom2gimbal_yaw * M_PI / 180;
  tf2::Quaternion oriodom2gimbal;
  oriodom2gimbal.setRPY(odom2gimbal_roll_, odom2gimbal_pitch_,
                        odom2gimbal_yaw_);

  tf_tree_.setTransform("gimbal_odom", "gimbal_link",
                        createTf(0, 0, 0, oriodom2gimbal), false);
  gimbal2camera_roll = gimbal2camera_roll_ * M_PI / 180;
  gimbal2camera_pitch = gimbal2camera_pitch_ * M_PI / 180;
  gimbal2camera_yaw = gimbal2camera_yaw_ * M_PI / 180;
  tf2::Quaternion origimbal2camera;
  origimbal2camera.setRPY(gimbal2camera_roll, gimbal2camera_pitch,
                          gimbal2camera_yaw);
  tf_tree_.setTransform("gimbal_link", "camera",
                        createTf(gimbal2camera_x_, gimbal2camera_y_,
                                 gimbal2camera_z_, origimbal2camera),
                        true);

  t_gimbal_to_camera = Eigen::Vector3d(
  gimbal2camera_x_, 
  gimbal2camera_y_, 
  gimbal2camera_z_);
  




  // 转换为旋转矩阵使用
  R_gimbal_camera  << 
  0,  0, 1,
  -1,  0, 0,
  0, -1, 0;

  // camera_optical_frame 相对于 camera，设置 camera -> camera_optical_frame
  // 的旋转变换
  double yaw = M_PI / 2;
  double roll = -M_PI / 2;
  double pitch = 0.0;

  tf2::Quaternion orientation;
  orientation.setRPY(roll, pitch, yaw);

  tf_tree_.setTransform("camera", "camera_optical_frame",
                        createTf(0, 0, 0, orientation), true);
}
void WustVision::initSerial() {
  SerialPortConfig cfg{/*baud*/ 115200, /*csize*/ 8,
                       boost::asio::serial_port_base::parity::none,
                       boost::asio::serial_port_base::stop_bits::one,
                       boost::asio::serial_port_base::flow_control::none};

  std::string device_name = config["control"]["device_name"].as<std::string>();
  serial_.init(device_name, cfg);
  serial_.alpha_yaw = config["control"]["alpha_yaw"].as<double>();
  serial_.alpha_pitch = config["control"]["alpha_pitch"].as<double>();
  serial_.max_yaw_change = config["control"]["max_yaw_change"].as<double>();
  serial_.max_pitch_change = config["control"]["max_pitch_change"].as<double>();
  bool if_use_nav = config["control"]["use_nav"].as<bool>(false);
  serial_.startThread(use_serial, if_use_nav);
}
void WustVision::initTracker(const YAML::Node &config) {
  // 目标参考坐标系
  target_frame_ = config["target_frame"].as<std::string>("odom");
  tracker_manager_ = std::make_unique<TrackerManager>(config);
}
void WustVision::armorsCallback(Armors armors_, const cv::Mat &src_img) {

  
  
  if (armors_.timestamp <= tracker_manager_->last_time_) {
    // WUST_WARN(vision_logger) << "Received out-of-order armor data,
    // discarded.";
    return;
  }

  if (debug_mode_) {
    std::lock_guard<std::mutex> target_lock(img_mutex_);
    imgframe_.img = src_img.clone();
    imgframe_.timestamp = armors_.timestamp;
    armors_gobal = armors_;
  }
  if (use_calculation_) {
    command_callbackypd(armors_);
    // return;
  }
  Target target_;
  std::vector<OneTarget> one_targets_;
  auto time = armors_.timestamp;
  target_.timestamp = time;
  target_.frame_id = target_frame_;
  tracker_manager_->update(target_, one_targets_, armors_, time);

 
  armor_target = target_;
  one_armor_targets = one_targets_;
  auto now = std::chrono::steady_clock::now();

  auto latency_nano = std::chrono::duration_cast<std::chrono::nanoseconds>(
    now - target_.timestamp)
    .count();
  latency_ms = static_cast<double>(latency_nano) / 1e6;
}


Armors WustVision::visualizeTargetProjection(
    Target armor_target_, std::vector<OneTarget> one_armor_targets_) {

  Armors armor_data;
  armor_data.frame_id = "odom";
  armor_data.timestamp = armor_target_.timestamp;

  if (armor_target_.tracking) {
    double yaw = armor_target_.yaw, r1 = armor_target_.radius_1,
           r2 = armor_target_.radius_2;
    float xc = armor_target_.position_.x, yc = armor_target_.position_.y,
          zc = armor_target_.position_.z;
    double d_za = armor_target_.d_za, d_zc = armor_target_.d_zc;
    xc = xc + armor_target_.velocity_.x * debug_show_dt_;
    yc = yc + armor_target_.velocity_.y * debug_show_dt_;
    zc = zc + armor_target_.velocity_.z * debug_show_dt_;
    yaw = yaw + armor_target_.v_yaw * debug_show_dt_;

    bool is_current_pair = true;

    armor_data.armors.clear();

    size_t a_n = armor_target_.armors_num;

    armor_data.armors.reserve(a_n);

    for (size_t i = 0; i < a_n; ++i) {
      double tmp_yaw = yaw + i * (2 * M_PI / a_n);
      double cos_yaw = std::cos(tmp_yaw);
      double sin_yaw = std::sin(tmp_yaw);

      Position pos;
      if (a_n == 4) {
        double r = is_current_pair ? r1 : r2;
        pos.z = zc + d_zc + (is_current_pair ? 0 : d_za);
        pos.x = xc - r * cos_yaw;
        pos.y = yc - r * sin_yaw;
        is_current_pair = !is_current_pair;
      } else {
        pos.z = zc;
        pos.x = xc - r1 * cos_yaw;
        pos.y = yc - r1 * sin_yaw;
      }

      tf2::Quaternion ori;
      ori.setRPY(M_PI / 2,
                 armor_target_.id == ArmorNumber::OUTPOST ? -0.2618 : 0.2618,
                 tmp_yaw);

      armor_data.armors.emplace_back(Armor{.type = armor_target_.type,
                                           .pos = pos,
                                           .ori = ori,
                                           .is_ok = true,
                                           //.target_pos = {xc, yc, zc},
                                           .distance_to_image_center = 0.0f});
    }
  }
  for (auto one_armor_target_ : one_armor_targets_) {
    if (one_armor_target_.tracking) {
      Position pos;
      pos.x = one_armor_target_.position_.x +
              one_armor_target_.velocity_.x * debug_show_dt_;
      pos.y = one_armor_target_.position_.y +
              one_armor_target_.velocity_.y * debug_show_dt_;
      pos.z = one_armor_target_.position_.z +
              one_armor_target_.velocity_.z * debug_show_dt_;
      double tmp_yaw =
          one_armor_target_.yaw + one_armor_target_.v_yaw * debug_show_dt_;
      tf2::Quaternion ori;
      ori.setRPY(M_PI / 2,
                 one_armor_target_.id == ArmorNumber::OUTPOST ? -0.2618
                                                              : 0.2618,
                 tmp_yaw);

      armor_data.armors.emplace_back(Armor{.type = one_armor_target_.type,
                                           .pos = pos,
                                           .ori = ori,
                                           .is_ok = false,
                                           .distance_to_image_center = 0.0f});
    }
  }

  return armor_data;
}

void WustVision::DetectCallback(const std::vector<ArmorObject> &objs,
                                std::chrono::steady_clock::time_point timestamp,
                                const cv::Mat &src_img,
                                Eigen::Matrix4d T_camera_to_odom) {
  std::lock_guard<std::mutex> lock(callback_mutex_);
  detect_finish_count_++;
  if (objs.size() >= 6) {
    WUST_WARN(vision_logger) << "Detected " << objs.size() << " objects"
                             << "too much";
    infer_running_count_--;
    return;
  }
  if (measure_tool_ == nullptr) {
    WUST_WARN(vision_logger) << "NO camera info";
    return;
  }
  Armors armors;
  armors.timestamp = timestamp;
  armors.frame_id = "camera_optical_frame";
  try {
    // auto target_time = armors.timestamp;
    // Transform tf;
    // if (!tf_tree_.getTransform(armors.frame_id, target_frame_, target_time,
    //                            tf)) {
    //   throw std::runtime_error("Transform not found.");
    // }

    // tf2::Quaternion tf_quat = tf.orientation;
    // // std::cout<<tf.orientation.x<<" "<<tf.orientation.y<<"
    // // "<<tf.orientation.z<<" "<<tf.orientation.w<<std::endl;
    // Eigen::Quaterniond eigen_quat(tf_quat.w, tf_quat.x, tf_quat.y, tf_quat.z);
    // imu_to_camera_ = eigen_quat.toRotationMatrix(); // Eigen::Matrix3d
    // imu_to_camera_ =
    //     Sophus::SO3d::fitToSO3(eigen_quat.toRotationMatrix()).matrix();

    Eigen::Matrix3d R_camera_to_odom = T_camera_to_odom.block<3,3>(0,0);
    imu_to_camera_ = R_camera_to_odom;

    // std::cout<<imu_to_camera_<<std::endl;

  } catch (const std::exception &e) {

    return;
  }

  armors.armors =
      armor_pose_estimator_->extractArmorPoses(objs, imu_to_camera_);

  measure_tool_->processDetectedArmors(objs, detect_color_, armors);

  infer_running_count_--;
  if (use_auto_labeler) {
    static int save_counter = 0; // 静态计数器记录保存次数

    for (const auto &obj : objs) {
      std::vector<float> csv_data;

      int number_ = formArmorNumber(obj.number);
      int color_ = formArmorColor(obj.color);

      const auto &pts = obj.is_ok ? obj.pts_binary : obj.pts;

      for (int i = 0; i < 4; ++i) {
        csv_data.push_back(pts[i].x);
        csv_data.push_back(pts[i].y);
      }

      csv_data.push_back(number_);
      csv_data.push_back(color_);

      save_counter++; // 每次处理一个对象后计数器加一
      if (save_counter % 10 == 0) { // 每10次执行一次保存
        cv::Mat img_save;
        cv::cvtColor(src_img, img_save, cv::COLOR_RGB2BGR);
        auto_labeler_->save(img_save, csv_data);
      }
    }
  }
  transformArmorData(armors,T_camera_to_odom);
  armorsCallback(armors, src_img);
}
void WustVision::transformArmorData(Armors &armors) {
  for (auto &armor : armors.armors) {
    // armor.number = ArmorNumber::OUTPOST;
    try {
      Transform tf(armor.pos, armor.ori, armors.timestamp);
      auto pose_in_target_frame = tf_tree_.transform(
          tf, armors.frame_id, target_frame_, armors.timestamp);

      armor.target_pos = pose_in_target_frame.position;
      armor.target_ori = pose_in_target_frame.orientation;

      armor.yaw = getRPYFromQuaternion(armor.target_ori).yaw;
      double yaw = armor.yaw * 180 / M_PI;

    } catch (const std::exception &e) {
      WUST_ERROR(vision_logger)
          << "Can't find transform from " << armors.frame_id << " to "
          << target_frame_ << ": " << e.what();
      return;
    }
  }
}
void WustVision::transformArmorData(Armors &armors, Eigen::Matrix4d T_camera_to_odom) {
  T_camera_to_odom_= T_camera_to_odom;
  for (auto &armor : armors.armors) {
    try {
      // Step 1: Transform position from camera to odom
      Eigen::Vector4d pos_camera(armor.pos.x, armor.pos.y, armor.pos.z, 1.0);
      Eigen::Vector4d pos_odom = T_camera_to_odom * pos_camera;

      armor.target_pos.x = pos_odom.x();
      armor.target_pos.y = pos_odom.y();
      armor.target_pos.z = pos_odom.z();

      // Step 2: Transform orientation from camera to odom
      Eigen::Matrix3d R_camera_to_odom = T_camera_to_odom.block<3, 3>(0, 0);
      Eigen::Quaterniond q_camera(armor.ori.w, armor.ori.x, armor.ori.y, armor.ori.z);
      Eigen::Matrix3d R_ori_camera = q_camera.normalized().toRotationMatrix();

      Eigen::Matrix3d R_ori_odom = R_camera_to_odom * R_ori_camera;
      Eigen::Quaterniond q_odom(R_ori_odom);

      armor.target_ori.w = q_odom.w();
      armor.target_ori.x = q_odom.x();
      armor.target_ori.y = q_odom.y();
      armor.target_ori.z = q_odom.z();
      //std::cout << "x" << armor.target_pos.x<< "y"<< armor.target_pos.y << "z"<< armor.target_pos.z << std::endl;

      // Step 3: Extract yaw (assuming you have a function like this)
      Eigen::Vector3d euler = R_ori_odom.eulerAngles(2, 1, 0); // ZYX
      armor.yaw = euler[0]; // yaw

    } catch (const std::exception &e) {
      WUST_ERROR(vision_logger) << "Error in camera-to-odom transform: " << e.what();
      return;
    }
  }
}


void WustVision::timerCallback() {

  if (!is_inited_)
    return;

  Target target;

  target = armor_target;

  std::vector<OneTarget> one_targets;
  one_targets = one_armor_targets;

  Tracker::State state;
  bool appear;
  bool one_appear = false;
  for (auto &one_target : one_targets) {
    if (one_target.tracking) {
      one_appear = true;
    }
  }
  if (target.tracking || one_appear) {
    appear = true;
    state = Tracker::TRACKING;

  } else {
    appear = false;
    state = Tracker::LOST;
  }
  auto now = std::chrono::steady_clock::now();


  GimbalCmd gimbal_cmd;

  if (target.tracking || one_appear) {
    try {

      gimbal_cmd = solver_->solve(target, one_targets, now);

      last_cmd_ = gimbal_cmd;
      if (gimbal_cmd.fire_advice) {
        fire_count_++;
      }
      serial_.transformGimbalCmd(gimbal_cmd, appear);
    } catch (...) {
      WUST_ERROR(vision_logger) << "solver error";
      serial_.transformGimbalCmd(last_cmd_, appear);
    }
  } else {
    serial_.transformGimbalCmd(last_cmd_, appear);
  }

  if (debug_mode_) {
    cv::Mat src;
    {
      std::lock_guard<std::mutex> lock(img_mutex_);
      src = imgframe_.img.clone();
    }


    Armors armors;

    armors = armors_gobal;

    

      Armors armor_data = visualizeTargetProjection(target, one_targets);

      // for (auto &armor : armor_data.armors) {
      //   try {
      //     Transform tf(armor.pos, armor.ori);
      //     auto pose_in_target_frame =
      //         tf_tree_.transform(tf, armor_data.frame_id,
      //                            "camera_optical_frame", target.timestamp);

      //     armor.target_pos = pose_in_target_frame.position;
      //     armor.target_ori = pose_in_target_frame.orientation;
      //   } catch (const std::exception &e) {
      //     WUST_ERROR(vision_logger)
      //         << "Can't find transform from " << armor_data.frame_id << " to "
      //         << target_frame_ << ": " << e.what();
      //     continue;
      //   }
      // }
      for (auto &armor : armor_data.armors) {
        try {
          // Step 1: 位置变换：odom → camera_optical_frame
          Eigen::Vector4d pos_odom;
          pos_odom << armor.pos.x, armor.pos.y, armor.pos.z, 1.0;
      
          Eigen::Matrix4d T_odom_to_camera = T_camera_to_odom_.inverse();
          Eigen::Vector4d pos_camera = T_odom_to_camera * pos_odom;
      
          armor.target_pos.x = pos_camera.x();
          armor.target_pos.y = pos_camera.y();
          armor.target_pos.z = pos_camera.z();
      
          // Step 2: 姿态变换：odom → camera_optical_frame
          Eigen::Quaterniond q_odom(armor.ori.w, armor.ori.x, armor.ori.y, armor.ori.z);
          Eigen::Matrix3d R_odom = q_odom.normalized().toRotationMatrix();
          Eigen::Matrix3d R_odom_to_camera = T_odom_to_camera.block<3, 3>(0, 0);
      
          Eigen::Matrix3d R_camera = R_odom_to_camera * R_odom;
          Eigen::Quaterniond q_camera(R_camera);
      
          armor.target_ori.x = q_camera.x();
          armor.target_ori.y = q_camera.y();
          armor.target_ori.z = q_camera.z();
          armor.target_ori.w = q_camera.w();
      
        } catch (const std::exception &e) {
          WUST_ERROR(vision_logger)
              << "Transform from odom to camera_optical_frame failed: " << e.what();
          continue;
        }
      }
      Target_info target_info;
      target_info.select_id = gimbal_cmd.select_id;

      if (!measure_tool_->reprojectArmorsCorners(armor_data, target_info))
        return;
      write_target_log_to_json(target);
      try{
      draw_debug_overlaywrite(imgframe_, &armors, &target_info, &target, state,
              gimbal_cmd);
      }catch (const std::exception &e) {
      std::cerr << "draw_debug_overlaywrite failed: " << e.what() << '\n';
      }


   


    double t = std::chrono::duration<double>(now - start_time_).count();
    {
      std::lock_guard<std::mutex> lock(yaw_log_mutex_);

      target_yaw_log_.emplace_back(t, target.yaw);
      if (target_yaw_log_.size() > 1000) {
        target_yaw_log_.erase(target_yaw_log_.begin(),
                              target_yaw_log_.begin() + target_yaw_log_.size() -
                                  1000);
      }
    }
    {
      std::lock_guard<std::mutex> lock(robot_cmd_mutex_);
      time_log_.push_back(t);
      cmd_yaw_log_.push_back(last_cmd_.yaw);
      cmd_pitch_log_.push_back(last_cmd_.pitch);
      if (!armors.armors.empty()) {
      
        std::vector<Armor> ok_armors;
        for (const auto &armor : armors.armors) {
          if (armor.is_ok&&armor.number != ArmorNumber::OUTPOST) {
            ok_armors.push_back(armor);
          }
        }
      
        if (!ok_armors.empty()) {
    
          auto min_armor_it = std::min_element(
              ok_armors.begin(), ok_armors.end(),
              [](const Armor &a, const Armor &b) {
                return a.distance_to_image_center < b.distance_to_image_center;
              });
      
          const Armor &min_armor = *min_armor_it;
          last_distance =
              std::sqrt(min_armor.target_pos.x * min_armor.target_pos.x +
                        min_armor.target_pos.y * min_armor.target_pos.y +
                        min_armor.target_pos.z * min_armor.target_pos.z);
          armor_dis_log_.push_back(last_distance);
        } else {
         
          armor_dis_log_.push_back(last_distance);
        }
      
      } else {
 
        armor_dis_log_.push_back(last_distance);
      }

      if (time_log_.size() > 1000) {
        time_log_.erase(time_log_.begin());
        cmd_yaw_log_.erase(cmd_yaw_log_.begin());
        cmd_pitch_log_.erase(cmd_pitch_log_.begin());
        armor_dis_log_.erase(armor_dis_log_.begin());
      }
    }
  }
}
// void WustVision::processImage(const cv::Mat
// &frame,std::chrono::steady_clock::time_point timestamp) {

//   img_recv_count_++;
//   if (infer_running_count_.load() >= max_infer_running_) {
//     return;
//   }

//   infer_running_count_++;
//   printStats();
//   detector_->pushInput(frame, timestamp);
// }
void WustVision::processImage(const ImageFrame &frame,Eigen::Matrix3d R_gimbal2odom) {

  img_recv_count_++;
  if (infer_running_count_.load() >= max_infer_running_) {

    return;
  }
  cv::Mat img;
  if (!use_video) {
    img = convertToMatrgb(frame);
  } else {
    img = convertToMatbgr(frame);
  }
  
 
  // Step 1: gimbal → odom
  Eigen::Matrix4d T_gimbal_to_odom = Eigen::Matrix4d::Identity();
  T_gimbal_to_odom.block<3, 3>(0, 0) = R_gimbal2odom;

  // Step 2: camera → gimbal （取 R 的转置）
  Eigen::Matrix3d R_camera_to_gimbal = R_gimbal_camera;
  Eigen::Vector3d t_camera_to_gimbal = -R_camera_to_gimbal * t_gimbal_to_camera;

  Eigen::Matrix4d T_camera_to_gimbal = Eigen::Matrix4d::Identity();
  T_camera_to_gimbal.block<3, 3>(0, 0) = R_camera_to_gimbal;
  T_camera_to_gimbal.block<3, 1>(0, 3) = t_camera_to_gimbal;

  // Step 3: camera → odom
  Eigen::Matrix4d T_camera_to_odom = T_gimbal_to_odom * T_camera_to_gimbal;


  infer_running_count_++;
  printStats();

 
  detector_->pushInput(img, frame.timestamp,T_camera_to_odom);
 
}
void WustVision::printStats() {
  using namespace std::chrono;

  auto now = steady_clock::now();

  if (last_stat_time_steady_.time_since_epoch().count() == 0) {
    last_stat_time_steady_ = now;
    return;
  }

  auto elapsed = duration_cast<duration<double>>(now - last_stat_time_steady_);
  if (elapsed.count() >= 1.0) {
    WUST_INFO(vision_logger)
        << "Received: " << img_recv_count_
        << ", Detected: " << detect_finish_count_
        << ", FPS: " << detect_finish_count_ / elapsed.count()
        << " Latency: " << latency_ms << "ms"
        << "  Fire: " << fire_count_;

    img_recv_count_ = 0;
    detect_finish_count_ = 0;
    fire_count_ = 0;
    last_stat_time_steady_ = now;
  }
}

WustVision *global_vision = nullptr;
std::mutex mtx;
std::condition_variable c;

void signalHandler(int signum) {
  WUST_INFO("main") << "Interrupt signal (" << signum << ") received.";
  exit_flag.store(true, std::memory_order_release);
}

int main() {
  WustVision vision;
  global_vision = &vision;

  std::signal(SIGINT, signalHandler);

  std::thread wait_thread([] {
    while (!exit_flag.load(std::memory_order_acquire)) {
      std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
    c.notify_one();
  });

  {
    std::unique_lock<std::mutex> lk(mtx);
    c.wait(lk, [] { return exit_flag.load(std::memory_order_acquire); });
  }

  wait_thread.join();
  vision.stop();
  return 0;
}