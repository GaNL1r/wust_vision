
// #include "ros2/ros2.hpp"
// #include "sensor_msgs/msg/camera_info.hpp"
// #include "sensor_msgs/msg/image.hpp"
// #include "tasks/utils/config.hpp"
// #include "tasks/utils/main_base.hpp"
// #include "tasks/vision_base.hpp"
// ENABLE_BACKWARD()
// namespace wust_vision {
// class vision {
// public:
//     vision() {
//         common_config_ = COMMON_CONFIG;
//         camera_config_ = CAMERA_CONFIG;
//         auto_aim_config_ = AUTO_AIM_CONFIG;
//         auto_buff_config_ = AUTO_BUFF_CONFIG;
//     };
//     ~vision() {
//         run_flag_ = false;
//         has_camera_info_ = true;
//         if (debug_thread_.joinable()) {
//             debug_thread_.join();
//         }

//         WUST_MAIN("main") << "vision stop already!";
//     }
//     bool init(bool debug_mode) {
//         try {
//             rclcpp::init(0, nullptr);
//             ros2_ = std::make_shared<Ros2Node>("vison_node");
//             ros2_->add_subscription<sensor_msgs::msg::Image>(
//                 "image_raw",
//                 std::bind(&vision::imageCallback, this, std::placeholders::_1),
//                 rclcpp::SensorDataQoS()
//             );

//             ros2_->add_subscription<sensor_msgs::msg::CameraInfo>(
//                 "camera_info",
//                 [this](sensor_msgs::msg::CameraInfo::ConstSharedPtr _camera_info) {
//                     if (has_camera_info_)
//                         return;
//                     std::cout << "camera info received" << std::endl;
//                     auto& msg = *_camera_info;

//                     cv::Mat K(3, 3, CV_64F);
//                     std::memcpy(K.data, msg.k.data(), 9 * sizeof(double));

//                     cv::Mat D(1, msg.d.size(), CV_64F);
//                     std::memcpy(D.data, msg.d.data(), msg.d.size() * sizeof(double));

//                     camera_info_ = std::make_pair(K.clone(), D.clone());
//                     has_camera_info_ = true;
//                 }
//             );
//             ros2_->start();
//             while (rclcpp::ok() && !has_camera_info_) {
//                 std::this_thread::sleep_for(std::chrono::milliseconds(1000));
//                 WUST_INFO("sim") << "Waiting for camera info...";
//             }

//             const char* v = std::getenv("VISION_ROOT");
//             if (v)
//                 std::cout << "[env] VISION_ROOT = " << v << "\n";
//             else
//                 std::cout << "[env] VISION_ROOT not set in this process\n";
//             debug_mode_ = debug_mode;

//             control_config_ = ControlConfig::create(this);
//             shoot_config_ = ShootConfig::create();
//             logger_config_ = LoggerConfig::create();
//             tf_config_ = TFConfig::create();
//             max_infer_running_config_ = MaxInferRunningConfig::create();
//             common_config_parameter_.registerGroup(*control_config_);
//             common_config_parameter_.registerGroup(*shoot_config_);
//             common_config_parameter_.registerGroup(*logger_config_);
//             common_config_parameter_.registerGroup(*tf_config_);
//             common_config_parameter_.registerGroup(*max_infer_running_config_);
//             common_config_parameter_.loadFromFile(common_config_);
//             auto config = common_config_parameter_.getConfig();
//             debug_fps_ = config["debug_fps"].as<int>();
//             attack_mode_ = config["attack_mode"].as<int>();
//             detect_color_ = config["detect_color"].as<int>();

//             wust_vl::common::utils::ParameterManager::instance().registerParameter(
//                 common_config_parameter_
//             );
//             auto_aim_ = std::make_unique<auto_aim::AutoAim>(
//                 auto_aim_config_,
//                 tf_config_,
//                 camera_info_,
//                 debug_mode
//             );
//             auto_buff_ = std::make_unique<auto_buff::AutoBuff>(
//                 auto_buff_config_,
//                 tf_config_,
//                 camera_info_,
//                 debug_mode
//             );
//             thread_pool_ = std::make_unique<wust_vl::common::concurrency::ThreadPool>(
//                 std::thread::hardware_concurrency() * 2
//             );
//             motion_buffer_ =
//                 std::make_shared<wust_vl::common::utils::MotionBufferGeneric<Motion, 1024>>();

//             timer_ = std::make_unique<wust_vl::common::utils::Timer>();
//         } catch (std::exception& e) {
//             std::cerr << "init exception: " << e.what() << std::endl;
//         }
//         return true;
//     }
//     void imageCallback(const sensor_msgs::msg::Image::ConstSharedPtr img_msg) {
//         if (!run_flag_) {
//             return;
//         }
//         cv::Mat image(
//             img_msg->height,
//             img_msg->width,
//             CV_8UC3,
//             const_cast<unsigned char*>(img_msg->data.data()), // raw pointer
//             img_msg->step
//         );
//         CommonFrame common_frame;
//         common_frame.img_frame.src_img = std::move(image);
//         common_frame.detect_color = detect_color_;
//         common_frame.img_frame.timestamp = std::chrono::steady_clock::now();
//         common_frame.img_frame.pixel_format = wust_vl::video::PixelFormat::BGR;
//         common_frame.expanded = cv::Rect(
//             0,
//             0,
//             common_frame.img_frame.src_img.cols,
//             common_frame.img_frame.src_img.rows
//         );
//         common_frame.any_ctx = VisionCtx { .motion_buffer = motion_buffer_,
//                                            .communication_delay_μs =
//                                                control_config_->communication_delay_us_param.get(),
//                                            .attack_mode = toAttackMode(attack_mode_) };
//         common_frame.offset = cv::Point2f(0, 0);
//         thread_pool_->enqueue([this, frame = std::move(common_frame)]() mutable {
//             infer_running_count_++;
//             if (frame.img_frame.src_img.empty()) {
//                 return;
//             }
//             AttackMode mode = toAttackMode(attack_mode_);
//             switch (mode) {
//                 case AttackMode::ARMOR: {
//                     auto_aim_->pushInput(frame);
//                 } break;
//                 case AttackMode::SMALL_RUNE: {
//                     auto_buff_->pushInput(frame);
//                 } break;
//                 case AttackMode::BIG_RUNE: {
//                     auto_buff_->pushInput(frame);
//                 } break;
//                 case AttackMode::UNKNOWN: {
//                     auto_aim_->pushInput(frame);
//                 } break;
//             }
//             infer_running_count_--;
//         });
//     }

//     void start() {
//         run_flag_ = true;
//         auto_aim_->start();
//         auto_buff_->start();
//         if (timer_) {
//             auto timercallback = std::bind(&vision::timerCallback, this, std::placeholders::_1);
//             double rate_hz = control_config_->control_rate_param.get();
//             timer_->start(rate_hz, timercallback);
//         }

//         if (debug_mode_) {
//             debug_thread_ = std::thread([this]() { this->debugThread(); });
//         }
//     }
//     void timerCallback(double dt_ms) {
//         if (!run_flag_) {
//             return;
//         }
//         geometry_msgs::msg::TransformStamped tf;
//         if (!ros2_->lookup_transform("odom", "gimbal_link", tf)) {
//             RCLCPP_WARN(ros2_->get_logger(), "TF lookup failed");
//         } else {
//             Eigen::Quaterniond q(
//                 tf.transform.rotation.w,
//                 tf.transform.rotation.x,
//                 tf.transform.rotation.y,
//                 tf.transform.rotation.z
//             );

//             // Convert to rotation matrix
//             Eigen::Matrix3d R = q.toRotationMatrix();

//             // Euler angles (ZYX order -> yaw pitch roll)
//             Eigen::Vector3d euler = R.eulerAngles(2, 1, 0); // yaw, pitch, roll

//             double yaw = euler[0];
//             double pitch = euler[1];
//             double roll = euler[2];
//             if (motion_buffer_) {
//                 Motion motion { yaw, -pitch, roll, 0.0, 0, 0, 0, 0, 0 };
//                 motion_buffer_->push(motion, std::chrono::steady_clock::now());
//             }
//         }
//         GimbalCmd cmd;
//         try {
//             AttackMode mode = toAttackMode(attack_mode_);
//             switch (mode) {
//                 case AttackMode::ARMOR: {
//                     cmd = auto_aim_->solve(shoot_config_->bullet_speed_param.get());
//                 } break;
//                 case AttackMode::SMALL_RUNE:
//                 case AttackMode::BIG_RUNE: {
//                     cmd = auto_buff_->solve(shoot_config_->bullet_speed_param.get());
//                 } break;
//                 case AttackMode::UNKNOWN: {
//                     cmd = auto_aim_->solve(shoot_config_->bullet_speed_param.get());
//                 } break;
//             }
//         } catch (const std::exception& e) {
//             std::cout << "auto_aim_solve error: " << e.what() << std::endl;
//         }

//         last_cmd_ = cmd;
//     }

//     void debugThread() {
//         const double us_interval = 1e6 / static_cast<double>(debug_fps_);
//         const auto kInterval = std::chrono::microseconds(static_cast<int64_t>(us_interval));
//         while (run_flag_) {
//             const auto start_time = std::chrono::steady_clock::now();
//             do {
//                 try {
//                     if (!auto_aim_ || !auto_buff_) {
//                         break;
//                     }
//                     auto dbg_armor = auto_aim_->getDebugFrame();
//                     auto dbg_rune = auto_buff_->getDebugFrame();
//                     AttackMode mode = toAttackMode(attack_mode_);
//                     switch (mode) {
//                         case AttackMode::UNKNOWN:
//                         case AttackMode::ARMOR: {
//                             drawDebugOverlayShm(dbg_armor, camera_info_, false);
//                         } break;
//                         case AttackMode::SMALL_RUNE:
//                         case AttackMode::BIG_RUNE: {
//                             drawDebugOverlayShm(dbg_rune, camera_info_, false);
//                         } break;
//                     }
//                     std::pair<double, double> gimbal_py;
//                     if (motion_buffer_) {
//                         auto last_att = motion_buffer_->get_last();
//                         if (last_att) {
//                             gimbal_py.first = last_att->data.pitch;
//                             gimbal_py.second = last_att->data.yaw;
//                         }
//                     }

//                     debuglog(dbg_armor, dbg_rune, last_cmd_, gimbal_py);
//                     utils::XSecOnce(
//                         [this]() {
//                             wust_vl::common::utils::ParameterManager::instance()
//                                 .allReloadFromOldPath();
//                         },
//                         1.0
//                     );

//                 } catch (std::exception& e) {
//                     std::cout << "debug thread error: " << e.what() << std::endl;
//                 }
//             } while (0);

//             const auto elapsed = std::chrono::steady_clock::now() - start_time;
//             if (elapsed < kInterval) {
//                 std::this_thread::sleep_for(kInterval - elapsed);
//             }
//         }
//     }
//     void checkStateMatchMode() {
//         AttackMode mode = toAttackMode(attack_mode_);
//         switch (mode) {
//             case AttackMode::ARMOR: {
//                 if (!auto_aim_->isActive()) {
//                     auto_aim_->processingUp();
//                 }
//                 if (auto_buff_->isActive()) {
//                     auto_buff_->processingWait();
//                 }

//             } break;
//             case AttackMode::SMALL_RUNE:
//             case AttackMode::BIG_RUNE: {
//                 if (auto_aim_->isActive()) {
//                     auto_aim_->processingWait();
//                 }
//                 if (!auto_buff_->isActive()) {
//                     auto_buff_->processingUp();
//                 }
//             } break;
//             case AttackMode::UNKNOWN: {
//                 if (!auto_aim_->isActive()) {
//                     auto_aim_->processingUp();
//                 }
//                 if (auto_buff_->isActive()) {
//                     auto_buff_->processingWait();
//                 }
//             } break;
//         }
//     }
//     struct ControlConfig: wust_vl::common::utils::ParamGroup {
//     public:
//         static constexpr const char* kKey = "control";
//         static constexpr const char* Logger = "Config: common::control";
//         const char* key() const override {
//             return kKey;
//         }
//         using Ptr = std::shared_ptr<ControlConfig>;
//         ControlConfig(vision* b) {
//             base = b;
//             communication_delay_us_param.onChange([this](int o, int n) {
//                 if (isBaseACtive()) {
//                     WUST_DEBUG(Logger)
//                         << "communication_delay_μs from: " << o << " to: " << n << " us";
//                 }
//             });
//         }
//         static Ptr create(vision* b) {
//             return std::make_shared<ControlConfig>(b);
//         }
//         GEN_PARAM(double, communication_delay_us);
//         GEN_PARAM(double, yaw_ramp);
//         GEN_PARAM(double, pitch_ramp);
//         GEN_PARAM(double, control_rate);
//         vision* base;
//         bool first_load = false;
//         bool isBaseACtive() {
//             return base != nullptr;
//         }
//         void loadSelf(const YAML::Node& node) override {
//             if (!isBaseACtive())
//                 return;
//             if (!first_load) {
//                 communication_delay_us_param.set(node["communication_delay_us"].as<double>());
//                 yaw_ramp_param.set(node["yaw_ramp"].as<double>());
//                 pitch_ramp_param.set(node["pitch_ramp"].as<double>());
//                 control_rate_param.set(node["control_rate"].as<double>());
//                 first_load = true;
//             } else {
//                 communication_delay_us_param.load(node);
//                 yaw_ramp_param.load(node);
//                 pitch_ramp_param.load(node);
//                 control_rate_param.load(node);
//             }
//         }
//     };
//     ControlConfig::Ptr control_config_;
//     struct ShootConfig: wust_vl::common::utils::ParamGroup {
//     public:
//         static constexpr const char* kKey = "shoot";
//         static constexpr const char* Logger = "Config: common::shoot";
//         const char* key() const override {
//             return kKey;
//         }
//         using Ptr = std::shared_ptr<ShootConfig>;
//         ShootConfig() {
//             rate_param.onChange([this](int o, int n) {
//                 WUST_DEBUG(Logger) << "shoot_rate from: " << o << " to: " << n << " HZ";
//             });
//         }

//         static Ptr create() {
//             return std::make_shared<ShootConfig>();
//         }
//         GEN_PARAM(int, rate);
//         GEN_PARAM(double, bullet_speed);
//         bool first_load = false;

//         void loadSelf(const YAML::Node& node) override {
//             if (!first_load) {
//                 rate_param.set(node["rate"].as<int>());
//                 bullet_speed_param.set(node["bullet_speed"].as<double>());
//                 first_load = true;
//             } else {
//                 rate_param.load(node);
//             }
//         }
//     };
//     ShootConfig::Ptr shoot_config_;

//     LoggerConfig::Ptr logger_config_;

//     TFConfig::Ptr tf_config_;

//     MaxInferRunningConfig::Ptr max_infer_running_config_;

//     int attack_mode_;
//     int debug_fps_;
//     bool detect_color_;

//     wust_vl::common::utils::Parameter common_config_parameter_;
//     std::unique_ptr<wust_vl::common::concurrency::ThreadPool> thread_pool_;
//     std::unique_ptr<auto_aim::AutoAim> auto_aim_;
//     std::unique_ptr<auto_buff::AutoBuff> auto_buff_;
//     std::unique_ptr<wust_vl::common::utils::Timer> timer_;
//     std::shared_ptr<wust_vl::common::utils::MotionBufferGeneric<Motion, 1024>> motion_buffer_;
//     std::thread debug_thread_;
//     GimbalCmd last_cmd_;
//     std::pair<cv::Mat, cv::Mat> camera_info_;
//     bool run_flag_ = false;
//     bool debug_mode_ = false;
//     std::atomic<int> infer_running_count_ { 0 };
//     std::string common_config_;
//     std::string camera_config_;
//     std::string auto_aim_config_;
//     std::string auto_buff_config_;
//     bool has_camera_info_ = false;
//     std::shared_ptr<Ros2Node> ros2_;
// };
// } // namespace wust_vision
// // VISION_MAIN(wust_vision::vision)
// int main(int argc, char** argv) {
//     wust_vision::printBanner();
//     bool debug = false;
//     if (argc > 1) {
//         std::string firstArg = argv[1];
//         debug = (firstArg == "true" || firstArg == "1");
//         std::cout << "debug: " << firstArg << std::endl;
//     }
//     std::set_terminate([]() {
//         std::cerr << "Uncaught exception, terminating program.\n";
//         if (auto e = std::current_exception()) {
//             try {
//                 std::rethrow_exception(e);
//             } catch (const std::exception& ex) {
//                 std::cerr << "Exception: " << ex.what() << std::endl;
//             } catch (...) {
//                 std::cerr << "Unknown exception" << std::endl;
//             }
//         }
//         std::abort();
//     });

//     try {
//         int exit_code = 0;

//         {
//             wust_vision::vision v;
//             v.init(debug);
//             v.start();

//             wust_vl::common::utils::SignalHandler sig;
//             sig.start([&] { rclcpp::shutdown(); });

//             bool exit_flag = false;

//             while (!sig.shouldExit() && !exit_flag) {
//                 wust_vl::common::concurrency::ThreadManager::instance().printStatus();
//                 auto all_status =
//                     wust_vl::common::concurrency::ThreadManager::instance().getAllThreadStatuses();
//                 v.checkStateMatchMode();

//                 for (auto& status: all_status) {
//                     if (status.second
//                         == wust_vl::common::concurrency::MonitoredThread::Status::Hung) {
//                         std::cerr << status.first << " is Hunging! Exiting program..." << std::endl;
//                         exit_flag = true;
//                         exit_code = -1;
//                         sig.requestExit();
//                         break;
//                     }
//                 }
//                 std::this_thread::sleep_for(std::chrono::milliseconds(1000));
//             }
//         }

//         std::cout << "Exiting program..." << std::endl;

//         return exit_code;

//     } catch (const std::exception& e) {
//         std::cerr << "Caught exception in main: " << e.what() << "\n";
//         throw;
//         return -1;
//     } catch (...) {
//         std::cerr << "Unknown exception caught in main!\n";
//         return -1;
//     }
// }