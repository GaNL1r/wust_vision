#include "3rdparty/backward-cpp/backward.hpp"
#include "ros2/ros2.hpp"
#include "sensor_msgs/msg/camera_info.hpp"
#include "sensor_msgs/msg/image.hpp"
#include "tasks/vision_base.hpp"

ENABLE_BACKWARD()
class vision {
public:
    vision() {
        common_config_ = COMMON_CONFIG;
        camera_config_ = CAMERA_CONFIG;
        auto_aim_config_ = AUTO_AIM_CONFIG;
        auto_buff_config_ = AUTO_BUFF_CONFIG;
    };
    ~vision() {
        run_flag_ = false;
        has_camera_info_ = true;
        if (debug_thread_.joinable()) {
            debug_thread_.join();
        }
        if (thread_pool_)
            thread_pool_->waitUntilEmpty();

#ifdef USE_NCNN
        if (use_ncnn_count_ > 0) {
            ncnn::destroy_gpu_instance();
        }
#endif
        if (ros2_) {
            ros2_.reset();
        }

        WUST_MAIN("main") << "vision stop already!";
    }
    bool init(bool debug_mode) {
        rclcpp::init(0, nullptr);
        ros2_ = std::make_shared<Ros2Node>("vison_node");
        ros2_->add_subscription<sensor_msgs::msg::Image>(
            "image_raw",
            std::bind(&vision::imageCallback, this, std::placeholders::_1),
            rclcpp::SensorDataQoS()
        );

        ros2_->add_subscription<sensor_msgs::msg::CameraInfo>(
            "camera_info",
            [this](sensor_msgs::msg::CameraInfo::ConstSharedPtr _camera_info) {
                if (has_camera_info_)
                    return;
                std::cout << "camera info received" << std::endl;
                auto& msg = *_camera_info;

                cv::Mat K(3, 3, CV_64F);
                std::memcpy(K.data, msg.k.data(), 9 * sizeof(double));

                cv::Mat D(1, msg.d.size(), CV_64F);
                std::memcpy(D.data, msg.d.data(), msg.d.size() * sizeof(double));

                camera_info_ = std::make_pair(K.clone(), D.clone());
                has_camera_info_ = true;
            }
        );
        ros2_->start();
        while (rclcpp::ok() && !has_camera_info_) {
            std::this_thread::sleep_for(std::chrono::milliseconds(1000));
            WUST_INFO("sim") << "Waiting for camera info...";
        }

        const char* v = std::getenv("VISION_ROOT");
        if (v)
            std::cout << "[env] VISION_ROOT = " << v << "\n";
        else
            std::cout << "[env] VISION_ROOT not set in this process\n";
        debug_mode_ = debug_mode;
        config_ = YAML::LoadFile(common_config_);

        std::string log_level_ = config_["logger"]["log_level"].as<std::string>("INFO");
        std::string log_path_ = config_["logger"]["log_path"].as<std::string>("wust_log");
        bool use_logcli = config_["logger"]["use_logcli"].as<bool>();
        bool use_logfile = config_["logger"]["use_logfile"].as<bool>();
        bool use_simplelog = config_["logger"]["use_simplelog"].as<bool>();
        initLogger(log_level_, log_path_, use_logcli, use_logfile, use_simplelog);
        max_infer_running_ = config_["max_infer_running"].as<int>(1);
        attack_mode_ = config_["attack_mode"].as<int>();
        auto t_vec = config_["tf"]["t_camera2gimbal"].as<std::vector<double>>();
        if (t_vec.size() != 3) {
            throw std::runtime_error("YAML tf.t_camera2gimbal must have 3 elements");
        }
        t_camera2gimbal_ = Eigen::Vector3d(t_vec[0], t_vec[1], t_vec[2]);

        auto R_vec = config_["tf"]["R_camera2gimbal"].as<std::vector<double>>();
        if (R_vec.size() != 9) {
            throw std::runtime_error("YAML tf.R_camera2gimbal must have 9 elements");
        }
        R_camera2gimbal_ =
            Eigen::Map<const Eigen::Matrix<double, 3, 3, Eigen::RowMajor>>(R_vec.data());

        YAML::Node auto_aim_config = YAML::LoadFile(auto_aim_config_);
        auto_aim_ = std::make_unique<auto_aim::AutoAim>();
        auto_aim_->setDebug(debug_mode_);
        auto_aim_->init(
            auto_aim_config,
            use_ncnn_count_,
            R_camera2gimbal_,
            t_camera2gimbal_,
            camera_info_
        );
        YAML::Node auto_buff_config = YAML::LoadFile(auto_buff_config_);
        auto_buff_ = std::make_unique<auto_buff::AutoBuff>();
        auto_buff_->setDebug(debug_mode);
        auto_buff_->init(
            auto_buff_config,
            use_ncnn_count_,
            R_camera2gimbal_,
            t_camera2gimbal_,
            camera_info_,
            max_infer_running_
        );
        thread_pool_ = std::make_unique<wust_vl::common::concurrency::ThreadPool>(
            std::thread::hardware_concurrency() * 2
        );
        motion_buffer_ =
            std::make_shared<wust_vl::common::utils::MotionBufferGeneric<Motion, 1024>>();
        double bullet_speed = config_["shoot"]["bullet_speed"].as<double>(20.0);
        shoot_rate_ = config_["shoot"]["rate"].as<int>(3);
        double communication_delay_μs =
            config_["control"]["communication_delay_us"].as<double>(1000.0);

        auto_aim_shared_ = std::make_shared<auto_aim::AutoAimShared>(
            motion_buffer_,
            bullet_speed,
            communication_delay_μs
        );
        auto_aim_->setShared(auto_aim_shared_);
        auto_buff_shared_ = std::make_shared<auto_buff::AutoBuffShared>(
            motion_buffer_,
            bullet_speed,
            communication_delay_μs
        );
        auto_buff_->setShared(auto_buff_shared_);
        timer_ = std::make_unique<wust_vl::common::utils::Timer>();
        detect_color_ = config_["detect_color"].as<int>(0);
        return true;
    }
    void imageCallback(const sensor_msgs::msg::Image::ConstSharedPtr img_msg) {
        if (!run_flag_) {
            return;
        }
        cv::Mat image(
            img_msg->height,
            img_msg->width,
            CV_8UC3,
            const_cast<unsigned char*>(img_msg->data.data()), // raw pointer
            img_msg->step
        );
        CommonFrame common_frame;
        // cv::GaussianBlur(image, common_frame.src_img, cv::Size(5, 5), 5);
        common_frame.src_img = std::move(image);
        common_frame.detect_color = detect_color_;
        common_frame.timestamp = std::chrono::steady_clock::now();
        common_frame.expanded =
            cv::Rect(0, 0, common_frame.src_img.cols, common_frame.src_img.rows);
        common_frame.offset = cv::Point2f(0, 0);
        thread_pool_->enqueue([this, frame = std::move(common_frame)]() mutable {
            infer_running_count_++;
            if (frame.src_img.data == nullptr) {
                return;
            }
            if (frame.src_img.empty()) {
                return;
            }
            AttackMode mode = toAttackMode(attack_mode_);
            switch (mode) {
                case AttackMode::ARMOR: {
                    auto_aim_->pushInput(frame);
                } break;
                case AttackMode::SMALL_RUNE: {
                    auto_buff_->pushInput(frame, false);
                } break;
                case AttackMode::BIG_RUNE: {
                    auto_buff_->pushInput(frame, true);
                } break;
                case AttackMode::UNKNOWN: {
                    auto_aim_->pushInput(frame);
                } break;
            }
            infer_running_count_--;
        });
    }

    void start() {
        run_flag_ = true;
        auto_aim_->start();
        auto_buff_->start();
        if (timer_) {
            auto timercallback = std::bind(&vision::timerCallback, this, std::placeholders::_1);
            double rate_hz = static_cast<double>(config_["control"]["control_rate"].as<int>());
            timer_->start(rate_hz, timercallback);
        }

        if (debug_mode_) {
            debug_thread_ = std::thread([this]() { this->debugThread(); });
        }
    }
    void timerCallback(double dt_ms) {
        if (!run_flag_) {
            return;
        }
        geometry_msgs::msg::TransformStamped tf;
        if (!ros2_->lookup_transform("odom", "gimbal_link", tf)) {
            RCLCPP_WARN(ros2_->get_logger(), "TF lookup failed");

        } else {
            Eigen::Quaterniond q(
                tf.transform.rotation.w,
                tf.transform.rotation.x,
                tf.transform.rotation.y,
                tf.transform.rotation.z
            );

            // Convert to rotation matrix
            Eigen::Matrix3d R = q.toRotationMatrix();

            // Euler angles (ZYX order -> yaw pitch roll)
            Eigen::Vector3d euler = R.eulerAngles(2, 1, 0); // yaw, pitch, roll

            double yaw = euler[0];
            double pitch = euler[1];
            double roll = euler[2];
            if (motion_buffer_) {
                Motion motion { yaw, -pitch, roll, 0.0, 0, 0, 0, 0, 0 };
                motion_buffer_->push(motion, std::chrono::steady_clock::now());
            }
        }
        GimbalCmd cmd;
        try {
            AttackMode mode = toAttackMode(attack_mode_);
            switch (mode) {
                case AttackMode::ARMOR: {
                    cmd = auto_aim_->solve(dt_ms);
                } break;
                case AttackMode::SMALL_RUNE:
                case AttackMode::BIG_RUNE: {
                    cmd = auto_buff_->solve();
                } break;
                case AttackMode::UNKNOWN: {
                    cmd = auto_aim_->solve(dt_ms);
                } break;
            }
        } catch (const std::exception& e) {
            std::cout << "auto_aim_solve error: " << e.what() << std::endl;
        }

        last_cmd_ = cmd;
    }

    void debugThread() {
        using namespace std::chrono;

        double us_interval = 1e6 / static_cast<double>(30.0);
        auto kInterval = std::chrono::microseconds(static_cast<int64_t>(us_interval));
        while (run_flag_) {
            auto start_time = steady_clock::now();
            try {
                auto dbg_armor = auto_aim_->getDebugFrame();
                auto dbg_rune = auto_buff_->getDebugFrame();
                AttackMode mode = toAttackMode(attack_mode_);
                switch (mode) {
                    case AttackMode::ARMOR: {
                        drawDebugOverlayShm(dbg_armor, camera_info_, false);
                    } break;
                    case AttackMode::SMALL_RUNE:
                    case AttackMode::BIG_RUNE: {
                        drawDebugOverlayShm(dbg_rune, camera_info_, false);
                    } break;
                    case AttackMode::UNKNOWN: {
                        drawDebugOverlayShm(dbg_armor, camera_info_, false);
                    } break;
                }
                auto last_att = motion_buffer_->get_last();
                std::pair<double, double> gimbal_py;
                if (last_att) {
                    gimbal_py.first = last_att->data.pitch;
                    gimbal_py.second = last_att->data.yaw;
                }
                debuglog(dbg_armor, dbg_rune, last_cmd_, gimbal_py);
            } catch (std::exception& e) {
                std::cout << "debug thread error: " << e.what() << std::endl;
            }

            auto elapsed = steady_clock::now() - start_time;
            if (elapsed < kInterval) {
                std::this_thread::sleep_for(kInterval - elapsed);
            }
        }
    }
    void checkStateMatchMode() {
        AttackMode mode = toAttackMode(attack_mode_);
        switch (mode) {
            case AttackMode::ARMOR: {
                if (!auto_aim_->isActive()) {
                    auto_aim_->processingUp();
                }
                if (auto_buff_->isActive()) {
                    auto_buff_->processingWait();
                }

            } break;
            case AttackMode::SMALL_RUNE:
            case AttackMode::BIG_RUNE: {
                if (auto_aim_->isActive()) {
                    auto_aim_->processingWait();
                }
                if (!auto_buff_->isActive()) {
                    auto_buff_->processingUp();
                }
            } break;
            case AttackMode::UNKNOWN: {
                if (!auto_aim_->isActive()) {
                    auto_aim_->processingUp();
                }
                if (auto_buff_->isActive()) {
                    auto_buff_->processingWait();
                }
            } break;
        }
    }
    std::unique_ptr<wust_vl::common::concurrency::ThreadPool> thread_pool_;
    std::unique_ptr<auto_aim::AutoAim> auto_aim_;
    std::unique_ptr<auto_buff::AutoBuff> auto_buff_;
    std::unique_ptr<wust_vl::common::utils::Timer> timer_;
    std::shared_ptr<auto_aim::AutoAimShared> auto_aim_shared_;
    std::shared_ptr<auto_buff::AutoBuffShared> auto_buff_shared_;
    std::shared_ptr<wust_vl::common::utils::MotionBufferGeneric<Motion, 1024>> motion_buffer_;
    std::thread debug_thread_;
    GimbalCmd last_cmd_;
    std::pair<cv::Mat, cv::Mat> camera_info_;
    double bullet_speed_;
    int attack_mode_;
    int max_infer_running_;
    bool run_flag_ = false;
    int detect_color_ = 0;
    bool debug_mode_ = false;
    int use_ncnn_count_ = 0;
    int shoot_rate_ = 3;
    std::atomic<int> infer_running_count_ { 0 };
    std::string common_config_;
    std::string camera_config_;
    std::string auto_aim_config_;
    std::string auto_buff_config_;
    YAML::Node config_;
    Eigen::Matrix3d R_camera2gimbal_;
    Eigen::Vector3d t_camera2gimbal_;
    bool has_camera_info_ = false;
    std::shared_ptr<Ros2Node> ros2_;
};

// VISION_MAIN(vision)
int main(int argc, char** argv) {
    bool debug = false;
    if (argc > 1) {
        std::string firstArg = argv[1];
        debug = (firstArg == "true" || firstArg == "1");
        std::cout << "debug: " << firstArg << std::endl;
    }
    std::set_terminate([]() {
        std::cerr << "Uncaught exception, terminating program.\n";
        if (auto e = std::current_exception()) {
            try {
                std::rethrow_exception(e);
            } catch (const std::exception& ex) {
                std::cerr << "Exception: " << ex.what() << std::endl;
            } catch (...) {
                std::cerr << "Unknown exception" << std::endl;
            }
        }
        std::abort();
    });

    try {
        int exit_code = 0;

        {
            vision v;
            v.init(debug);
            v.start();

            wust_vl::common::utils::SignalHandler sig;
            sig.start([&] { rclcpp::shutdown(); });

            bool exit_flag = false;

            while (!sig.shouldExit() && !exit_flag) {
                wust_vl::common::concurrency::ThreadManager::instance().printStatus();
                auto all_status =
                    wust_vl::common::concurrency::ThreadManager::instance().getAllThreadStatuses();
                v.checkStateMatchMode();

                for (auto& status: all_status) {
                    if (status.second
                        == wust_vl::common::concurrency::MonitoredThread::Status::Hung) {
                        std::cerr << status.first << " is Hunging! Exiting program..." << std::endl;
                        exit_flag = true;
                        exit_code = -1;
                        sig.requestExit();
                        break;
                    }
                }
                std::this_thread::sleep_for(std::chrono::milliseconds(1000));
            }
        }

        std::cout << "Exiting program..." << std::endl;

        return exit_code;

    } catch (const std::exception& e) {
        std::cerr << "Caught exception in main: " << e.what() << "\n";
        throw;
        return -1;
    } catch (...) {
        std::cerr << "Unknown exception caught in main!\n";
        return -1;
    }
}