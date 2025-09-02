#include "3rdparty/backward-cpp/backward.hpp"
#include "ros2/ros2.hpp"
#include "tasks/auto_aim/auto_aim.hpp"
#include "tasks/auto_buff/auto_buff.hpp"
#include "tasks/utils.hpp"
#include "wust_vl/common/drivers/serial_driver.hpp"
#include "wust_vl/common/utils/config_binder.hpp"
#include <sys/mman.h>
#include <wust_vl/wust_vl.hpp>
#define COMMON_CONFIG "config/common.yaml"
#define CAMERA_CONFIG "config/camera.yaml"
#define AUTO_AIM_CONFIG "config/auto_aim.yaml"
#define AUTO_BUFF_CONFIG "config/auto_buff.yaml"

namespace backward {
backward::SignalHandling sh;
}
class vision {
public:
    void stop() {
        run_flag_ = false;
        if (camera_) {
            camera_->stop();
            camera_.reset();
        }
        if (auto_aim_) {
            auto_aim_.reset();
        }
        WUST_INFO("stop") << "auto_aim stop";

        if (timer_) {
            timer_.reset();
        }
        WUST_INFO("stop") << "timer stop";
        if (serial_) {
            serial_.reset();
        }
        WUST_INFO("stop") << "serial stop";
        if (debug_thread_.joinable()) {
            debug_thread_.join();
        }
#ifdef USE_NCNN
        if (use_ncnn_count_ > 0) {
            ncnn::destroy_gpu_instance();
        }
#endif
        WUST_MAIN("main") << "vision stop already!";
    }
    bool init() {
        config_ = YAML::LoadFile(COMMON_CONFIG);
        config_binder_ = std::make_shared<ConfigBinder>(COMMON_CONFIG);
        std::string log_level_ = config_["logger"]["log_level"].as<std::string>("INFO");
        std::string log_path_ = config_["logger"]["log_path"].as<std::string>("wust_log");
        bool use_logcli = config_["logger"]["use_logcli"].as<bool>();
        bool use_logfile = config_["logger"]["use_logfile"].as<bool>();
        bool use_simplelog = config_["logger"]["use_simplelog"].as<bool>();
        initLogger(log_level_, log_path_, use_logcli, use_logfile, use_simplelog);
        bindConfig(config_binder_, { "max_infer_running" }, &max_infer_running_);
        // bindConfig(config_binder_, {    "attack_mode"   }, &attack_mode_);
        attack_mode_ = config_["attack_mode"].as<int>();
        double gimbal2camera_roll = config_["tf"]["gimbal2camera_roll"].as<double>(0.0);
        double gimbal2camera_pitch = config_["tf"]["gimbal2camera_pitch"].as<double>(0.0);
        double gimbal2camera_yaw = config_["tf"]["gimbal2camera_yaw"].as<double>(0.0);
        gimbal2camera_yaw_ = gimbal2camera_yaw * M_PI / 180.0;
        gimbal2camera_pitch_ = gimbal2camera_pitch * M_PI / 180.0;
        gimbal2camera_roll_ = gimbal2camera_roll * M_PI / 180.0;
        double gimbal2camera_x_ = config_["tf"]["gimbal2camera_x"].as<double>(0.0);
        double gimbal2camera_y_ = config_["tf"]["gimbal2camera_y"].as<double>(0.0);
        double gimbal2camera_z_ = config_["tf"]["gimbal2camera_z"].as<double>(0.0);
        t_gimbal_to_camera_ = Eigen::Vector3d(gimbal2camera_x_, gimbal2camera_y_, gimbal2camera_z_);

        R_camera2gimbal_ << 0, 0, 1, -1, 0, 0, 0, -1, 0;
        YAML::Node camera_config = YAML::LoadFile(CAMERA_CONFIG);
        camera_ = std::make_unique<wust_vl_video::Camera>();
        camera_->init(camera_config);
        camera_->setFrameCallback(std::bind(&vision::frameCallback, this, std::placeholders::_1));
        const std::string camera_info_path = camera_config["camera_info_path"].as<std::string>();
        YAML::Node config_camera_info = YAML::LoadFile(camera_info_path);
        std::vector<double> camera_k =
            config_camera_info["camera_matrix"]["data"].as<std::vector<double>>();
        std::vector<double> camera_d =
            config_camera_info["distortion_coefficients"]["data"].as<std::vector<double>>();

        assert(camera_k.size() == 9);
        assert(camera_d.size() == 5);

        cv::Mat K(3, 3, CV_64F);
        std::memcpy(K.data, camera_k.data(), 9 * sizeof(double));

        cv::Mat D(1, 5, CV_64F);
        std::memcpy(D.data, camera_d.data(), 5 * sizeof(double));

        auto camera_info = std::make_pair(K.clone(), D.clone());
        camera_info_ = camera_info;

        YAML::Node auto_aim_config = YAML::LoadFile(AUTO_AIM_CONFIG);
        auto_aim_config_binder_ = std::make_shared<ConfigBinder>(AUTO_AIM_CONFIG);
        auto_aim_ = std::make_unique<auto_aim::AutoAim>();
        auto_aim_->init(
            auto_aim_config,
            use_ncnn_count_,
            R_camera2gimbal_,
            t_gimbal_to_camera_,
            camera_info,
            auto_aim_config_binder_
        );
        YAML::Node auto_buff_config = YAML::LoadFile(AUTO_BUFF_CONFIG);
        auto_buff_ = std::make_unique<auto_buff::AutoBuff>();
        auto_buff_->init(
            auto_buff_config,
            use_ncnn_count_,
            R_camera2gimbal_,
            t_gimbal_to_camera_,
            camera_info
        );
        thread_pool_ = std::make_unique<ThreadPool>(std::thread::hardware_concurrency());
        motion_buffer_ = std::make_shared<MotionBuffer>();
        auto_aim_shared_ = std::make_shared<auto_aim::AutoAimShared>(motion_buffer_);
        auto_aim_shared_->bullet_speed = config_["shoot"]["bullet_speed"].as<double>(20.0);
        auto_aim_shared_->controller_delay = config_["shoot"]["controller_delay"].as<double>(0.0);
        auto_aim_shared_->R_camera2gimbal = R_camera2gimbal_;
        auto_aim_shared_->t_gimbal_to_camera = t_gimbal_to_camera_;
        auto_aim_->setShared(auto_aim_shared_);
        auto_buff_shared_ = std::make_shared<auto_buff::AutoBuffShared>(motion_buffer_);
        auto_buff_shared_->bullet_speed = config_["shoot"]["bullet_speed"].as<double>(20.0);
        auto_buff_shared_->controller_delay = config_["shoot"]["controller_delay"].as<double>(0.0);
        auto_buff_shared_->R_camera2gimbal = R_camera2gimbal_;
        auto_buff_shared_->t_gimbal_to_camera = t_gimbal_to_camera_;
        auto_buff_->setShared(auto_buff_shared_);

        std::string device_name = config_["control"]["device_name"].as<std::string>();
        bindConfig(
            config_binder_,
            { "control", "communication_delay_us" },
            &communication_delay_μs_
        );
        bool use_serial = config_["control"]["use_serial"].as<bool>();
        if (use_serial) {
            SerialDriver::SerialPortConfig cfg {
                /*baud*/ 115200,
                /*csize*/ 8,
                boost::asio::serial_port_base::parity::none,
                boost::asio::serial_port_base::stop_bits::one,
                boost::asio::serial_port_base::flow_control::none
            };
            serial_ = std::make_shared<SerialDriver>();
            serial_->init_port(device_name, cfg);
            serial_->set_receive_callback(std::bind(
                &vision::serialCallback,
                this,
                std::placeholders::_1,
                std::placeholders::_2

            ));
            serial_->set_error_callback([&](const boost::system::error_code& ec) {
                WUST_ERROR("serial") << "serial error: " << ec.message();
            });
        }
        ros2_ = std::make_unique<Ros2>(std::bind(&vision::TwistCb, this, std::placeholders::_1));

        timer_ = std::make_unique<Timer>();
        detect_color_ = config_["detect_color"].as<int>(0);
        debug_mode_ = config_["debug_mode"].as<bool>(false);

        if (auto_aim_) {
            auto_aim_->setDebug(debug_mode_);
        }
        if (auto_buff_) {
            auto_buff_->setDebug(debug_mode_);
        }

        return true;
    }
    void serialCallback(const uint8_t* data, std::size_t len) {
        try {
            std::vector<uint8_t> buf(data, data + len);
            ReceiveAimINFO aim_data = fromVector<ReceiveAimINFO>(buf);

            if (std::isnan(aim_data.roll) || std::isnan(aim_data.pitch) || std::isnan(aim_data.yaw)
                || !this->run_flag_)
            {
                return;
            }

            double roll = -(aim_data.roll) * M_PI / 180.0;
            double pitch = (aim_data.pitch) * M_PI / 180.0;
            double yaw = (aim_data.yaw) * M_PI / 180.0;
            double v_x = aim_data.v_x;
            double v_y = aim_data.v_y;
            double v_z = aim_data.v_z;

            auto now = std::chrono::steady_clock::now();
            if (motion_buffer_) {
                motion_buffer_->push(
                    yaw + gimbal2camera_yaw_,
                    pitch + gimbal2camera_pitch_,
                    roll + gimbal2camera_roll_,
                    v_x,
                    v_y,
                    v_z,
                    now
                );
            }

            writeSerialLogToJson(aim_data);

        } catch (const std::exception& e) {
            std::cerr << "serialCallback exception: " << e.what() << std::endl;
        } catch (...) {
            std::cerr << "serialCallback unknown exception" << std::endl;
        }
    }

    void frameCallback(wust_vl_video::ImageFrame& frame) {
        if (!run_flag_ || infer_running_count_ >= max_infer_running_) {
            return;
        }
        CommonFrame common_frame;
        common_frame.timestamp = frame.timestamp;
        if (frame.src_img.empty()) {
            return;
        }
        common_frame.detect_color = detect_color_;
        common_frame.src_img = std::move(frame.src_img);
        infer_running_count_++;

        thread_pool_->enqueue([this, frame = std::move(common_frame)]() mutable {
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
                case AttackMode::SMALL_RUNE:
                case AttackMode::BIG_RUNE: {
                    auto_buff_->pushInput(frame);
                } break;
                case AttackMode::UNKNOWN: {
                    auto_aim_->pushInput(frame);
                } break;
            }
            infer_running_count_--;
        });
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
    void timerCallback(double dt_ms) {
        static Averager<double> yaw_avg(5);
        static Averager<double> pitch_avg(5);
        static double last_yaw = 0.0;
        static double last_pitch = 0.0;

        if (!run_flag_) {
            return;
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
        double cmd_pitch = cmd.pitch;
        double cmd_yaw = cmd.yaw;
        if (cmd.pitch >= 15.0) {
            cmd_pitch = 15.0;
        }
        const double max_delta_yaw = 5.0;
        const double max_delta_pitch = 1.0;
        double pitch_delta = cmd_pitch - last_pitch;
        double yaw_delta = cmd_yaw - last_yaw;
        if (pitch_delta > max_delta_pitch)
            pitch_delta = max_delta_pitch;
        if (pitch_delta < -max_delta_pitch)
            pitch_delta = -max_delta_pitch;
        if (yaw_delta > max_delta_yaw)
            yaw_delta = max_delta_yaw;
        if (yaw_delta < -max_delta_yaw)
            yaw_delta = -max_delta_yaw;
        yaw_avg.add(last_yaw + yaw_delta);
        pitch_avg.add(last_pitch + pitch_delta);
        last_pitch = last_pitch + pitch_delta;
        last_yaw = last_yaw + yaw_delta;
        SendRobotCmdData send_data;
        send_data.cmd_ID = ID_ROBOT_CMD;
        send_data.appear = cmd.appera;

        send_data.detect_color = detect_color_;
        send_data.distance = cmd.distance;
        send_data.fire = cmd.fire_advice;
        double avg_pitch = pitch_avg.average();
        double avg_yaw = yaw_avg.average();
        send_data.pitch = avg_pitch;
        send_data.yaw = avg_yaw;
        send_data.pitch_diff = cmd.pitch_diff;
        send_data.yaw_diff = cmd.yaw_diff;
        send_data.v_pitch = cmd.v_pitch;
        send_data.v_yaw = cmd.v_yaw;
        send_data.enable_pitch_diff = cmd.enable_pitch_diff;
        send_data.enable_yaw_diff = cmd.enable_yaw_diff;

        if (serial_) {
            serial_->write(std::move(toVector(send_data)));
        }
    }

    void start() {
        run_flag_ = true;
        camera_->start();
        auto_aim_->start();
        auto_buff_->start();
        if (timer_) {
            auto timercallback = std::bind(&vision::timerCallback, this, std::placeholders::_1);
            double rate_hz = static_cast<double>(config_["control"]["control_rate"].as<int>());
            timer_->start(rate_hz, timercallback);
        }
        if (serial_) {
            serial_->start();
        }
        if (debug_mode_) {
            debug_thread_ = std::thread([this]() { this->debugThread(); });
        }
    }
    void TwistCb(const geometry_msgs::msg::Twist::SharedPtr msg) {
        NavRobotCmdData send_data;
        send_data.cmd_ID = ID_NAV_CMD;
        send_data.check = true;
        send_data.time_stamp =
            static_cast<uint32_t>(std::chrono::duration_cast<std::chrono::milliseconds>(
                                      time_utils::now().time_since_epoch()
            )
                                      .count());
        send_data.vx = msg->linear.x;
        send_data.vy = msg->linear.y;
        send_data.wz = msg->angular.z;
        if (serial_) {
            serial_->write(std::move(toVector(send_data)));
        }
    }
    std::thread debug_thread_;
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
                    gimbal_py.first = last_att->pitch;
                    gimbal_py.second = last_att->yaw;
                }
                debuglog(dbg_armor, dbg_rune, last_cmd_, gimbal_py);
                config_binder_->reload(COMMON_CONFIG);
                auto_aim_config_binder_->reload(AUTO_AIM_CONFIG);
            } catch (std::exception& e) {
                std::cout << "debug thread error: " << e.what() << std::endl;
            }

            auto elapsed = steady_clock::now() - start_time;
            if (elapsed < kInterval) {
                std::this_thread::sleep_for(kInterval - elapsed);
            }
        }
    }
    std::unique_ptr<ThreadPool> thread_pool_;
    std::unique_ptr<Ros2> ros2_;
    std::unique_ptr<auto_aim::AutoAim> auto_aim_;
    std::unique_ptr<auto_buff::AutoBuff> auto_buff_;
    std::unique_ptr<wust_vl_video::Camera> camera_;
    std::shared_ptr<SerialDriver> serial_;
    std::unique_ptr<Timer> timer_;
    std::shared_ptr<auto_aim::AutoAimShared> auto_aim_shared_;
    std::shared_ptr<auto_buff::AutoBuffShared> auto_buff_shared_;
    std::shared_ptr<MotionBuffer> motion_buffer_;
    std::shared_ptr<ConfigBinder> config_binder_;
    std::shared_ptr<ConfigBinder> auto_aim_config_binder_;
    YAML::Node config_;
    Eigen::Matrix3d R_camera2gimbal_;
    Eigen::Vector3d t_gimbal_to_camera_;
    double communication_delay_μs_;
    GimbalCmd last_cmd_;
    double gimbal2camera_roll_;
    double gimbal2camera_pitch_;
    double gimbal2camera_yaw_;

    std::pair<cv::Mat, cv::Mat> camera_info_;
    int attack_mode_;
    int max_infer_running_;
    bool run_flag_ = true;
    int detect_color_ = 0;
    bool debug_mode_ = false;
    int use_ncnn_count_ = 0;
    std::atomic<int> infer_running_count_ { 0 };
};
int main(int argc, char** argv) {
    std::set_terminate([]() {
        std::cerr << "Uncaught exception, terminating program.\n";
        std::abort();
    });

    try {
        vision v;
        v.init();
        v.start();

        SignalHandler sig;
        sig.start([&] { v.stop(); });

        bool exit_flag = false;
        int exit_code = 0;

        while (!sig.shouldExit() && !exit_flag) {
            ThreadManager::instance().printStatus();
            auto all_status = ThreadManager::instance().getAllThreadStatuses();
            v.checkStateMatchMode();

            for (auto& status: all_status) {
                if (status.second == MonitoredThread::Status::Hung) {
                    std::cerr << status.first << " is Hunging! Exiting program..." << std::endl;
                    exit_flag = true;
                    exit_code = -1;
                    break;
                }
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(1000));
        }

        auto stop_future = std::async(std::launch::async, [&v]() { v.stop(); });
        if (stop_future.wait_for(std::chrono::seconds(5)) == std::future_status::timeout) {
            std::cerr << "v.stop() timed out, forcing exit!" << std::endl;
            std::exit(1);
        }

        return exit_code;

    } catch (const std::exception& e) {
        std::cerr << "Caught exception in main: " << e.what() << "\n";
        return -1;
    } catch (...) {
        std::cerr << "Unknown exception caught in main!\n";
        return -1;
    }
}