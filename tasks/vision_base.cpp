

#include "vision_base.hpp"
VisionBase::VisionBase(
    std::string common_config,
    std::string camera_config,
    std::string auto_aim_config,
    std::string auto_buff_config
):
    common_config_(common_config),
    camera_config_(camera_config),
    auto_aim_config_(auto_aim_config),
    auto_buff_config_(auto_buff_config) {}
VisionBase::~VisionBase() {
    stop();
}
void VisionBase::stop() {
    run_flag_ = false;
    if (camera_) {
        camera_->stop();
        camera_.reset();
    }
    if (auto_aim_) {
        auto_aim_.reset();
    }
    WUST_INFO("stop") << "auto_aim stop";
    if (auto_buff_) {
        auto_buff_.reset();
    }
    WUST_INFO("stop") << "auto_buff stop";
    if (timer_) {
        timer_.reset();
    }
    WUST_INFO("stop") << "timer stop";
    if (serial_) {
        serial_->stop();
        serial_.reset();
    }
    if (motion_buffer_) {
        motion_buffer_.reset();
    }
    if (auto_aim_shared_) {
        auto_aim_shared_.reset();
    }
    if (auto_buff_shared_) {
        auto_buff_shared_.reset();
    }
    WUST_INFO("stop") << "serial stop";
    if (debug_thread_.joinable()) {
        debug_thread_.join();
    }
    if (thread_pool_) {
        thread_pool_.reset();
    }

#ifdef USE_NCNN
    if (use_ncnn_count_ > 0) {
        ncnn::destroy_gpu_instance();
    }
#endif
    WUST_MAIN("main") << "vision stop already!";
}
bool VisionBase::init(bool debug_mode) {
    debug_mode_ = debug_mode;
    config_ = YAML::LoadFile(common_config_);
    config_binder_ = std::make_shared<wust_vl_utils::ConfigBinder>(common_config_);
    std::string log_level_ = config_["logger"]["log_level"].as<std::string>("INFO");
    std::string log_path_ = config_["logger"]["log_path"].as<std::string>("wust_log");
    bool use_logcli = config_["logger"]["use_logcli"].as<bool>();
    bool use_logfile = config_["logger"]["use_logfile"].as<bool>();
    bool use_simplelog = config_["logger"]["use_simplelog"].as<bool>();
    initLogger(log_level_, log_path_, use_logcli, use_logfile, use_simplelog);
    bindConfig(config_binder_, { "max_infer_running" }, &max_infer_running_);
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
    R_camera2gimbal_ = Eigen::Map<const Eigen::Matrix<double, 3, 3, Eigen::RowMajor>>(R_vec.data());

    YAML::Node camera_config = YAML::LoadFile(camera_config_);
    camera_ = std::make_unique<wust_vl_video::Camera>();
    camera_->init(camera_config);
    camera_->setFrameCallback(std::bind(&VisionBase::frameCallback, this, std::placeholders::_1));
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

    YAML::Node auto_aim_config = YAML::LoadFile(auto_aim_config_);
    auto_aim_config_binder_ = std::make_shared<wust_vl_utils::ConfigBinder>(auto_aim_config_);
    auto_aim_ = std::make_unique<auto_aim::AutoAim>();
    auto_aim_->init(
        auto_aim_config,
        use_ncnn_count_,
        R_camera2gimbal_,
        t_camera2gimbal_,
        camera_info,
        auto_aim_config_binder_
    );
    YAML::Node auto_buff_config = YAML::LoadFile(auto_buff_config_);
    auto_buff_ = std::make_unique<auto_buff::AutoBuff>();
    auto_buff_->init(
        auto_buff_config,
        use_ncnn_count_,
        R_camera2gimbal_,
        t_camera2gimbal_,
        camera_info,
        max_infer_running_
    );
    thread_pool_ = std::make_unique<ThreadPool>(std::thread::hardware_concurrency());
    motion_buffer_ = std::make_shared<MotionBufferGeneric<Motion, 1024>>();
    double bullet_speed = config_["shoot"]["bullet_speed"].as<double>(20.0);
    double communication_delay_μs = config_["control"]["communication_delay_us"].as<double>(1000.0);
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

    std::string device_name = config_["control"]["device_name"].as<std::string>();

    serial_ = std::make_shared<SerialDriver>();
    bool use_serial = config_["control"]["use_serial"].as<bool>();
    if (use_serial) {
        SerialDriver::SerialPortConfig cfg { /*baud*/ 115200,
                                             /*csize*/ 8,
                                             boost::asio::serial_port_base::parity::none,
                                             boost::asio::serial_port_base::stop_bits::one,
                                             boost::asio::serial_port_base::flow_control::none };
        serial_->init_port(device_name, cfg);
        serial_->set_receive_callback(std::bind(
            &VisionBase::serialCallback,
            this,
            std::placeholders::_1,
            std::placeholders::_2

        ));
        serial_->set_error_callback([&](const boost::system::error_code& ec) {
            WUST_ERROR("serial") << "serial error: " << ec.message();
        });
    }

    double yaw_avg_windows = config_["control"]["yaw_avg_windows"].as<double>(0.0);
    double pitch_avg_windows = config_["control"]["pitch_avg_windows"].as<double>(0.0);
    yaw_avg_ = std::make_unique<Averager<double>>(yaw_avg_windows);
    pitch_avg_ = std::make_unique<Averager<double>>(pitch_avg_windows);
    timer_ = std::make_unique<Timer>();
    detect_color_ = config_["detect_color"].as<int>(0);
    // debug_mode_ = config_["debug_mode"].as<bool>(false);
    auto_exposure_cfg_.loadFromYaml(config_["auto_exposure"]);
    if (auto_aim_) {
        auto_aim_->setDebug(debug_mode_);
    }
    if (auto_buff_) {
        auto_buff_->setDebug(debug_mode_);
    }

    return true;
}
void VisionBase::updateBulletSpeed(double bullet_speed) {
    if (auto_aim_shared_) {
        auto_aim_shared_->bullet_speed = bullet_speed;
    }
    if (auto_buff_shared_) {
        auto_buff_shared_->bullet_speed = bullet_speed;
    }
    bullet_speed_ = bullet_speed;
}
void VisionBase::serialCallback(const uint8_t* data, std::size_t len) {
    static Averager<double> vyaw_avg(100);
    if (len != sizeof(ReceiveAimINFO)) {
        return;
    }
    try {
        std::vector<uint8_t> buf(data, data + len);
        ReceiveAimINFO aim_data = fromVector<ReceiveAimINFO>(buf);

        if (std::isnan(aim_data.roll) || std::isnan(aim_data.pitch) || std::isnan(aim_data.yaw)
            || !this->run_flag_)
        {
            return;
        }
        //detect_color_ = aim_data.detect_color;
        double roll = -(aim_data.roll) * M_PI / 180.0;
        double pitch = (aim_data.pitch) * M_PI / 180.0;
        double yaw = (aim_data.yaw) * M_PI / 180.0;
        double v_roll = aim_data.roll_vel * M_PI / 180.0;
        double v_pitch = aim_data.pitch_vel * M_PI / 180.0;
        double v_yaw = aim_data.yaw_vel * M_PI / 180.0;
        vyaw_avg.add(v_yaw);
        updateBulletSpeed(aim_data.bullet_speed);
        double v_x = 0.0;
        double v_y = 0.0;
        double v_z = 0.0;

        auto now = std::chrono::steady_clock::now();
        if (motion_buffer_) {
            Motion motion { yaw, pitch, roll, 0.0, v_pitch, v_roll, v_x, v_y, v_z };
            motion_buffer_->push(motion, now);
        }

        writeSerialLogToJson(aim_data);

    } catch (const std::exception& e) {
        std::cerr << "serialCallback exception: " << e.what() << std::endl;
    } catch (...) {
        std::cerr << "serialCallback unknown exception" << std::endl;
    }
}
double computeBrightness(const cv::Mat& frame) {
    cv::Mat gray;
    cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);
    return cv::mean(gray)[0];
}
void VisionBase::autoExposureControl(const cv::Mat& frame) {
    if (!auto_exposure_cfg_.enable) {
        return;
    }
    static auto last_update = std::chrono::steady_clock::now();

    auto now = std::chrono::steady_clock::now();
    double elapsed_ms =
        std::chrono::duration_cast<std::chrono::milliseconds>(now - last_update).count();

    if (elapsed_ms >= auto_exposure_cfg_.control_interval_ms) {
        double brightness = computeBrightness(frame);
        double diff = brightness - auto_exposure_cfg_.target_brightness;
        const double exposure_min = auto_exposure_cfg_.exposure_min;
        const double exposure_max = auto_exposure_cfg_.exposure_max;
        double exposure_time = camera_->getHikExposureTime();
        if (fabs(diff) > auto_exposure_cfg_.tolerance && exposure_time > 0.0) {
            exposure_time -= diff * auto_exposure_cfg_.step_gain;
        } else {
            exposure_time -= auto_exposure_cfg_.decay_step;
        }
        if (exposure_time < exposure_min)
            exposure_time = exposure_min;
        if (exposure_time > exposure_max)
            exposure_time = exposure_max;
        camera_->setHikExposureTime(exposure_time);
        last_update = now;
    }
}
void VisionBase::frameCallback(wust_vl_video::ImageFrame& frame) {
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
    autoExposureControl(common_frame.src_img);

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
void VisionBase::checkStateMatchMode() {
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

void VisionBase::timerCallback(double dt_ms) {
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
    if (cmd.pitch >= 35.0) {
        cmd_pitch = 35.0;
    }
    const double max_delta_yaw = 5.0;
    const double max_delta_pitch = 1.0;
    double pitch_delta = cmd_pitch - last_pitch;
    double yaw_delta = cmd_yaw - last_yaw;
    if (yaw_delta > 180.0)
        yaw_delta -= 360.0;
    if (yaw_delta < -180.0)
        yaw_delta += 360.0;
    if (pitch_delta > max_delta_pitch)
        pitch_delta = max_delta_pitch;
    if (pitch_delta < -max_delta_pitch)
        pitch_delta = -max_delta_pitch;
    if (yaw_delta > max_delta_yaw)
        yaw_delta = max_delta_yaw;
    if (yaw_delta < -max_delta_yaw)
        yaw_delta = -max_delta_yaw;
    yaw_avg_->add(last_yaw + yaw_delta);
    pitch_avg_->add(last_pitch + pitch_delta);
    last_pitch = last_pitch + pitch_delta;
    last_yaw = last_yaw + yaw_delta;
    SendRobotCmdData send_data;
    send_data.cmd_ID = ID_ROBOT_CMD;
    if (cmd.distance > 0.5) {
        send_data.appear = cmd.appera;
    } else {
        send_data.appear = false;
    }

    send_data.detect_color = detect_color_;
    double avg_pitch = pitch_avg_->average();
    double avg_yaw = yaw_avg_->average();
    send_data.pitch = avg_pitch;
    send_data.yaw = avg_yaw;
    send_data.v_pitch = cmd.v_pitch;
    send_data.v_yaw = cmd.v_yaw;
    send_data.target_yaw = cmd.target_yaw;
    send_data.target_pitch = cmd.target_pitch;
    send_data.enable_pitch_diff = cmd.enable_pitch_diff;
    send_data.enable_yaw_diff = cmd.enable_yaw_diff;

    if (serial_) {
        serial_->write(std::move(toVector(send_data)));
    }
}

void VisionBase::start() {
    run_flag_ = true;
    camera_->start();
    auto_aim_->start();
    auto_buff_->start();
    if (timer_) {
        auto timercallback = std::bind(&VisionBase::timerCallback, this, std::placeholders::_1);
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

void VisionBase::debugThread() {
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
                gimbal_py.second = last_att->data.vyaw;
            }
            debuglog(dbg_armor, dbg_rune, last_cmd_, gimbal_py);
            config_binder_->reload(common_config_);
            auto_aim_config_binder_->reload(auto_aim_config_);
        } catch (std::exception& e) {
            std::cout << "debug thread error: " << e.what() << std::endl;
        }

        auto elapsed = steady_clock::now() - start_time;
        if (elapsed < kInterval) {
            std::this_thread::sleep_for(kInterval - elapsed);
        }
    }
}