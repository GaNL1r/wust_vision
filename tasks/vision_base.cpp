#include "vision_base.hpp"
#ifdef USE_NCNN
    #include <ncnn/gpu.h>
#endif
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
    run_flag_ = false;
    if (camera_) {
        camera_->stop();
        camera_.reset();
    }
    if (serial_) {
        serial_->stop();
        serial_.reset();
    }
    if (debug_thread_.joinable()) {
        debug_thread_.join();
    }
    thread_pool_->waitUntilEmpty();

#ifdef USE_NCNN
    if (use_ncnn_count_ > 0) {
        ncnn::destroy_gpu_instance();
    }
#endif
    WUST_MAIN("main") << "vision stop already!";
}
bool VisionBase::init(bool debug_mode) {
    const char* v = std::getenv("VISION_ROOT");
    if (v)
        std::cout << "[env] VISION_ROOT = " << v << "\n";
    else
        std::cout << "[env] VISION_ROOT not set in this process\n";
    debug_mode_ = debug_mode;
    config_ = YAML::LoadFile(common_config_);
    debug_fps_ = config_["debug_fps"].as<int>(30);
    std::string log_level = config_["logger"]["log_level"].as<std::string>("INFO");
    std::string log_path = config_["logger"]["log_path"].as<std::string>("wust_log");
    bool use_logcli = config_["logger"]["use_logcli"].as<bool>();
    bool use_logfile = config_["logger"]["use_logfile"].as<bool>();
    bool use_simplelog = config_["logger"]["use_simplelog"].as<bool>();
    initLogger(log_level, log_path, use_logcli, use_logfile, use_simplelog);
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
    R_camera2gimbal_ = Eigen::Map<const Eigen::Matrix<double, 3, 3, Eigen::RowMajor>>(R_vec.data());

    YAML::Node camera_config = YAML::LoadFile(camera_config_);
    camera_ = std::make_shared<wust_vl_video::Camera>();
    camera_->init(camera_config);
    camera_->setFrameCallback(std::bind(&VisionBase::frameCallback, this, std::placeholders::_1));
    std::string camera_info_path =
        utils::expandEnv(camera_config["camera_info_path"].as<std::string>());
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
    auto_aim_ = std::make_shared<auto_aim::AutoAim>();
    auto_aim_
        ->init(auto_aim_config, use_ncnn_count_, R_camera2gimbal_, t_camera2gimbal_, camera_info);
    YAML::Node auto_buff_config = YAML::LoadFile(auto_buff_config_);
    auto_buff_ = std::make_shared<auto_buff::AutoBuff>();
    auto_buff_->init(
        auto_buff_config,
        use_ncnn_count_,
        R_camera2gimbal_,
        t_camera2gimbal_,
        camera_info,
        max_infer_running_
    );
    thread_pool_ = std::make_unique<ThreadPool>(max_infer_running_);
    motion_buffer_ = std::make_shared<MotionBufferGeneric<Motion, 1024>>();
    double bullet_speed = config_["shoot"]["bullet_speed"].as<double>(20.0);
    shoot_rate_ = config_["shoot"]["rate"].as<int>(3);
    double communication_delay_μs = config_["control"]["communication_delay_us"].as<double>(1000.0);

    auto_aim_shared_ = std::make_shared<auto_aim::AutoAimShared>(
        motion_buffer_,
        bullet_speed,
        communication_delay_μs
    );
    auto_aim_->setShared(auto_aim_shared_);
    auto_aim_->setDebug(debug_mode_);
    auto_buff_shared_ = std::make_shared<auto_buff::AutoBuffShared>(
        motion_buffer_,
        bullet_speed,
        communication_delay_μs
    );
    auto_buff_->setShared(auto_buff_shared_);
    auto_buff_->setDebug(debug_mode_);
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

    timer_ = std::make_unique<Timer>();
    detect_color_ = config_["detect_color"].as<int>(0);

    bool use_record = config_["record"]["use_record"].as<bool>(false);
    if (use_record) {
        std::string folder_path = config_["record"]["folder_path"].as<std::string>();
        auto file_name = utils::makeTimestampedFileName();
        std::string text_path = fmt::format("{}/{}.csv", folder_path, file_name);
        std::string video_path = fmt::format("{}/{}.avi", folder_path, file_name);

        std::filesystem::create_directory(folder_path);
        auto rw = std::make_shared<RotateWriterCSV>(true);
        rotate_writer_ = std::make_shared<wust_vl::Recorder<Eigen::Vector3d>>(text_path, rw);
        auto imgw = std::make_shared<ImgWriter>(
            video_path,
            30,
            cv::VideoWriter::fourcc('M', 'J', 'P', 'G')
        );
        img_writer_ = std::make_shared<wust_vl::Recorder<cv::Mat>>("", imgw);
    }
    bool use_rotate_reader = config_["record"]["use_rotate_reader"].as<bool>(false);
    if (use_rotate_reader) {
        std::string csv_path = config_["record"]["read_csv_path"].as<std::string>();
        rotate_reader_ = std::make_shared<RotateReaderCSV>(csv_path);
    }
    yaw_ramp_ = config_["control"]["yaw_ramp"].as<double>(0.0);
    pitch_ramp_ = config_["control"]["pitch_ramp"].as<double>(0.0);

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
    if (len != sizeof(ReceiveAimINFO)) {
        return;
    }
    try {
        std::vector<uint8_t> buf(data, data + len);
        ReceiveAimINFO aim_data = fromVector<ReceiveAimINFO>(buf);
        processAimData(aim_data);

    } catch (const std::exception& e) {
        std::cerr << "serialCallback exception: " << e.what() << std::endl;
    } catch (...) {
        std::cerr << "serialCallback unknown exception" << std::endl;
    }
}
void VisionBase::processAimData(const ReceiveAimINFO& aim_data) {
    static Averager<double> vyaw_avg(100);
    if (!this->run_flag_) {
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
    //updateBulletSpeed(aim_data.bullet_speed);
    double v_x = 0.0;
    double v_y = 0.0;
    double v_z = 0.0;

    auto now = std::chrono::steady_clock::now();
    if (motion_buffer_) {
        Motion motion { yaw, pitch, roll, 0.0, v_pitch, v_roll, v_x, v_y, v_z };
        motion_buffer_->push(motion, now);
    }

    writeSerialLogToJson(aim_data);
    static auto last_push_time = std::chrono::steady_clock::now();
    auto elapsed =
        std::chrono::duration_cast<std::chrono::milliseconds>(now - last_push_time).count();
    if (elapsed >= 10) { // 至少间隔 10ms（100Hz）
        if (rotate_writer_) {
            rotate_writer_->push(Eigen::Vector3d(aim_data.yaw, aim_data.pitch, aim_data.roll));
        }
        last_push_time = now;
    }
}

void VisionBase::autoExposureControl(const cv::Mat& frame) {
    AttackMode mode = toAttackMode(attack_mode_);
    switch (mode) {
        case AttackMode::UNKNOWN:
        case AttackMode::ARMOR: {
            auto_aim_->autoExposureControl(frame, camera_);
        } break;
        case AttackMode::SMALL_RUNE:
        case AttackMode::BIG_RUNE: {
            auto_buff_->autoExposureControl(frame, camera_);
        } break;
    }
}
// #ifdef USE_TRT
//     #include "cuda_infer/cvtcolor.hpp"
// #endif
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
    if (frame.src_img.channels() == 3) {
        common_frame.src_img = std::move(frame.src_img);
    } else {
        // #ifdef USE_TRT
        //         static cuda_cvt::CudaBayer_EA cuda_cvt;
        //         cuda_cvt.process(frame.src_img, common_frame.src_img, frame.pixel_type);
        // #endif
    }

    common_frame.expanded = cv::Rect(0, 0, common_frame.src_img.cols, common_frame.src_img.rows);
    common_frame.offset = cv::Point2f(0, 0);
    autoExposureControl(common_frame.src_img);
    if (img_writer_) {
        img_writer_->push(common_frame.src_img);
    }
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
            case AttackMode::UNKNOWN:
            case AttackMode::ARMOR: {
                auto_aim_->pushInput(frame);
            } break;
            case AttackMode::SMALL_RUNE: {
                auto_buff_->pushInput(frame, false);
            } break;
            case AttackMode::BIG_RUNE: {
                auto_buff_->pushInput(frame, true);
            } break;
        }
        infer_running_count_--;
    });
}
void VisionBase::checkStateMatchMode() {
    AttackMode mode = toAttackMode(attack_mode_);
    switch (mode) {
        case AttackMode::UNKNOWN:
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
    }
}

void VisionBase::timerCallback(double dt_ms) {
    if (!run_flag_) {
        return;
    }

    GimbalCmd cmd;
    try {
        AttackMode mode = toAttackMode(attack_mode_);
        switch (mode) {
            case AttackMode::UNKNOWN:
            case AttackMode::ARMOR: {
                cmd = auto_aim_->solve(dt_ms);
            } break;
            case AttackMode::SMALL_RUNE:
            case AttackMode::BIG_RUNE: {
                cmd = auto_buff_->solve();
            } break;
        }
    } catch (const std::exception& e) {
        std::cout << "auto_aim_solve error: " << e.what() << std::endl;
    }
    if (!cmd.isValid()) {
        return;
    }
    last_cmd_ = cmd;

    double cmd_pitch = cmd.pitch;
    double cmd_yaw = cmd.yaw;
    if (cmd.pitch >= 45.0) {
        cmd_pitch = 45.0;
    }

    SendRobotCmdData send_data;
    send_data.cmd_ID = ID_ROBOT_CMD;
    send_data.time_stamp = std::chrono::duration_cast<std::chrono::milliseconds>(
                               std::chrono::steady_clock::now().time_since_epoch()
    )
                               .count();
    if (cmd.distance > 0.5) {
        send_data.appear = cmd.appera;
    } else {
        send_data.appear = false;
    }

    send_data.detect_color = detect_color_;
    send_data.pitch = cmd_pitch + cmd.v_pitch * pitch_ramp_;
    send_data.yaw = cmd_yaw + cmd.v_yaw * yaw_ramp_;
    send_data.v_pitch = cmd.v_pitch;
    send_data.v_yaw = cmd.v_yaw;
    send_data.target_yaw = cmd.target_yaw;
    send_data.target_pitch = cmd.target_pitch;
    send_data.enable_pitch_diff = cmd.enable_pitch_diff;
    send_data.enable_yaw_diff = cmd.enable_yaw_diff;
    send_data.shoot_rate = shoot_rate_;
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
    } else if (rotate_reader_) {
        rotate_reader_->replay([this](const Eigen::Vector3d& ypr) {
            ReceiveAimINFO aim_data;
            aim_data.yaw = ypr(0);
            aim_data.pitch = ypr(1);
            aim_data.roll = ypr(2);
            this->processAimData(aim_data);
        }

        );
    }
    if (debug_mode_) {
        debug_thread_ = std::thread([this]() { this->debugThread(); });
    }
    if (rotate_writer_) {
        rotate_writer_->start();
    }
    if (img_writer_) {
        img_writer_->start();
    }
}
bool isWebRunning() {
    static std::atomic<bool> cached { true };
    static std::atomic<long long> last_check_ms { 0 };

    long long now_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                           std::chrono::steady_clock::now().time_since_epoch()
    )
                           .count();
    if (last_check_ms.load() == 0) {
        last_check_ms = now_ms;
        return true;
    }

    if (now_ms - last_check_ms.load() >= 1000) {
        int ret = std::system("pgrep -x wust_vision_web > /dev/null 2>&1");
        cached = (ret == 0);
        last_check_ms = now_ms;
    }

    return cached.load();
}
void VisionBase::debugThread() {
    using namespace std::chrono;
    double us_interval = 1e6 / static_cast<double>(debug_fps_);
    auto kInterval = std::chrono::microseconds(static_cast<int64_t>(us_interval));
    while (run_flag_) {
        auto start_time = steady_clock::now();
        do {
            try {
                if (!isWebRunning()) {
                    break;
                }
                if (!auto_aim_ || !auto_buff_) {
                    break;
                }
                auto dbg_armor = auto_aim_->getDebugFrame();
                auto dbg_rune = auto_buff_->getDebugFrame();
                AttackMode mode = toAttackMode(attack_mode_);
                switch (mode) {
                    case AttackMode::UNKNOWN:
                    case AttackMode::ARMOR: {
                        drawDebugOverlayShm(dbg_armor, camera_info_, false);
                    } break;
                    case AttackMode::SMALL_RUNE:
                    case AttackMode::BIG_RUNE: {
                        drawDebugOverlayShm(dbg_rune, camera_info_, false);
                    } break;
                }
                std::pair<double, double> gimbal_py;
                if (motion_buffer_) {
                    auto last_att = motion_buffer_->get_last();
                    if (last_att) {
                        gimbal_py.first = last_att->data.pitch;
                        gimbal_py.second = last_att->data.yaw;
                    }
                }

                debuglog(dbg_armor, dbg_rune, last_cmd_, gimbal_py);
            } catch (std::exception& e) {
                std::cout << "debug thread error: " << e.what() << std::endl;
            }
        } while (0);

        auto elapsed = steady_clock::now() - start_time;
        if (elapsed < kInterval) {
            std::this_thread::sleep_for(kInterval - elapsed);
        }
    }
}