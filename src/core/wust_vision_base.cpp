// Copyright 2025 Xiaojian Wu
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
#include "common/debug/toolsgobal.hpp"
#include "core/wust_vision.hpp"
#include "type/type.hpp"

WustVision::WustVision() {}
WustVision::~WustVision() {}
void WustVision::stop() {
    gobal::is_inited_ = false;
    auto future = std::async(std::launch::async, [this]() {
        if (!only_nav_enable_) {
            if (processing_thread_.joinable()) {
                processing_thread_.join();
            }
            if (auto_labeler_) {
                auto_labeler_.reset();
            }
            if (video_player_) {
                video_player_->stop();
                video_player_.reset();
            }
            if (camera_) {
                camera_->stopCamera();
                camera_.reset();
            }

            if (omni_manager_) {
                omni_manager_->stop();
                omni_manager_.reset();
            }
            if (timer_) {
                timer_->stop();
                timer_.reset();
            }
            WUST_INFO("stop") << "timer stop";
            auto thread_pool = gobal::stringanything.get_ptr<ThreadPool>("thread_pool");
            if (thread_pool) {
                thread_pool->waitUntilEmpty();
                thread_pool.reset();
            }
            WUST_INFO("stop") << "thread pool stop";

            armor_detector_.reset();
            WUST_INFO("stop") << "armor detector stop";
            rune_detector_.reset();
            WUST_INFO("stop") << "rune detector stop";

#ifdef USE_NCNN
            auto common_info = gobal::stringanything.get_value<CommonInfo>("common_info");
            if (common_info.use_detect_ncnn_count > 0) {
                ncnn::destroy_gpu_instance();
            }
#endif

            if (debug_thread_.joinable()) {
                debug_thread_.join();
            }
            WUST_INFO("stop") << "debug thread stop";
        }

        if (serial_) {
            serial_->stopThread();
            serial_.reset();
        }
        WUST_INFO("stop") << "serial stop";
        if (have_fun_) {
            have_fun_->stop();
            have_fun_.reset();
        }

        WUST_MAIN(vision_logger_) << "WustVision shutdown complete.";
        return 0;
    });

    if (future.wait_for(std::chrono::seconds(5)) == std::future_status::timeout) {
        WUST_ERROR("stop") << "stop() timeout, forcing process exit";
        std::exit(0);
    }
}
bool WustVision::init() {
    WUST_MAIN(vision_logger_) << "WustVision init start";
    gobal::config = YAML::LoadFile(ROOT_CONFIG);

    std::string log_level_ = gobal::config["logger"]["log_level"].as<std::string>("INFO");
    std::string log_path_ = gobal::config["logger"]["log_path"].as<std::string>("wust_log");
    bool use_logcli = gobal::config["logger"]["use_logcli"].as<bool>();
    bool use_logfile = gobal::config["logger"]["use_logfile"].as<bool>();
    bool use_simplelog = gobal::config["logger"]["use_simplelog"].as<bool>();
    GobalState gobal_state;
    gobal_state.attack_mode = gobal::config["common"]["init_attack_mode"].as<int>();
    gobal::stringanything.set_value<GobalState>("gobal_state", gobal_state);
    initLogger(log_level_, log_path_, use_logcli, use_logfile, use_simplelog);

    CommonInfo common_info;
    common_info.debug_mode = gobal::config["debug"]["debug_mode"].as<bool>();

    common_info.use_serial = gobal::config["control"]["use_serial"].as<bool>();
    common_info.use_detect_ncnn_count = 0;
    gobal::stringanything.set_value<CommonInfo>("common_info", common_info);

    int detect_color = gobal::config["common"]["detect_color"].as<int>(0);
    gobal::stringanything.set_value<int>("detect_color", detect_color);
    auto motion_buffer = std::make_shared<MotionBuffer>();
    gobal::stringanything.set_ptr<MotionBuffer>("motion_buffer", motion_buffer);
    initSerial();
    only_nav_enable_ = gobal::config["common"]["only_nav_enable"].as<bool>();
    if (!only_nav_enable_) {
        toolsgobal::debug_w = gobal::config["debug"]["debug_w"].as<int>(640);
        toolsgobal::debug_h = gobal::config["debug"]["debug_h"].as<int>(480);
        debug_show_dt_ = gobal::config["debug"]["debug_show_dt"].as<double>(0.05);
        toolsgobal::debug_fps = gobal::config["debug"]["debug_fps"].as<double>(30);

        use_video_ = gobal::config["camera"]["video_player"]["use"].as<bool>(false);
        auto imgFrameCallback = [this](ImageFrame& frame) {
            if (gobal::is_inited_) {
                img_recv_count_++;
                if (infer_running_count_.load() >= max_infer_running_) {
                    return;
                }
                auto motion_buffer =
                    gobal::stringanything.try_get_ptr<MotionBuffer>("motion_buffer");
                auto gimbal2camera_rpy =
                    gobal::stringanything.get_value<std::array<double, 3>>("gimbal2camera_rpy");
                if (motion_buffer) {
                    auto apply_motion = [&](const MotionBuffer::MotionStamped& att) {
                        frame.v = Eigen::Vector3d(att.vx, att.vy, att.vz);
                        frame.R_gimbal2odom = Eigen::AngleAxisd(
                                                  att.yaw + gimbal2camera_rpy[2],
                                                  Eigen::Vector3d::UnitZ()
                                              )
                            * Eigen::AngleAxisd(-att.pitch - gimbal2camera_rpy[1],
                                                Eigen::Vector3d::UnitY())
                            * Eigen::AngleAxisd(att.roll + gimbal2camera_rpy[0],
                                                Eigen::Vector3d::UnitX());
                    };
                    auto delay = std::chrono::microseconds(
                        static_cast<int64_t>(std::round(-communication_delay_μs_))
                    );
                    auto t_query = std::chrono::steady_clock::now() - delay;
                    if (auto past_att = motion_buffer->get()->get_interpolated(t_query)) {
                        apply_motion(*past_att);
                    } else if (auto last_att = motion_buffer->get()->get_last()) {
                        apply_motion(*last_att);
                    } else {
                        frame.R_gimbal2odom = Eigen::Matrix3d::Identity();
                        frame.v = Eigen::Vector3d::Zero();
                    }
                } else {
                    frame.R_gimbal2odom = Eigen::Matrix3d::Identity();
                    frame.v = Eigen::Vector3d::Zero();
                }

                gobal::stringanything.get_ptr<ThreadPool>("thread_pool")
                    ->enqueue(
                        [frame = std::move(frame), this]() {
                            infer_running_count_++;
                            processImage(frame);
                            infer_running_count_--;
                        },
                        -1
                    );

            } else {
                return;
            }
        };
        if (use_video_) {
            std::string video_play_path =
                gobal::config["camera"]["video_player"]["path"].as<std::string>("");
            int video_play_fps = gobal::config["camera"]["video_player"]["fps"].as<int>(30);
            int start_frame = gobal::config["camera"]["video_player"]["start_frame"].as<int>(0);
            bool loop = gobal::config["camera"]["video_player"]["loop"].as<bool>(false);
            video_alpha_ = gobal::config["camera"]["video_player"]["alpha"].as<double>(1.0);
            video_beta_ = gobal::config["camera"]["video_player"]["beta"].as<int>(0);
            video_player_ =
                std::make_unique<VideoPlayer>(video_play_path, video_play_fps, start_frame, loop);
            video_player_->setCallback(imgFrameCallback);

        } else {
            camera_ = std::make_unique<HikCamera>();
            std::string target_sn = gobal::config["camera"]["target_sn"].as<std::string>();
            if (!camera_->initializeCamera(target_sn)) {
                WUST_ERROR(vision_logger_) << "Camera initialization failed.";
                return false;
            }

            camera_->setParameters(
                gobal::config["camera"]["acquisition_frame_rate"].as<int>(),
                gobal::config["camera"]["exposure_time"].as<int>(),
                gobal::config["camera"]["gain"].as<double>(),
                gobal::config["camera"]["gamma"].as<double>(),
                gobal::config["camera"]["adc_bit_depth"].as<std::string>(),
                gobal::config["camera"]["pixel_format"].as<std::string>(),
                gobal::config["camera"]["acquisition_frame_rate_enable"].as<bool>(),
                gobal::config["camera"]["reverse_x"].as<bool>(false),
                gobal::config["camera"]["reverse_y"].as<bool>(false)
            );
            camera_->setFrameCallback(imgFrameCallback);
        }
        use_auto_labeler_ = gobal::config["common"]["use_auto_labeler"].as<bool>(false);
        if (use_auto_labeler_) {
            auto_labeler_ = std::make_unique<Labeler>();
        }

        const std::string camera_info_path =
            gobal::config["camera"]["camera_info_path"].as<std::string>();
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
        gobal::stringanything.set_value<std::pair<cv::Mat, cv::Mat>>("camera_info", camera_info);

        armor_pose_estimator_ = std::make_unique<ArmorPoseEstimator>();
        bool use_ba = gobal::config["common"]["use_ba"].as<bool>(false);
        if (use_ba) {
            armor_pose_estimator_->enableBA(true);
        } else {
            armor_pose_estimator_->enableBA(false);
        }
        initTF();
        initTracker(gobal::config["armor_tracker"]);

        max_infer_running_ = gobal::config["common"]["max_infer_running"].as<int>(4);
        initDetector();
        initRune(camera_info_path); //无论是否使用仍然初始化rune_solver
        max_detect_armors_ = gobal::config["common"]["max_detect_armors"].as<int>(10);
        int thread_multiplier = gobal::config["common"]["thread_multiplier"].as<int>(1);
        auto thread_pool =
            std::make_shared<ThreadPool>(std::thread::hardware_concurrency() * thread_multiplier);
        gobal::stringanything.set_ptr<ThreadPool>("thread_pool", thread_pool);
        armor_solver_ = std::make_unique<ArmorSolver>(gobal::config);
        bool use_omni = gobal::config["common"]["use_omni"].as<bool>(false);
        if (use_omni) {
            auto omni_config = YAML::LoadFile(OMNI_CONFIG);
            omni_manager_ = std::make_unique<OmniManager>(omni_config);
        }
        timer_ = std::make_unique<Timer>();
        armor_queue_ = std::make_unique<OrderedQueue<armor::Armors>>(10, 500);
        rune_queue_ = std::make_unique<OrderedQueue<rune::Rune>>(10, 500);
        GimbalCmd gimbal_cmd;
        gobal::stringanything.set_value<GimbalCmd>("last_gimbal_cmd", gimbal_cmd);
        double bullet_speed = gobal::config["shoot"]["bullet_speed"].as<double>(20.0);
        gobal::stringanything.set_value<double>("bullet_speed", bullet_speed);
        double controller_delay = gobal::config["shoot"]["controller_delay"].as<double>(0.0);
        gobal::stringanything.set_value<double>("controller_delay", controller_delay);
    } else {
        WUST_MAIN(vision_logger_) << "only nav mode";
    }
    bool use_fun = gobal::config["common"]["use_fun"].as<bool>(false);
    if (use_fun) {
        have_fun_ = std::make_unique<HaveFun>(YAML::LoadFile(FUN_CONFIG));
    }

    gobal::is_inited_ = true;
    WUST_MAIN(vision_logger_) << "WustVision init success";
    return true;
}
void WustVision::start() {
    WUST_MAIN(vision_logger_) << "WustVision run start";
    auto common_info = gobal::stringanything.get_value<CommonInfo>("common_info");
    if (serial_) {
        bool if_use_nav = gobal::config["control"]["use_nav"].as<bool>(false);

        serial_->startThread(common_info.use_serial, if_use_nav);
    }
    if (!only_nav_enable_) {
        if (video_player_) {
            video_player_->start();
        }

        if (camera_) {
            bool if_recorder = gobal::config["camera"]["recorder"].as<bool>(false);
            camera_->startCamera(if_recorder);
        }
        processing_thread_ = std::thread(&WustVision::processingLoop, this);
        if (omni_manager_) {
            omni_manager_->start();
        }
        if (timer_) {
            auto timercallback = std::bind(&WustVision::timerCallback, this, std::placeholders::_1);
            double rate_hz =
                static_cast<double>(gobal::stringanything.get_value<int>("control_rate"));
            timer_->start(rate_hz, timercallback);
        }
    }
    if (common_info.debug_mode) {
        debug_thread_ = std::thread([this]() { this->debugThread(); });
    }
    if (have_fun_) {
        have_fun_->start();
    }
    WUST_MAIN(vision_logger_) << "WustVision run success";
}
void WustVision::initDetector() {
    std::string armor_detect_backend =
        gobal::config["common"]["armor_detect_backend"].as<std::string>("");
    std::string rune_detect_backend =
        gobal::config["common"]["rune_detect_backend"].as<std::string>("");

    auto isBackendEnabled = [](const std::string& backend) -> bool {
#ifdef USE_OPENVINO
        if (backend == "openvino")
            return true;
#endif
#ifdef USE_TRT
        if (backend == "tensorrt")
            return true;
#endif
#ifdef USE_NCNN
        if (backend == "ncnn") {
            auto common_info = gobal::stringanything.get_value<CommonInfo>("common_info");
            common_info.use_detect_ncnn_count++;
            gobal::stringanything.set_value<CommonInfo>("common_info", common_info);
            return true;
        }

#endif
#ifdef USE_ORT
        if (backend == "onnxruntime")
            return true;
#endif
        if (backend == "opencv")
            return true;
        return false;
    };

    auto getConfigPath = [](const std::string& backend) -> std::string {
        if (backend == "openvino")
            return OPENVINO_CONFIG;
        if (backend == "tensorrt")
            return TENSORRT_CONFIG;
        if (backend == "ncnn")
            return NCNN_CONFIG;
        if (backend == "onnxruntime")
            return ONNXRUNTIME_CONFIG;
        if (backend == "opencv")
            return OPENCV_CONFIG;
        return "";
    };

    auto loadArmorDetectorBackend = [&](const std::string& backend) {
        if (!isBackendEnabled(backend)) {
            throw std::runtime_error("Backend " + backend + " is not enabled at compile time.");
        }
        std::string config_path = getConfigPath(backend);
        if (config_path.empty()) {
            throw std::runtime_error("No config path for backend: " + backend);
        }
        armor_detect_config_ = YAML::LoadFile(config_path);
        return DetectorFactory::createArmorDetector(backend, armor_detect_config_, true);
    };

    auto loadRuneDetectorBackend = [&](const std::string& backend) {
        if (!isBackendEnabled(backend)) {
            throw std::runtime_error("Backend " + backend + " is not enabled at compile time.");
        }
        std::string config_path = getConfigPath(backend);
        if (config_path.empty()) {
            throw std::runtime_error("No config path for backend: " + backend);
        }
        rune_detect_config_ = YAML::LoadFile(config_path);
        return DetectorFactory::createRuneDetector(backend, rune_detect_config_);
    };

    if (armor_detect_backend.empty()) {
        throw std::runtime_error("armor_detect_backend not set in config.");
    }
    armor_detector_ = loadArmorDetectorBackend(armor_detect_backend);
    WUST_MAIN(vision_logger_) << "Using Armor Detector: " << armor_detect_backend;

    armor_detector_->setCallback(std::bind(
        &WustVision::ArmorDetectCallback,
        this,
        std::placeholders::_1,
        std::placeholders::_2
    ));
#ifdef USE_RUNE
    if (rune_detect_backend.empty()) {
        throw std::runtime_error("rune_detect_backend not set in config.");
    }
    rune_detector_ = loadRuneDetectorBackend(rune_detect_backend);
    WUST_MAIN(vision_logger_) << "Using Rune Detector: " << rune_detect_backend;
    rune_detector_->setCallback(std::bind(
        &WustVision::RuneDetectCallback,
        this,
        std::placeholders::_1,
        std::placeholders::_2
    ));

#endif
}
void WustVision::initRune(const std::string& camera_info_path) {
#ifdef USE_RUNE
    detect_r_tag_ = rune_detect_config_["rune_detector"]["detect_r_tag"].as<bool>();
    use_manual_r_ = rune_detect_config_["rune_detector"]["use_manual_r"].as<bool>();
    rune_binary_thresh_ = rune_detect_config_["rune_detector"]["min_lightness"].as<int>();
#endif

    rune_solver_ = std::make_unique<RuneSolver>(gobal::config);
}

void WustVision::initTF() {
    gimbal2camera_x_ = gobal::config["tf"]["gimbal2camera_x"].as<double>(0.0);
    gimbal2camera_y_ = gobal::config["tf"]["gimbal2camera_y"].as<double>(0.0);
    gimbal2camera_z_ = gobal::config["tf"]["gimbal2camera_z"].as<double>(0.0);
    gimbal2camera_roll_ = gobal::config["tf"]["gimbal2camera_roll"].as<double>(0.0);
    gimbal2camera_pitch_ = gobal::config["tf"]["gimbal2camera_pitch"].as<double>(0.0);
    gimbal2camera_yaw_ = gobal::config["tf"]["gimbal2camera_yaw"].as<double>(0.0);

    gobal::stringanything.set_value<std::array<double, 3>>(
        "gimbal2camera_rpy",
        { gimbal2camera_roll_, gimbal2camera_pitch_, gimbal2camera_yaw_ }
    );
    t_gimbal_to_camera_ = Eigen::Vector3d(gimbal2camera_x_, gimbal2camera_y_, gimbal2camera_z_);

    // 转换为旋转矩阵使用
    R_camera2gimbal_ << 0, 0, 1, -1, 0, 0, 0, -1, 0;
}
void WustVision::initSerial() {
    SerialPortConfig cfg { /*baud*/ 115200,
                           /*csize*/ 8,
                           boost::asio::serial_port_base::parity::none,
                           boost::asio::serial_port_base::stop_bits::one,
                           boost::asio::serial_port_base::flow_control::none };

    std::string device_name = gobal::config["control"]["device_name"].as<std::string>();
    serial_ = std::make_unique<serial::Serial>();
    serial_->init(device_name, cfg);
    serial_->alpha_yaw_ = gobal::config["control"]["alpha_yaw"].as<double>();
    serial_->alpha_pitch_ = gobal::config["control"]["alpha_pitch"].as<double>();
    serial_->max_yaw_change_ = gobal::config["control"]["max_yaw_change"].as<double>();
    serial_->max_pitch_change_ = gobal::config["control"]["max_pitch_change"].as<double>();
    communication_delay_μs_ = gobal::config["control"]["communication_delay"].as<double>();
    jump_yaw = gobal::config["control"]["jump_yaw"].as<double>();
    int control_rate = gobal::config["control"]["control_rate"].as<int>();
    gobal::stringanything.set_value<int>("control_rate", control_rate);
}

void WustVision::initTracker(const YAML::Node& config) {
    tracker_manager_ = std::make_unique<TrackerManager>(config);
}