#include "omni.hpp"
OmniVision::OmniVision() {
    // Constructor implementation
}

OmniVision::~OmniVision() {}
void OmniVision::stop() {
    is_inited_ = false;
    video_player_->stop();
}

bool OmniVision::init(
    const YAML::Node& config,
    std::function<void(const ImageFrame&, const Eigen::Matrix3d&, const Eigen::Vector3d&)>
        on_frame_callback_,
    size_t index
) {
    this->callback = std::move(on_frame_callback_);
    this->index = index;
    this->use_video = config["camera"]["video_player"]["use"].as<bool>(true);
    if (this->use_video) {
        std::string video_play_path = config["camera"]["video_player"]["path"].as<std::string>("");
        int video_play_fps = config["camera"]["video_player"]["fps"].as<int>(30);
        int start_frame = config["camera"]["video_player"]["start_frame"].as<int>(0);
        bool loop = config["camera"]["video_player"]["loop"].as<bool>(false);
        video_player_ =
            std::make_unique<VideoPlayer>(video_play_path, video_play_fps, start_frame, loop);
        video_player_->enableTriggerMode(true);
        video_player_->setCallback([this](const ImageFrame& frame) {
            static bool first_is_inited = false;

            if (is_inited_) {
                Eigen::Matrix3d R_gimbal2odom;
                R_gimbal2odom = Eigen::AngleAxisd(gimbal2camera_yaw_, Eigen::Vector3d::UnitZ())
                    * Eigen::AngleAxisd(gimbal2camera_pitch_, Eigen::Vector3d::UnitY())
                    * Eigen::AngleAxisd(gimbal2camera_roll_, Eigen::Vector3d::UnitX());
                Eigen::Vector3d fake_v(this->index, this->index, this->index);
                this->callback(frame, R_gimbal2odom, fake_v);
            } else {
                return;
            }
        });

    } else {
        camera_ = std::make_unique<HikCamera>();
        std::string target_sn = config["camera"]["target_sn"].as<std::string>();
        if (!camera_->initializeCamera(target_sn)) {
            WUST_ERROR("vision_logger") << "Camera initialization failed.";
            return false;
        }

        camera_->setParameters(
            config["camera"]["acquisition_frame_rate"].as<int>(),
            config["camera"]["exposure_time"].as<int>(),
            config["camera"]["gain"].as<double>(),
            config["camera"]["adc_bit_depth"].as<std::string>(),
            config["camera"]["pixel_format"].as<std::string>(),
            config["camera"]["acquisitionFrameRateEnable"].as<bool>()
        );
        camera_->enableTrigger(TriggerType::Software, "Software", 0);
        camera_->setFrameCallback(
            [this](const ImageFrame& frame, Eigen::Matrix3d R_gimbal2odom, Eigen::Vector3d v) {
                static bool first_is_inited = false;

                if (is_inited_) {
                    Eigen::Matrix3d R_gimbal2odom_;
                    R_gimbal2odom_ = Eigen::AngleAxisd(gimbal2camera_yaw_, Eigen::Vector3d::UnitZ())
                        * Eigen::AngleAxisd(gimbal2camera_pitch_, Eigen::Vector3d::UnitY())
                        * Eigen::AngleAxisd(gimbal2camera_roll_, Eigen::Vector3d::UnitX());
                    Eigen::Vector3d fake_v(this->index, this->index, this->index);
                    this->callback(frame, R_gimbal2odom_, fake_v);
                } else {
                    return;
                }
            }
        );
    }
    const std::string camera_info_path = config["camera"]["camera_info_path"].as<std::string>();
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

    camera_intrinsic_ = K.clone();
    camera_distortion_ = D.clone();
    initTF(config);
    is_inited_ = true;
    run();
    return true;
}
void OmniVision::initTF(const YAML::Node& config) {
    double gimbal2camera_roll = config["tf"]["gimbal2camera_roll"].as<double>(0.0);
    double gimbal2camera_pitch = config["tf"]["gimbal2camera_pitch"].as<double>(0.0);
    double gimbal2camera_yaw = config["tf"]["gimbal2camera_yaw"].as<double>(0.0);
    gimbal2camera_roll_ = gimbal2camera_roll / 180.0 * M_PI;
    gimbal2camera_pitch_ = gimbal2camera_pitch / 180.0 * M_PI;
    gimbal2camera_yaw_ = gimbal2camera_yaw / 180.0 * M_PI;
}
void OmniVision::run() {
    if (use_video && video_player_) {
        video_player_->start();
    }
}

OmniManager::OmniManager(const YAML::Node& config) {
    config_ = config;
    callback = [this](
                   const ImageFrame& frame,
                   const Eigen::Matrix3d& R_gimbal2odom,
                   const Eigen::Vector3d& v
               ) {
        static bool first_is_inited = false;

        if (gobal::is_inited_) {
            thread_pool_->enqueue([frame = std::move(frame), R_gimbal2odom, v, this]() {
                processImage(frame, R_gimbal2odom, v);
            });
        } else {
            return;
        }
    };
    omni_num_ = config["common"]["omni_num"].as<size_t>(1);
    total_fps_ = config["common"]["total_fps"].as<int>(30);
    for (size_t i = 0; i < omni_num_; ++i) {
        YAML::Node omni_config = config["omni_vision"][i];
        auto omni_vision = std::make_unique<OmniVision>();
        if (!omni_vision->init(omni_config, callback, i)) {
            WUST_ERROR(vision_logger) << "OmniVision initialization failed."
                                      << " Index: " << i;
        }

        omni_visions_.emplace_back(std::move(omni_vision));
    }
    measure_tool_ = std::make_unique<MonoMeasureTool>();
    initdetector();
    thread_pool_ = std::make_unique<ThreadPool>(std::thread::hardware_concurrency(), 100);
}
OmniManager::~OmniManager() {}
void OmniManager::stop() {
    for (auto& omni_vision: omni_visions_) {
        omni_vision->stop();
        omni_vision.reset();
    }
    omni_visions_.clear();
    stopTimer();
    std::this_thread::sleep_for(std::chrono::seconds(1));

    armor_detector_.reset();
    measure_tool_.reset();

    if (thread_pool_) {
        thread_pool_->waitUntilEmpty();
        thread_pool_.reset();
    }
    WUST_INFO(vision_logger) << "OmniManager shutdown complete.";
}
void OmniManager::initdetector() {
    max_infer_running_ = config_["common"]["max_infer_running"].as<int>(10);
    max_detect_armors_ = config_["common"]["max_detect_armors"].as<int>(10);
    bool use_armor_detect_opencv = config_["common"]["use_armor_detect_opencv"].as<bool>(false);
    bool ncnn_runeinited = false;
    bool ncnn_armorinited = false;
    bool use_armor_detect_ncnn = config_["common"]["use_armor_detect_ncnn"].as<bool>(false);

#ifdef USE_OPENVINO
    #ifdef USE_NCNN

    if (use_armor_detect_ncnn) {
        gobal::use_armor_detect_ncnn_count++;
        auto ncnn_config = config_["armor_detect_ncnn"];
        armor_detector_ = DetectorFactory::createArmorDetector("ncnn", ncnn_config, false);
        ncnn_armorinited = true;
        WUST_MAIN(vision_logger) << "Using Armor Detector: ncnn";
    }

    #endif
    if (!ncnn_armorinited) {
        if (use_armor_detect_opencv) {
            auto opencv_config = config_["armor_detect_opencv"];
            armor_detector_ = DetectorFactory::createArmorDetector("opencv", opencv_config, false);
            WUST_MAIN(vision_logger) << "Using Armor Detector: opencv";
        } else {
            auto openvino_config = config_["armor_detect_openvino"];
            armor_detector_ =
                DetectorFactory::createArmorDetector("openvino", openvino_config, false);
            WUST_MAIN(vision_logger) << "Using Armor Detector: openvino";
        }
    }

#elif defined(USE_TRT)
    #ifdef USE_NCNN

    if (use_armor_detect_ncnn) {
        auto ncnn_config = config_["armor_detect_ncnn"];
        armor_detector_ = DetectorFactory::createArmorDetector("ncnn", ncnn_config, false);
        ncnn_armorinited = true;
        WUST_MAIN(vision_logger) << "Using Armor Detector: ncnn";
    }

    #endif
    if (!ncnn_armorinited) {
        if (use_armor_detect_opencv) {
            auto opencv_config = config_["armor_detect_opencv"];
            armor_detector_ = DetectorFactory::createArmorDetector("opencv", opencv_config, false);
            WUST_MAIN(vision_logger) << "Using Armor Detector: opencv";
        } else {
            auto trt_config = config_["armor_detect_trt"];
            armor_detector_ = DetectorFactory::createArmorDetector("tensorrt", trt_config, false);
            WUST_MAIN(vision_logger) << "Using Armor Detector: tensorrt";
        }
    }

#elif defined(USE_NCNN_ONLY)
    if (use_armor_detect_opencv) {
        auto opencv_config = config_["armor_detect_opencv"];
        armor_detector_ = DetectorFactory::createArmorDetector("opencv", opencv_config, false);
        WUST_MAIN(vision_logger) << "Using Armor Detector: opencv";
    } else {
        auto ncnn_config = config_["armor_detect_ncnn"];
        armor_detector_ = DetectorFactory::createArmorDetector("ncnn", ncnn_config, false);
        WUST_MAIN(vision_logger) << "Using Armor Detector: ncnn";
        use_armor_detect_ncnn = true;
    }

#else
    static_assert(false, "No backend defined: USE_OPENVINO or USE_TRT");
#endif

    armor_detector_->setCallback(std::bind(
        &OmniManager::ArmorDetectCallback,
        this,
        std::placeholders::_1,
        std::placeholders::_2,
        std::placeholders::_3,
        std::placeholders::_4,
        std::placeholders::_5
    ));
}
void OmniManager::ArmorDetectCallback(
    const std::vector<ArmorObject>& objs,
    std::chrono::steady_clock::time_point timestamp,
    const cv::Mat& src_img,
    const Eigen::Matrix4d& T_camera_to_odom,
    const Eigen::Vector3d& v
) {
    std::lock_guard<std::mutex> lock(callback_mutex_);

    if (objs.size() >= max_detect_armors_) {
        WUST_WARN(vision_logger) << "Detected " << objs.size() << " objects"
                                 << "too much";
        detect_finish_count_++;
        infer_running_count_--;
        return;
    }
    Armors armors;
    armors.timestamp = timestamp;
    armors.frame_id = "camera_optical_frame";
    cv::Mat camera_intrinsic_ = omni_visions_[v.x()]->camera_intrinsic_;
    cv::Mat camera_distortion_ = omni_visions_[v.x()]->camera_distortion_;
    measure_tool_->processDetectedArmorsOmni(
        objs,
        gobal::detect_color_,
        armors,
        T_camera_to_odom,
        camera_intrinsic_,
        camera_distortion_
    );
    cv::Mat debug_img;
    debug_img = src_img.clone();
    cv::imshow("Debug Image", debug_img);
    cv::waitKey(1);
    detect_finish_count_++;
    infer_running_count_--;
}
void OmniManager::processImage(
    const ImageFrame& frame,
    const Eigen::Matrix3d& R_gimbal2odom,
    const Eigen::Vector3d& v
) {
    img_recv_count_++;
    if (infer_running_count_.load() >= max_infer_running_) {
        return;
    }
    cv::Mat img;
    if (!omni_visions_[v.x()]->use_video) {
        img = convertToMatrgb(frame);
    } else {
        img = convertToMatbgr(frame);
    }

    // Step 1: gimbal → odom
    Eigen::Matrix4d T_gimbal_to_odom = Eigen::Matrix4d::Identity();
    T_gimbal_to_odom.block<3, 3>(0, 0) = R_gimbal2odom;

    // Step 2: camera → gimbal （取 R 的转置）
    Eigen::Matrix3d R_camera_to_gimbal;
    R_camera_to_gimbal << 0, 0, 1, -1, 0, 0, 0, -1, 0;
    Eigen::Vector3d t_gimbal_to_camera(0, 0, 0); // 假设相机在云台中心
    Eigen::Vector3d t_camera_to_gimbal = -R_camera_to_gimbal * t_gimbal_to_camera;

    Eigen::Matrix4d T_camera_to_gimbal = Eigen::Matrix4d::Identity();
    T_camera_to_gimbal.block<3, 3>(0, 0) = R_camera_to_gimbal;
    T_camera_to_gimbal.block<3, 1>(0, 3) = t_camera_to_gimbal;

    // Step 3: camera → odom
    Eigen::Matrix4d T_camera_to_odom = T_gimbal_to_odom * T_camera_to_gimbal;

    infer_running_count_++;
    armor_detector_->pushInput(img, frame.timestamp, T_camera_to_odom, v);
}
void OmniManager::stopTimer() {
    {
        std::lock_guard<std::mutex> lk(timer_mtx_);
        timer_running_ = false;
    }
    timer_cv_.notify_one();
    if (timer_thread_.joinable()) {
        timer_thread_.join();
    }
    WUST_INFO(vision_logger) << "OmniManager timer stopped.";
}
void OmniManager::startTimer() {
    if (timer_running_)
        return;
    WUST_INFO(vision_logger) << "starting timer";

    timer_running_ = true;

    double us_interval = 1e6 / static_cast<double>(total_fps_);
    auto interval = std::chrono::microseconds(static_cast<int64_t>(us_interval));

    constexpr auto spin_margin = std::chrono::microseconds(200);

    timer_thread_ = std::thread([this, interval, spin_margin]() {
        auto next_time = std::chrono::steady_clock::now() + interval;
        auto last_time = std::chrono::steady_clock::now();

        while (true) {
            {
                std::unique_lock<std::mutex> lk(timer_mtx_);
                auto sleep_until = next_time - spin_margin;
                timer_cv_.wait_until(lk, sleep_until, [this]() { return !timer_running_; });
                if (!timer_running_)
                    break;
            }

            while (std::chrono::steady_clock::now() < next_time) {
                // busy‐wait
            }

            auto now = std::chrono::steady_clock::now();
            double dt_ms = std::chrono::duration<double, std::milli>(now - last_time).count();
            last_time = now;

            this->timerCallback(dt_ms);
            next_time += interval;
        }
    });
}
void OmniManager::timerCallback(double dt_ms) {
    size_t index = count_ % omni_num_;
    ++count_;
    if (omni_visions_[index]->use_video) {
        omni_visions_[index]->video_player_->read();

    } else {
        omni_visions_[index]->camera_->read();
    }
}
