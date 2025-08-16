#include "misc/omni.hpp"
#include "common/debug/tools.hpp"
#include "common/queues.hpp"
#include "common/utils.hpp"
OmniVision::OmniVision() {
    // Constructor implementation
}

OmniVision::~OmniVision() {}
void OmniVision::stop() {
    is_inited_ = false;
    video_player_->stop();
    video_player_.reset();
    camera_.reset();
}

bool OmniVision::init(
    const YAML::Node& config,
    std::function<void(const ImageFrame&)> on_frame_callback_,
    size_t index
) {
    this->callback_ = std::move(on_frame_callback_);
    this->index_ = index;
    this->use_video = config["camera"]["video_player"]["use"].as<bool>(true);
    if (this->use_video) {
        std::string video_play_path = config["camera"]["video_player"]["path"].as<std::string>("");
        int video_play_fps = config["camera"]["video_player"]["fps"].as<int>(30);
        int start_frame = config["camera"]["video_player"]["start_frame"].as<int>(0);
        bool loop = config["camera"]["video_player"]["loop"].as<bool>(false);
        video_alpha = config["camera"]["video_player"]["alpha"].as<double>(1.0);
        video_beta = config["camera"]["video_player"]["beta"].as<int>(0);
        video_player_ =
            std::make_unique<VideoPlayer>(video_play_path, video_play_fps, start_frame, loop);
        video_player_->enableTriggerMode(true);
        video_player_->setCallback([this](ImageFrame& frame) {
            static bool first_is_inited = false;

            if (is_inited_) {
                frame.R_gimbal2odom =
                    Eigen::AngleAxisd(gimbal2camera_yaw_, Eigen::Vector3d::UnitZ())
                    * Eigen::AngleAxisd(gimbal2camera_pitch_, Eigen::Vector3d::UnitY())
                    * Eigen::AngleAxisd(gimbal2camera_roll_, Eigen::Vector3d::UnitX());
                frame.v = Eigen::Vector3d(this->index_, this->index_, this->index_);
                this->callback_(frame);
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
            config["camera"]["gamma"].as<double>(),
            config["camera"]["adc_bit_depth"].as<std::string>(),
            config["camera"]["pixel_format"].as<std::string>(),
            config["camera"]["acquisitionFrameRateEnable"].as<bool>(),
            config["camera"]["reverse_x"].as<bool>(false),
            config["camera"]["reverse_y"].as<bool>(false)
        );
        camera_->enableTrigger(TriggerType::Software, "Software", 0);
        camera_->setFrameCallback([this](ImageFrame& frame) {
            static bool first_is_inited = false;

            if (is_inited_) {
                frame.R_gimbal2odom =
                    Eigen::AngleAxisd(gimbal2camera_yaw_, Eigen::Vector3d::UnitZ())
                    * Eigen::AngleAxisd(gimbal2camera_pitch_, Eigen::Vector3d::UnitY())
                    * Eigen::AngleAxisd(gimbal2camera_roll_, Eigen::Vector3d::UnitX());
                frame.v = Eigen::Vector3d(this->index_, this->index_, this->index_);
                this->callback_(frame);
            } else {
                return;
            }
        });
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
    auto callback = [this](const ImageFrame& frame) {
        static bool first_is_inited = false;

        if (gobal::is_inited_) {
            img_recv_count_++;
            if (infer_running_count_.load() >= max_infer_running_) {
                return;
            }
            gobal::stringanyting.get_ptr<ThreadPool>("thread_pool")
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
    omni_num_ = config["common"]["omni_num"].as<size_t>(1);
    total_fps_ = config["common"]["total_fps"].as<int>(30);
    for (size_t i = 0; i < omni_num_; ++i) {
        YAML::Node omni_config = config["omni_vision"][i];
        auto omni_vision = std::make_unique<OmniVision>();
        if (!omni_vision->init(omni_config, callback, i)) {
            WUST_ERROR(vision_logger_) << "OmniVision initialization failed."
                                       << " Index: " << i;
        }

        omni_visions_.emplace_back(std::move(omni_vision));
    }
    initDetector();
    timer_ = std::make_unique<Timer>();
    double valid_duration = gobal::config["common"]["valid_duration"].as<double>(0.1);
    auto omni_queue = std::make_shared<TimedQueue<armor::OneTarget>>(valid_duration);
    gobal::stringanyting.set_ptr<TimedQueue<armor::OneTarget>>("omni_queue", omni_queue);
}
OmniManager::~OmniManager() {}
void OmniManager::stop() {
    for (auto& omni_vision: omni_visions_) {
        omni_vision->stop();
        omni_vision.reset();
    }
    omni_visions_.clear();
    if (timer_) {
        timer_->stop();
        timer_.reset();
    }

    armor_detector_.reset();

    WUST_INFO(vision_logger_) << "OmniManager shutdown complete.";
}
void OmniManager::initDetector() {
    max_infer_running_ = config_["common"]["max_infer_running"].as<int>(10);
    max_detect_armors_ = config_["common"]["max_detect_armors"].as<int>(10);
    std::string armor_detect_backend =
        config_["common"]["armor_detect_backend"].as<std::string>("");

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
        {
            auto common_info = gobal::stringanyting.get_value<CommonInfo>("common_info");
            common_info.use_detect_ncnn_count++;
            gobal::stringanyting.set_value("common_info", common_info);
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
        return DetectorFactory::createArmorDetector(backend, YAML::LoadFile(config_path), false);
    };
    if (armor_detect_backend.empty()) {
        throw std::runtime_error("armor_detect_backend not set in config.");
    }
    armor_detector_ = loadArmorDetectorBackend(armor_detect_backend);
    WUST_MAIN(vision_logger_) << "Using Armor Detector: " << armor_detect_backend;

    armor_detector_->setCallback(std::bind(
        &OmniManager::ArmorDetectCallback,
        this,
        std::placeholders::_1,
        std::placeholders::_2
    ));
}
void OmniManager::start() {
    if (timer_) {
        auto timercallback = std::bind(&OmniManager::timerCallback, this, std::placeholders::_1);
        double rate_hz = static_cast<double>(total_fps_);
        timer_->start(rate_hz, timercallback);
    }
}
void OmniManager::ArmorDetectCallback(
    const std::vector<armor::ArmorObject>& objs,
    const CommonFrame& frame
) {
    std::lock_guard<std::mutex> lock(callback_mutex_);

    std::vector<armor::ArmorObject> sorted_objs = objs;
    if (sorted_objs.size() > max_detect_armors_) {
        WUST_WARN(vision_logger_) << "Detected " << sorted_objs.size() << " objects"
                                  << ", too much, keeping top " << max_detect_armors_;
        std::partial_sort(
            sorted_objs.begin(),
            sorted_objs.begin() + max_detect_armors_,
            sorted_objs.end(),
            [](const armor::ArmorObject& a, const armor::ArmorObject& b) {
                return a.confidence > b.confidence;
            }
        );
        sorted_objs.resize(max_detect_armors_);
    }

    armor::Armors armors;
    armors.timestamp = frame.timestamp;
    armors.frame_id = "camera_optical_frame";

    cv::Mat camera_intrinsic_ = omni_visions_[frame.v.x()]->camera_intrinsic_;
    cv::Mat camera_distortion_ = omni_visions_[frame.v.x()]->camera_distortion_;

    mono_measure_tool::processDetectedArmors(
        sorted_objs,
        armors,
        frame.T_camera_to_odom,
        camera_intrinsic_,
        camera_distortion_
    );

    buildOneTargetsfromOmni(armors);

    cv::Mat debug_img = frame.src_img.clone();
    imgframe debug_img_frame;
    debug_img_frame.img = debug_img;
    debug_img_frame.timestamp = frame.timestamp;
    detect_finish_count_++;
    drawResult(debug_img_frame, armors);
}

std::vector<armor::OneTarget> OmniManager::buildOneTargetsfromOmni(const armor::Armors& armors) {
    std::vector<armor::OneTarget> one_targets;
    auto omni_queue = gobal::stringanyting.try_get_ptr<TimedQueue<armor::OneTarget>>("omni_queue");
    if (!omni_queue) {
        WUST_ERROR(vision_logger_) << "Omni queue not initialized.";
        return one_targets;
    }
    omni_queue->get()->clear_stale();
    for (const auto& armor: armors.armors) {
        if (armor.is_none_purple) {
            continue;
        }
        armor::OneTarget target;
        target.type = armor.type;
        target.id = armor.number;
        target.position_ = armor.target_pos;
        target.yaw = armor.yaw;
        target.distance_to_image_center = armor.distance_to_image_center;
        target.timestamp = armors.timestamp;
        target.tracking = true;
        target.is_omni = true;
        one_targets.push_back(target);
        omni_queue->get()->push(target, target.timestamp);
    }
    return one_targets;
}
void OmniManager::processImage(const ImageFrame& frame) {
    CommonFrame common_frame;

    common_frame.timestamp = frame.timestamp;
    common_frame.v = frame.v;
    if (!omni_visions_[common_frame.v.x()]->use_video) {
        common_frame.src_img = std::move(convertToMat(frame));
        if (common_frame.src_img.empty()) {
            WUST_ERROR(vision_logger_) << "Received empty image frame.";
            return;
        }
    } else {
        if (!frame.src_img.empty()) {
            common_frame.src_img = std::move(frame.src_img);
            common_frame.src_img.convertTo(
                common_frame.src_img,
                -1,
                omni_visions_[common_frame.v.x()]->video_alpha,
                omni_visions_[common_frame.v.x()]->video_beta
            );
        } else {
            WUST_ERROR(vision_logger_) << "Received empty image frame.";
            return;
        }
    }
    Eigen::Matrix3d R_camera_to_gimbal;
    R_camera_to_gimbal << 0, 0, 1, -1, 0, 0, 0, -1, 0;

    Eigen::Matrix4d T_camera_to_odom = utils::computeCameraToOdomTransform(
        frame.R_gimbal2odom,
        R_camera_to_gimbal,
        Eigen::Vector3d(0, 0, 0)
    );

    armor_detector_->pushInput(common_frame);
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
