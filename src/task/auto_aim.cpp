#include "task/auto_aim.hpp"
#include "common/utils.hpp"
struct AutoAim::Impl {
    bool init(YAML::Node config) {
        std::string armor_detect_backend =
            config["common"]["armor_detect_backend"].as<std::string>("");
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
            auto armor_detect_config = YAML::LoadFile(config_path);
            return DetectorFactory::createArmorDetector(backend, armor_detect_config, true);
        };
        if (armor_detect_backend.empty()) {
            throw std::runtime_error("armor_detect_backend not set in config.");
        }
        armor_detector_ = loadArmorDetectorBackend(armor_detect_backend);
        WUST_MAIN(logger_) << "Using Armor Detector: " << armor_detect_backend;
        armor_pose_estimator_ = std::make_unique<ArmorPoseEstimator>();
        bool use_ba = config["common"]["use_ba"].as<bool>(false);
        if (use_ba) {
            armor_pose_estimator_->enableBA(true);
        } else {
            armor_pose_estimator_->enableBA(false);
        }
        tracker_manager_ = std::make_unique<TrackerManager>(config["armor_tracker"]);
        armor_detector_->setCallback(std::bind(
            &AutoAim::Impl::ArmorDetectCallback,
            this,
            std::placeholders::_1,
            std::placeholders::_2
        ));
        max_detect_armors_ = gobal::config["common"]["max_detect_armors"].as<int>(10);
    }
    void pushInput(const CommonFrame& frame);
    void
    ArmorDetectCallback(const std::vector<armor::ArmorObject>& objs, const CommonFrame& frame) {
        std::vector<armor::ArmorObject> sorted_objs = objs;

        if (sorted_objs.size() > max_detect_armors_) {
            WUST_WARN(logger_) << "Detected " << sorted_objs.size()
                               << " objects, too many, keeping top " << max_detect_armors_;

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
        auto camera_info =
            gobal::stringanything.get_value<std::pair<cv::Mat, cv::Mat>>("camera_info");
        armors.armors = armor_pose_estimator_->extractArmorPoses(
            sorted_objs,
            frame.T_camera_to_odom,
            camera_info.first,
            camera_info.second
        );

        mono_measure_tool::processDetectedArmors(
            sorted_objs,
            armors,
            frame.T_camera_to_odom,
            camera_info.first,
            camera_info.second
        );
        Eigen::Matrix3d R_camera2gimbal =
            gobal::stringanything.get_value<Eigen::Matrix3d>("R_camera2gimbal");
        Eigen::Vector3d t_gimbal_to_camera =
            gobal::stringanything.get_value<Eigen::Vector3d>("t_gimbal_to_camera");
        Eigen::Matrix3d R_gimbal2odom =
            utils::getRGimbalToOdom(frame.T_camera_to_odom, R_camera2gimbal, t_gimbal_to_camera);
        armors.R_gimbal2odom = R_gimbal2odom;
        armors.v = frame.v;
        armors.id = frame.id;
        armor_queue_->enqueue(armors);
        detect_finish_count_++;
        // auto common_info = gobal::stringanything.get_value<CommonInfo>("common_info");
        // if (common_info.debug_mode) {
        //     std::lock_guard<std::mutex> lock(dbg_mutex_);
        //     debug_gobal_frame_.imgframe.img = frame.src_img.clone();
        //     debug_gobal_frame_.imgframe.timestamp = armors.timestamp;
        //     debug_gobal_frame_.armors_gobal = armors;
        // }
    }
    void armorsCallback(armor::Armors armors) {
        if (armors.timestamp <= tracker_manager_->last_time_) {
            WUST_WARN(logger_) << "Received out-of-order armor data, discarded.";
            return;
        }
        armor::Target target;
        std::vector<armor::OneTarget> one_targets;
        auto time = armors.timestamp;

        tracker_manager_->update(target, one_targets, armors, time, armors.R_gimbal2odom, armors.v);
        {
            std::lock_guard<std::mutex> lock(armor_solver_target_mutex_);
            armor_solver_target_.target = target;
            armor_solver_target_.one_targets = one_targets;
        }

        // auto now = std::chrono::steady_clock::now();

        // auto latency_nano =
        //     std::chrono::duration_cast<std::chrono::nanoseconds>(now - target.timestamp).count();
        // latency_averager_.add(latency_nano);
        // toolsgobal::latency_ms = latency_averager_.average_ms();
        // auto common_info = gobal::stringanything.get_value<CommonInfo>("common_info");
        // if (common_info.debug_mode) {
        //     std::lock_guard<std::mutex> lock(dbg_mutex_);
        //     debug_gobal_frame_.armor_target = target;
        //     debug_gobal_frame_.one_armor_targets = one_targets;
        // }
    }
    void processingLoop() {
        while (!run_flag_) {
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
        while (run_flag_) {
            auto gobal_state = gobal::stringanything.get_value<GobalState>("gobal_state");
            auto mode = toAttackMode(gobal_state.attack_mode);

            armor::Armors armors_;
            if (!armor_queue_->try_dequeue(armors_)) {
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
                continue;
            }

            armorsCallback(armors_);
            tracker_finish_count_++;
        }
    }
    std::unique_ptr<ArmorDetectorBase> armor_detector_;
    std::unique_ptr<TrackerManager> tracker_manager_;
    std::unique_ptr<ArmorPoseEstimator> armor_pose_estimator_;
    std::string logger_ = "auto_aim";
    std::unique_ptr<OrderedQueue<armor::Armors>> armor_queue_;
    std::thread processing_thread_;
    std::unique_ptr<Timer> timer_;
    std::unique_ptr<ArmorSolver> armor_solver_;
    int max_detect_armors_;
    bool run_flag_ = false;
    int detect_finish_count_ = 0;
    int tracker_finish_count_ = 0;
    ArmorSolverTarget armor_solver_target_;
    std::mutex armor_solver_target_mutex_;
};