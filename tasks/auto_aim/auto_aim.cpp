#include "auto_aim.hpp"
#include "tasks/mono_measure_tool.hpp"
#include "tasks/utils.hpp"
#include "wust_vl/common/concurrency/queues.hpp"
namespace auto_aim {
struct AutoAim::Impl {
    ~Impl() {
        run_flag_ = false;
        if (armor_detector_) {
            armor_detector_.reset();
        }
        if (processing_thread_.joinable()) {
            processing_thread_.join();
        }
    }
    bool init(
        const YAML::Node& config,
        int& use_detect_ncnn_count,
        const Eigen::Matrix3d& R_camera2gimbal,
        const Eigen::Vector3d& t_gimbal_to_camera,
        const std::pair<cv::Mat, cv::Mat>& camera_info
    ) {
        config_ = config;
        R_camera2gimbal_ = R_camera2gimbal;
        t_gimbal_to_camera_ = t_gimbal_to_camera;
        camera_info_ = camera_info;

        std::string armor_detect_backend = config_["armor_detect_backend"].as<std::string>("");
        auto isBackendEnabled = [&use_detect_ncnn_count](const std::string& backend) -> bool {
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
                // auto common_info = gobal::stringanything.get_value<CommonInfo>("common_info");
                use_detect_ncnn_count++;
                //gobal::stringanything.set_value<CommonInfo>("common_info", common_info);
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
        armor_detector_->setCallback(std::bind(
            &AutoAim::Impl::ArmorDetectCallback,
            this,
            std::placeholders::_1,
            std::placeholders::_2
        ));
        WUST_MAIN(logger_) << "Using Armor Detector: " << armor_detect_backend;
        armor_pose_estimator_ = std::make_unique<ArmorPoseEstimator>(config, camera_info_);
        bool use_ba = config_["use_ba"].as<bool>(false);
        armor_pose_estimator_->enableBA(use_ba);
        tracker_manager_ = std::make_unique<TrackerManager>(config_["armor_tracker"]);
        armor_solver_ = std::make_unique<ArmorSolver>(config_);
        armor_solver_->setStateCallback(
            std::bind(&AutoAim::Impl::armorSloverStateCallback, this, std::placeholders::_1)
        );
        max_detect_armors_ = config_["max_detect_armors"].as<int>(10);
        armor_queue_ = std::make_unique<OrderedQueue<armor::Armors>>(10, 500);
        latency_averager_ = std::make_unique<Averager<double>>(100);
        return true;
    }
    void start() {
        processing_thread_ = std::thread(&AutoAim::Impl::processingLoop, this);
        run_flag_ = true;
    }
    void pushInput(CommonFrame& frame) {
        img_recv_count_++;
        if (armor_detector_) {
            armor_detector_->pushInput(frame);
        }
    }
    void
    ArmorDetectCallback(const std::vector<armor::ArmorObject>& objs, const CommonFrame& frame) {
        std::lock_guard<std::mutex> lock(callback_mutex_);
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
        // auto camera_info =
        //     gobal::stringanything.get_value<std::pair<cv::Mat, cv::Mat>>("camera_info");
        armors.armors = armor_pose_estimator_->extractArmorPoses(
            sorted_objs,
            frame.T_camera_to_odom,
            camera_info_.first,
            camera_info_.second
        );

        mono_measure_tool::processDetectedArmors(
            sorted_objs,
            armors,
            frame.T_camera_to_odom,
            camera_info_.first,
            camera_info_.second
        );
        // Eigen::Matrix3d R_camera2gimbal =
        //     gobal::stringanything.get_value<Eigen::Matrix3d>("R_camera2gimbal");
        // Eigen::Vector3d t_gimbal_to_camera =
        //     gobal::stringanything.get_value<Eigen::Vector3d>("t_gimbal_to_camera");
        Eigen::Matrix3d R_gimbal2odom =
            utils::getRGimbalToOdom(frame.T_camera_to_odom, R_camera2gimbal_, t_gimbal_to_camera_);
        armors.R_gimbal2odom = R_gimbal2odom;
        armors.v = frame.v;
        armors.id = frame.id;
        armor_queue_->enqueue(armors);
        detect_finish_count_++;
        if (debug_mode_) {
            std::lock_guard<std::mutex> lock(dbg_mutex_);
            auto_aim_debug_.src_img.img = frame.src_img.clone();
            auto_aim_debug_.src_img.timestamp = armors.timestamp;
            auto_aim_debug_.armors = armors;
            auto_aim_debug_.T_camera_to_odom = frame.T_camera_to_odom;
        }
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
        auto now = std::chrono::steady_clock::now();

        auto latency_ms = time_utils::durationMs(armors.timestamp, now);
        latency_averager_->add(latency_ms);
        auto_aim_debug_.latency_ms = latency_averager_->average();
        if (debug_mode_) {
            std::lock_guard<std::mutex> lock(dbg_mutex_);
            auto_aim_debug_.target = target;
            auto_aim_debug_.one_targets = one_targets;
        }
    }
    ArmorSolver::StateResult armorSloverStateCallback(const ArmorSolver::ArmorSloveState state) {
        auto result = ArmorSolver::StateResult();
        if (shared_->motion_buffer) {
            auto last_att = shared_->motion_buffer->get_last();
            if (last_att) {
                result.rpy[0] = last_att->roll;
                result.rpy[1] = last_att->pitch;
                result.rpy[2] = last_att->yaw;
            }
        }
        result.bullet_speed = shared_->bullet_speed;
        result.controller_delay = shared_->controller_delay;
        armor_slove_state_ = state;

        return result;
    }
    GimbalCmd solve() {
        GimbalCmd gimbal_cmd;
        armor::Target target;
        std::vector<armor::OneTarget> one_targets;
        {
            std::lock_guard<std::mutex> lock(armor_solver_target_mutex_);
            target = armor_solver_target_.target;
            one_targets = armor_solver_target_.one_targets;
        }
        bool appear = utils::checkTargetAppear(target, one_targets);
        Tracker::State state = appear ? Tracker::TRACKING : Tracker::LOST;
        if (appear) {
            auto now = std::chrono::steady_clock::now();
            try {
                gimbal_cmd = armor_solver_->solve(armor_solver_target_, now);
            } catch (const std::exception& e) {
                gimbal_cmd = armor_solver_->returnDefaultCmd();
            }
        } else {
            gimbal_cmd = armor_solver_->returnDefaultCmd();
        }
        if (gimbal_cmd.fire_advice) {
            fire_count_++;
        }
        if (debug_mode_) {
            std::lock_guard<std::mutex> lock(dbg_mutex_);
            auto_aim_debug_.gimbal_cmd = gimbal_cmd;
        }
        return gimbal_cmd;
    }
    void processingLoop() {
        while (!run_flag_) {
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
        while (run_flag_) {
            armor::Armors armors_;
            if (!armor_queue_->try_dequeue(armors_)) {
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
                continue;
            }

            armorsCallback(armors_);
            tracker_finish_count_++;
            printStats();
        }
    }
    void setDebug(bool debug) {
        debug_mode_ = debug;
    }
    DebugArmor getDebugFrame() {
        std::lock_guard<std::mutex> lock(dbg_mutex_);
        return auto_aim_debug_;
    }
    ArmorSolverTarget getArmorSolverTarget() {
        std::lock_guard<std::mutex> lock(armor_solver_target_mutex_);
        return armor_solver_target_;
    }
    void printStats() {
        using namespace std::chrono;
        auto now = steady_clock::now();

        if (last_stat_time_steady_.time_since_epoch().count() == 0) {
            last_stat_time_steady_ = now;
            return;
        }

        auto elapsed = duration_cast<duration<double>>(now - last_stat_time_steady_);
        if (elapsed.count() >= 1.0) {
            WUST_INFO(logger_) << "Rec: " << img_recv_count_ << ", Det: " << detect_finish_count_
                               << ", Fin: " << tracker_finish_count_
                               << ", Lat: " << auto_aim_debug_.latency_ms << "ms"
                               << ", Fire: " << fire_count_;
            img_recv_count_ = 0;
            detect_finish_count_ = 0;
            fire_count_ = 0;
            tracker_finish_count_ = 0;
            last_stat_time_steady_ = now;
        }
    }
    std::mutex callback_mutex_;
    std::unique_ptr<ArmorDetectorBase> armor_detector_;
    std::unique_ptr<TrackerManager> tracker_manager_;
    std::unique_ptr<ArmorPoseEstimator> armor_pose_estimator_;
    std::string logger_ = "auto_aim";
    std::unique_ptr<OrderedQueue<armor::Armors>> armor_queue_;
    std::thread processing_thread_;
    std::unique_ptr<Timer> timer_;
    std::unique_ptr<ArmorSolver> armor_solver_;
    ArmorSolver::ArmorSloveState armor_slove_state_ = ArmorSolver::TRACKING_ARMOR;
    int max_detect_armors_;
    bool run_flag_ = false;
    int detect_finish_count_ = 0;
    int img_recv_count_ = 0;
    int tracker_finish_count_ = 0;
    int fire_count_;
    std::chrono::steady_clock::time_point last_stat_time_steady_ = std::chrono::steady_clock::now();
    ArmorSolverTarget armor_solver_target_;
    std::mutex armor_solver_target_mutex_;
    bool debug_mode_ = false;
    DebugArmor auto_aim_debug_;
    std::mutex dbg_mutex_;
    std::unique_ptr<Averager<double>> latency_averager_;
    Eigen::Matrix3d R_camera2gimbal_;
    Eigen::Vector3d t_gimbal_to_camera_;
    std::pair<cv::Mat, cv::Mat> camera_info_;
    YAML::Node config_;
    std::shared_ptr<AutoAimShared> shared_;
    void setShared(std::shared_ptr<AutoAimShared> shared) {
        shared_ = shared;
    }
};
AutoAim::AutoAim(): _impl(std::make_unique<Impl>()) {}
AutoAim::~AutoAim() {
    _impl.reset();
}
bool AutoAim::init(
    const YAML::Node& config,
    int& use_detect_ncnn_count,
    const Eigen::Matrix3d& R_camera2gimbal,
    const Eigen::Vector3d& t_gimbal_to_camera,
    const std::pair<cv::Mat, cv::Mat>& camera_info
) {
    return _impl
        ->init(config, use_detect_ncnn_count, R_camera2gimbal, t_gimbal_to_camera, camera_info);
}
void AutoAim::start() {
    _impl->start();
}
void AutoAim::pushInput(CommonFrame& frame) {
    _impl->pushInput(frame);
}
void AutoAim::setDebug(bool debug) {
    _impl->setDebug(debug);
}
DebugArmor AutoAim::getDebugFrame() {
    return _impl->getDebugFrame();
}
ArmorSolverTarget AutoAim::getArmorSolverTarget() {
    return _impl->getArmorSolverTarget();
}
void AutoAim::setShared(std::shared_ptr<AutoAimShared> shared) {
    _impl->setShared(shared);
}
GimbalCmd AutoAim::solve() {
    return _impl->solve();
}
} // namespace auto_aim