#include "auto_aim.hpp"
#include "tasks/auto_aim/armor_control/aimer.hpp"
#include "tasks/auto_aim/armor_control/planner.hpp"
#include "tasks/auto_aim/armor_control/shooter.hpp"
#include "tasks/utils.hpp"
#include "wust_vl/common/concurrency/queues.hpp"
namespace auto_aim {
struct AutoAim::Impl {
    ~Impl() {
        run_flag_ = false;
        if (armor_detector_) {
            armor_detector_.reset();
        }
        if (processing_thread_) {
            processing_thread_->stop();
            wust_vl_concurrency::ThreadManager::instance().unregisterThread(
                processing_thread_->getName()
            );
        }
    }
    bool init(
        const YAML::Node& config,
        int& use_detect_ncnn_count,
        const Eigen::Matrix3d& R_camera2gimbal,
        const Eigen::Vector3d& t_camera2gimbal,
        const std::pair<cv::Mat, cv::Mat>& camera_info,
        std::shared_ptr<wust_vl_utils::ConfigBinder> config_binder
    ) {
        config_ = config;
        R_camera2gimbal_ = R_camera2gimbal;
        t_camera2gimbal_ = t_camera2gimbal;
        camera_info_ = camera_info;
        config_binder_ = config_binder;

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
                use_detect_ncnn_count++;
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
        armor_pose_estimator_ = std::make_unique<ArmorPoseEstimator>(config_, camera_info_);
        tracker_manager_ = std::make_unique<TrackerManager>(config_, config_binder_);
        auto_aim_fsm_cl_.load(config_);
        std::string comp_type =
            config["trajectory_compensator"]["compenstator_type"].as<std::string>("ideal");
        double gravity_ = config["trajectory_compensator"]["gravity"].as<double>(10.0);
        double resistance_ = config["trajectory_compensator"]["resistance"].as<double>(0.092);
        int iteration_times_ = config["trajectory_compensator"]["iteration_times"].as<int>(20);
        double rho = config["trajectory_compensator"]["rho"].as<double>(0.0);
        double Cd = config["trajectory_compensator"]["Cd"].as<double>(0.47);
        double A = config["trajectory_compensator"]["A"].as<double>(0.0);
        double mass = config["trajectory_compensator"]["mass"].as<double>(0.5);
        auto trajectory_compensator = CompensatorFactory::createCompensator(comp_type);
        trajectory_compensator->iteration_times_ = iteration_times_;
        trajectory_compensator->gravity_ = gravity_;
        trajectory_compensator->resistance_ = resistance_;
        trajectory_compensator->rho_ = rho;
        trajectory_compensator->Cd_ = Cd;
        trajectory_compensator->A_ = A;
        trajectory_compensator->mass_ = mass;
        aimer_ = std::make_shared<Aimer>();
        aimer_->init(config_, trajectory_compensator);
        shooter_ = std::make_shared<Shooter>();
        shooter_->init(config_, trajectory_compensator);
        planner_ = std::make_unique<Planner>(config_["planner"], aimer_, shooter_);
        max_detect_armors_ = config_["max_detect_armors"].as<int>(10);
        armor_queue_ = std::make_unique<OrderedQueue<armor::Armors>>(10, 500);
        latency_averager_ = std::make_unique<Averager<double>>(100);
        return true;
    }
    void start() {
        processing_thread_ = wust_vl_concurrency::MonitoredThread::create(
            "AutoAimProcessingThread",
            [this](std::shared_ptr<wust_vl_concurrency::MonitoredThread> self) {
                this->processingLoop(self);
            }
        );

        wust_vl_concurrency::ThreadManager::instance().registerThread(processing_thread_);
        run_flag_ = true;
    }
    void pushInput(CommonFrame& frame) {
        img_recv_count_++;
        if (armor_solver_target_.checkTargetAppear()) {
            auto bbox = armor_solver_target_.expanded(
                T_camera_to_odom_,
                camera_info_.first,
                camera_info_.second,
                frame.src_img.size()
            );
            if (bbox.area() > 100) {
                frame.expanded = bbox;
                frame.offset = cv::Point2f(bbox.x, bbox.y);
            }
        }
        if (armor_detector_) {
            armor_detector_->pushInput(frame);
        }
    }

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
        for (auto& obj: sorted_objs) {
            obj.addOffset(frame.offset);
        }
        armor::Armors armors;
        armors.timestamp = frame.timestamp;
        Eigen::Vector3d v = Eigen::Vector3d::Zero();
        Eigen::Matrix3d R_gimbal2odom = Eigen::Matrix3d::Identity();
        Eigen::Vector3d gimbal = Eigen::Vector3d::Zero();
        if (shared_->motion_buffer) {
            auto apply_motion = [&](const auto& att) {
                v = Eigen::Vector3d(att.data.vx, att.data.vy, att.data.vz);
                R_gimbal2odom = Eigen::AngleAxisd(att.data.yaw, Eigen::Vector3d::UnitZ())
                    * Eigen::AngleAxisd(-att.data.pitch, Eigen::Vector3d::UnitY())
                    * Eigen::AngleAxisd(att.data.roll, Eigen::Vector3d::UnitX());
                gimbal.x() = att.data.yaw;
                gimbal.y() = att.data.pitch;
                gimbal.z() = att.data.roll;
            };
            auto delay = std::chrono::microseconds(
                static_cast<int64_t>(std::round(shared_->communication_delay_μs))
            );
            auto t_query = armors.timestamp + delay;
            if (auto past_att = shared_->motion_buffer->get_interpolated(t_query)) {
                apply_motion(*past_att);
            } else if (auto last_att = shared_->motion_buffer->get_last()) {
                apply_motion(*last_att);
            }
        }

        T_camera_to_odom_ =
            utils::computeCameraToOdomTransform(R_gimbal2odom, R_camera2gimbal_, t_camera2gimbal_);
        armors.armors = armor_pose_estimator_->extractArmorPoses(
            sorted_objs,
            T_camera_to_odom_,
            camera_info_.first,
            camera_info_.second
        );

        armors.v = v;
        armors.id = frame.id;
        for (auto& armor: armors.armors) {
            armor.timestamp = armors.timestamp;
        }
        armor_queue_->enqueue(armors);
        detect_finish_count_++;
        if (debug_mode_) {
            std::lock_guard<std::mutex> lock(dbg_mutex_);
            auto_aim_debug_.src_img.img = std::move(frame.src_img);
            auto_aim_debug_.src_img.timestamp = armors.timestamp;
            auto_aim_debug_.armors = armors;
            auto_aim_debug_.T_camera_to_odom = T_camera_to_odom_;
            auto_aim_debug_.detect_color = frame.detect_color;
            auto_aim_debug_.armor_objs = sorted_objs;
            auto_aim_debug_.expanded = frame.expanded;
        }
    }
    void armorsCallback(const armor::Armors& armors) {
        if (armors.timestamp <= tracker_manager_->last_time_) {
            WUST_WARN(logger_) << "Received out-of-order armor data, discarded.";
            return;
        }
        Target target;
        if (!tracker_manager_) {
            std::cout << "cao" << std::endl;
            return;
        }

        target = tracker_manager_->update(armors, auto_aim_fsm_cl_);
        auto now = std::chrono::steady_clock::now();
        {
            std::lock_guard<std::mutex> lock(armor_solver_target_mutex_);
            armor_solver_target_ = target;
        }

        auto latency_ms = time_utils::durationMs(armors.timestamp, now);
        latency_averager_->add(latency_ms);
        auto_aim_debug_.latency_ms = latency_averager_->average();
        if (debug_mode_) {
            std::lock_guard<std::mutex> lock(dbg_mutex_);
            auto_aim_debug_.target = target;
            auto_aim_debug_.fsm = auto_aim_fsm_cl_.fsm_state_;
        }
    }

    GimbalCmd solve(double dt_ms) {
        GimbalCmd gimbal_cmd;
        Target target;
        {
            std::lock_guard<std::mutex> lock(armor_solver_target_mutex_);
            target = armor_solver_target_;
        }
        AimTarget aim_target;
        bool appear = target.checkTargetAppear();
        if (appear) {
            auto now = time_utils::now();
            GimbalCmd tmp_cmd =
                aimer_->aim(target, now, shared_->bullet_speed, auto_aim_fsm_cl_.fsm_state_);
            aim_target = tmp_cmd.aim_target;
            auto last_att = shared_->motion_buffer->get_last();
            gimbal_cmd = shooter_->shoot(
                tmp_cmd,
                0,
                0,
                shared_->bullet_speed,
                true,
                auto_aim_fsm_cl_.fsm_state_,
                last_att->data.vyaw
            );

            gimbal_cmd.raw_yaw = gimbal_cmd.yaw;
            gimbal_cmd.raw_pitch = gimbal_cmd.pitch;
            auto plan = planner_->plan(target, shared_->bullet_speed, auto_aim_fsm_cl_.fsm_state_);
            if (plan.control) {
                gimbal_cmd.yaw = plan.yaw / M_PI * 180.0;
                gimbal_cmd.v_yaw = plan.yaw_vel / M_PI * 180.0;
                gimbal_cmd.pitch = plan.pitch / M_PI * 180.0;
                gimbal_cmd.v_pitch = plan.pitch_vel / M_PI * 180.0;
                tmp_cmd.armor_posandyaw = plan.armor_posandyaw;
                auto only_check_fire = shooter_->shoot(
                    tmp_cmd,
                    gimbal_cmd.yaw * M_PI / 180.0,
                    gimbal_cmd.pitch * M_PI / 180.0,
                    shared_->bullet_speed,
                    true,
                    auto_aim_fsm_cl_.fsm_state_,
                    last_att->data.vyaw
                );
                gimbal_cmd.fire_advice = only_check_fire.fire_advice;
                gimbal_cmd.enable_pitch_diff = only_check_fire.enable_pitch_diff;
                gimbal_cmd.enable_yaw_diff = only_check_fire.enable_yaw_diff;
                gimbal_cmd.target_yaw = plan.target_yaw / M_PI * 180.0;
                gimbal_cmd.target_pitch = plan.target_pitch / M_PI * 180.0;
                gimbal_cmd.raw_yaw = gimbal_cmd.target_yaw;
                gimbal_cmd.raw_pitch = gimbal_cmd.target_pitch;
                aim_target = plan.aim_target;
            } else {
                gimbal_cmd.appera = false;
            }
        } else {
            gimbal_cmd.appera = false;
        }

        if (gimbal_cmd.fire_advice) {
            fire_count_++;
        }
        if (debug_mode_) {
            std::lock_guard<std::mutex> lock(dbg_mutex_);
            auto_aim_debug_.gimbal_cmd = gimbal_cmd;
            auto_aim_debug_.aim_target = aim_target;
        }
        timer_cout_++;
        return gimbal_cmd;
    }
    void processingLoop(std::shared_ptr<wust_vl_concurrency::MonitoredThread> self) {
        while (!self->isAlive()) {
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
        while (self->isAlive()) {
            if (!self->waitPoint())
                break;
            printStats();
            armor::Armors armors;
            self->heartbeat();
            if (!armor_queue_->try_dequeue(armors)) {
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
                continue;
            }
            armorsCallback(armors);
            tracker_finish_count_++;
        }
    }
    void setDebug(bool debug) {
        debug_mode_ = debug;
    }
    DebugArmor getDebugFrame() {
        std::lock_guard<std::mutex> lock(dbg_mutex_);
        return auto_aim_debug_;
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
            double found_ratio = 0.0;
            if (img_recv_count_ > 0) {
                found_ratio = static_cast<double>(tracker_manager_->tracker_v3_->found_count_)
                    / static_cast<double>(img_recv_count_);
            }

            WUST_INFO(logger_) << "Rec: " << img_recv_count_ << ", Det: " << detect_finish_count_
                               << ", Fin: " << tracker_finish_count_ << ", Tc: " << timer_cout_
                               << ", Lat: " << auto_aim_debug_.latency_ms << "ms"
                               << ", Fire: " << fire_count_
                               << ", Found: " << tracker_manager_->tracker_v3_->found_count_
                               << ", Found_ratio: " << found_ratio;

            //camera_->setHikExposureTime(exposure_time);
            img_recv_count_ = 0;
            detect_finish_count_ = 0;
            fire_count_ = 0;
            tracker_finish_count_ = 0;
            timer_cout_ = 0;
            tracker_manager_->tracker_v3_->found_count_ = 0;

            last_stat_time_steady_ = now;
        }
    }

    std::unique_ptr<ArmorDetectorBase> armor_detector_;
    std::unique_ptr<TrackerManager> tracker_manager_;
    std::string logger_ = "auto_aim";
    std::unique_ptr<OrderedQueue<armor::Armors>> armor_queue_;
    std::shared_ptr<wust_vl_concurrency::MonitoredThread> processing_thread_;
    std::unique_ptr<Timer> timer_;
    std::shared_ptr<Aimer> aimer_;
    std::shared_ptr<Shooter> shooter_;
    std::unique_ptr<Planner> planner_;
    std::unique_ptr<ArmorPoseEstimator> armor_pose_estimator_;
    AutoAimFsmController auto_aim_fsm_cl_;
    GimbalCmd last_cmd_;
    int max_detect_armors_;
    bool run_flag_ = false;
    int detect_finish_count_ = 0;
    int img_recv_count_ = 0;
    int tracker_finish_count_ = 0;
    int fire_count_ = 0;
    int timer_cout_ = 0;
    std::chrono::steady_clock::time_point last_stat_time_steady_ = std::chrono::steady_clock::now();
    Target armor_solver_target_;
    std::mutex armor_solver_target_mutex_;
    bool debug_mode_ = false;
    DebugArmor auto_aim_debug_;
    std::mutex dbg_mutex_;
    std::unique_ptr<Averager<double>> latency_averager_;
    Eigen::Matrix3d R_camera2gimbal_;
    Eigen::Vector3d t_camera2gimbal_;
    std::pair<cv::Mat, cv::Mat> camera_info_;
    YAML::Node config_;
    std::shared_ptr<wust_vl_utils::ConfigBinder> config_binder_;
    std::shared_ptr<AutoAimShared> shared_;
    Eigen::Matrix4d T_camera_to_odom_;
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
    const Eigen::Vector3d& t_camera2gimbal,
    const std::pair<cv::Mat, cv::Mat>& camera_info,
    std::shared_ptr<wust_vl_utils::ConfigBinder> config_binder
) {
    return _impl->init(
        config,
        use_detect_ncnn_count,
        R_camera2gimbal,
        t_camera2gimbal,
        camera_info,
        config_binder
    );
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
void AutoAim::setShared(std::shared_ptr<AutoAimShared> shared) {
    _impl->setShared(shared);
}
GimbalCmd AutoAim::solve(double dt_ms) {
    return _impl->solve(dt_ms);
}
bool AutoAim::isActive() {
    if (_impl->processing_thread_->getStatus()
        == wust_vl_concurrency::MonitoredThread::Status::Running) {
        return true;
    } else {
        return false;
    }
}
void AutoAim::processingWait() {
    _impl->processing_thread_->pause();
}
void AutoAim::processingUp() {
    _impl->processing_thread_->resume();
}
} // namespace auto_aim