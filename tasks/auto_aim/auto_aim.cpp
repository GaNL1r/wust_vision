#include "auto_aim.hpp"
#include "tasks/auto_aim/armor_control/very_aimer_factory.hpp"
#include "tasks/auto_aim/armor_detect/armor_pose_estimator.hpp"
#include "tasks/auto_aim/armor_tracker/trackerv3.hpp"
#include "tasks/config.hpp"
#include "wust_vl/common/concurrency/queues.hpp"
// clang-format off
#include "tasks/auto_aim/armor_detect/armor_detector_factory.hpp"
// clang-format on
namespace wust_vision {
namespace auto_aim {

    struct AutoAim::Impl {
        ~Impl() {
            run_flag_ = false;
            if (armor_detector_) {
                armor_detector_.reset();
            }
            armor_queue_->stop();
            if (processing_thread_) {
                processing_thread_->stop();
                wust_vl::common::concurrency::ThreadManager::instance().unregisterThread(
                    processing_thread_->getName()
                );
            }
        }
        bool init(
            const std::string& config_path,
            int& use_detect_ncnn_count,
            TFConfig::Ptr tf_config,
            const std::pair<cv::Mat, cv::Mat>& camera_info
        ) {
            tf_config_ = tf_config;
            camera_info_ = camera_info;
            auto_aim_config_parameter_ = wust_vl::common::utils::Parameter::create();
            auto config = YAML::LoadFile(config_path);
            auto_aim_config_parameter_->loadFromFile(config_path);
            auto_exposure_cfg_ = AutoExposureCfg::create();
            very_aimer_ =
                VeryAimerFactory::create(config["very_aimer"], auto_aim_config_parameter_);
            auto_aim_config_parameter_->registerGroup(*very_aimer_->trajectory_compensator_config_);
            auto_aim_config_parameter_->registerGroup(*auto_exposure_cfg_);
            auto_aim_config_parameter_->registerGroup(*auto_aim_fsm_cl_.config_);
            auto_aim_config_parameter_->registerGroup(*very_aimer_->config_);
            tracker_ = Tracker::create(auto_aim_config_parameter_);
            auto_aim_config_parameter_->reloadFromOldPath();

            wust_vl::common::utils::ParameterManager::instance().registerParameter(
                *auto_aim_config_parameter_.get()
            );
            max_detect_armors_ = config["max_detect_armors"].as<int>(10);
            armor_pose_estimator_ = std::make_unique<ArmorPoseEstimator>(config, camera_info_);

            const std::string armor_detect_backend =
                config["armor_detect_backend"].as<std::string>("");
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
                if (backend == "opencv")
                    return OPENCV_CONFIG;
                else
                    return ML_CONFIG;
                return "";
            };

            auto loadArmorDetectorBackend = [&](const std::string& backend) {
                if (!isBackendEnabled(backend)) {
                    throw std::runtime_error(
                        "Backend " + backend + " is not enabled at compile time."
                    );
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
            armor_queue_ =
                std::make_unique<wust_vl::common::concurrency::OrderedQueue<Armors>>(50, 500);
            latency_averager_ =
                std::make_unique<wust_vl::common::concurrency::Averager<double>>(100);

            return true;
        }
        void start() {
            run_flag_ = true;
            processing_thread_ = wust_vl::common::concurrency::MonitoredThread::create(
                "AutoAimProcessingThread",
                [this](wust_vl::common::concurrency::MonitoredThread::Ptr self) {
                    this->processingLoop(self);
                }
            );
            wust_vl::common::concurrency::ThreadManager::instance().registerThread(
                processing_thread_
            );
        }
        void pushInput(CommonFrame& frame) {
            img_recv_count_++;

            auto bbox = target_.expanded(
                T_camera_to_odom_,
                camera_info_.first,
                camera_info_.second,
                frame.src_img.size()
            );
            if (bbox.area() > 100) {
                frame.expanded = bbox;
                frame.offset = cv::Point2f(bbox.x, bbox.y);
            }
            expanded_ = frame.expanded;
            const std::optional<ArmorNumber> target_number = target_.getArmorNumber();
            if (armor_detector_) {
                armor_detector_->pushInput(frame, target_number);
            }
        }

        void ArmorDetectCallback(const std::vector<ArmorObject>& objs, const CommonFrame& frame) {
            std::vector<ArmorObject> sorted_objs = objs;

            if (sorted_objs.size() > max_detect_armors_) [[unlikely]] {
                WUST_WARN(logger_) << "Detected " << sorted_objs.size()
                                   << " objects, too many, keeping top " << max_detect_armors_;

                std::partial_sort(
                    sorted_objs.begin(),
                    sorted_objs.begin() + max_detect_armors_,
                    sorted_objs.end(),
                    [](const ArmorObject& a, const ArmorObject& b) {
                        return a.confidence > b.confidence;
                    }
                );

                sorted_objs.resize(max_detect_armors_);
            }
            for (auto& obj: sorted_objs) {
                obj.addOffset(frame.offset);
            }

            Armors armors;
            armors.timestamp = frame.timestamp;
            armors.id = frame.id;
            Eigen::Vector3d v = Eigen::Vector3d::Zero();
            Eigen::Matrix3d R_gimbal2odom = Eigen::Matrix3d::Identity();
            if (shared_->motion_buffer) {
                const auto delay =
                    std::chrono::microseconds(static_cast<int64_t>(shared_->communication_delay_μs)
                    );
                const auto t_query = armors.timestamp + delay;
                auto apply_motion = [&](const auto& att) {
                    v << att.data.vx, att.data.vy, att.data.vz;
                    R_gimbal2odom = Eigen::AngleAxisd(att.data.yaw, Eigen::Vector3d::UnitZ())
                        * Eigen::AngleAxisd(-att.data.pitch, Eigen::Vector3d::UnitY())
                        * Eigen::AngleAxisd(att.data.roll, Eigen::Vector3d::UnitX());
                };
                if (auto past_att = shared_->motion_buffer->get_interpolated(t_query)) {
                    apply_motion(*past_att);
                } else if (auto last_att = shared_->motion_buffer->get_last()) {
                    apply_motion(*last_att);
                }
            }
            T_camera_to_odom_ = utils::computeCameraToOdomTransform(
                R_gimbal2odom,
                tf_config_->R_camera2gimbal,
                tf_config_->t_camera2gimbal
            );
            armors.armors = armor_pose_estimator_->extractArmorPoses(
                sorted_objs,
                T_camera_to_odom_,
                camera_info_.first,
                camera_info_.second
            );
            armors.v = v;
            for (auto& armor: armors.armors) {
                armor.timestamp = armors.timestamp;
            }
            armor_queue_->enqueue(armors);
            ++detect_finish_count_;
            if (debug_mode_) {
                std::lock_guard<std::mutex> lock(dbg_mutex_);
                auto_aim_debug_.src_img.img = frame.src_img;
                auto_aim_debug_.src_img.timestamp = armors.timestamp;
                auto_aim_debug_.armors = armors;
                auto_aim_debug_.T_camera_to_odom = T_camera_to_odom_;
                auto_aim_debug_.detect_color = frame.detect_color;
                auto_aim_debug_.armor_objs = sorted_objs;
                auto_aim_debug_.expanded = frame.expanded;
            }
        }
        void armorsCallback(const Armors& armors) {
            if (armors.timestamp <= tracker_->getLastTime()) {
                WUST_WARN(logger_) << "Received out-of-order armor data, discarded.";
                return;
            }
            Target target = tracker_->track(armors);
            auto_aim_fsm_cl_.update(std::abs(target.v_yaw()), target.jumped);
            const auto now = std::chrono::steady_clock::now();
            {
                std::lock_guard<std::mutex> lock(target_mutex_);
                target_ = target;
            }
            const auto latency_ms =
                wust_vl::common::utils::time_utils::durationMs(armors.timestamp, now);
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
                std::lock_guard<std::mutex> lock(target_mutex_);
                target = target_;
            }
            AimTarget aim_target;
            const bool appear = target.checkTargetAppear();
            if (appear && target.position().norm() > 0.5) {
                try {
                    gimbal_cmd = very_aimer_->veryAim(
                        target,
                        shared_->bullet_speed,
                        auto_aim_fsm_cl_.fsm_state_
                    );
                    aim_target = gimbal_cmd.aim_target;
                } catch (...) {
                    WUST_ERROR(logger_) << "VeryAim error";
                }
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
        void processingLoop(wust_vl::common::concurrency::MonitoredThread::Ptr self) {
            while (!self->isAlive()) {
                std::this_thread::sleep_for(std::chrono::milliseconds(10));
            }
            while (self->isAlive() && run_flag_) {
                if (!self->waitPoint())
                    break;
                self->heartbeat();
                printStats();
                Armors armors;
                bool skip;
                if (armor_queue_->dequeue_wait(armors, skip)) {
                    armorsCallback(armors);
                    tracker_finish_count_++;
                    if (skip) {
                        WUST_DEBUG(logger_) << "OrderQueue skip";
                    }
                }
                // if (!armor_queue_->try_dequeue(armors)) {
                //     std::this_thread::sleep_for(std::chrono::milliseconds(1));
                //     continue;
                // }
                // armorsCallback(armors);
                // tracker_finish_count_++;
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
            utils::XSecOnce(
                [&] {
                    double found_ratio = 0.0;
                    if (img_recv_count_ > 0) {
                        found_ratio = static_cast<double>(tracker_->getFoundCount())
                            / static_cast<double>(img_recv_count_);
                    }

                    WUST_INFO(logger_)
                        << "Rec: " << img_recv_count_ << ", Det: " << detect_finish_count_
                        << ", Fin: " << tracker_finish_count_ << ", Tc: " << timer_cout_
                        << ", Lat: " << auto_aim_debug_.latency_ms << "ms"
                        << ", Fire: " << fire_count_ << ", Found: " << tracker_->getFoundCount()
                        << ", Found_ratio: " << found_ratio;

                    img_recv_count_ = 0;
                    detect_finish_count_ = 0;
                    fire_count_ = 0;
                    tracker_finish_count_ = 0;
                    timer_cout_ = 0;
                    tracker_->setFoundCount(0);
                },
                1.0
            );
        }
        void
        autoExposureControl(const cv::Mat& frame, std::shared_ptr<wust_vl::video::Camera> camera) {
            const double dt = auto_exposure_cfg_->control_interval_ms_param.get() / 1000.0;
            utils::XSecOnce(
                [&] {
                    if (!auto_exposure_cfg_->enable_param.get() || frame.empty()) {
                        return;
                    }
                    if (auto* hik = dynamic_cast<wust_vl::video::HikCamera*>(camera->getDevice())) {
                        cv::Mat i_use = frame(expanded_);
                        if (expanded_.area() < 100 || i_use.empty()) {
                            i_use = frame;
                        }
                        const double brightness = utils::computeBrightness(i_use);

                        const double diff =
                            brightness - auto_exposure_cfg_->target_brightness_param.get();
                        const double exposure_min = auto_exposure_cfg_->exposure_min_param.get();
                        const double exposure_max = auto_exposure_cfg_->exposure_max_param.get();
                        double exposure_time = hik->getExposureTime();
                        const double last_exposure_time = exposure_time;
                        if (std::fabs(diff) > auto_exposure_cfg_->tolerance_param.get()
                            && exposure_time > 0.0) {
                            exposure_time -= diff * auto_exposure_cfg_->step_gain_param.get();
                        } else {
                            exposure_time -= auto_exposure_cfg_->decay_step_param.get();
                        }
                        if (exposure_time < exposure_min)
                            exposure_time = exposure_min;
                        if (exposure_time > exposure_max)
                            exposure_time = exposure_max;
                        if (std::abs(exposure_time - last_exposure_time) > 10) {
                            hik->setExposureTime(exposure_time);
                        }
                    }
                },
                dt
            );
        }
        wust_vl::common::utils::Parameter::Ptr auto_aim_config_parameter_;
        Tracker::Ptr tracker_;
        ArmorDetectorBase::Ptr armor_detector_;
        std::string logger_ = "auto_aim";
        std::unique_ptr<wust_vl::common::concurrency::OrderedQueue<Armors>> armor_queue_;
        wust_vl::common::concurrency::MonitoredThread::Ptr processing_thread_;
        std::unique_ptr<wust_vl::common::utils::Timer> timer_;
        VeryAimerBase::Ptr very_aimer_;
        std::unique_ptr<ArmorPoseEstimator> armor_pose_estimator_;
        AutoAimFsmController auto_aim_fsm_cl_;
        AutoExposureCfg::Ptr auto_exposure_cfg_;
        cv::Rect expanded_;
        GimbalCmd last_cmd_;
        int max_detect_armors_;
        bool run_flag_ = false;
        int detect_finish_count_ = 0;
        int img_recv_count_ = 0;
        int tracker_finish_count_ = 0;
        int fire_count_ = 0;
        int timer_cout_ = 0;
        Target target_;
        bool debug_mode_ = false;
        DebugArmor auto_aim_debug_;
        std::unique_ptr<wust_vl::common::concurrency::Averager<double>> latency_averager_;
        TFConfig::Ptr tf_config_;
        std::pair<cv::Mat, cv::Mat> camera_info_;
        std::shared_ptr<AutoAimShared> shared_;
        Eigen::Matrix4d T_camera_to_odom_;
        std::mutex target_mutex_;
        std::mutex dbg_mutex_;
        void setShared(std::shared_ptr<AutoAimShared> shared) {
            shared_ = shared;
        }
    };
    AutoAim::AutoAim(): _impl(std::make_unique<Impl>()) {}
    AutoAim::~AutoAim() {
        _impl.reset();
    }
    bool AutoAim::init(
        const std::string& config_path,
        int& use_detect_ncnn_count,
        TFConfig::Ptr tf_config,
        const std::pair<cv::Mat, cv::Mat>& camera_info
    ) {
        return _impl->init(config_path, use_detect_ncnn_count, tf_config, camera_info);
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
            == wust_vl::common::concurrency::MonitoredThread::Status::Running)
        {
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
    void AutoAim::autoExposureControl(
        const cv::Mat& frame,
        std::shared_ptr<wust_vl::video::Camera> camera
    ) {
        _impl->autoExposureControl(frame, camera);
    }
} // namespace auto_aim
} // namespace wust_vision