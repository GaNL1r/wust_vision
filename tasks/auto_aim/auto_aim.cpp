#include "auto_aim.hpp"
#include "tasks/auto_aim/armor_control/very_aimer.hpp"
#include "tasks/auto_aim/armor_detect/armor_detector_factory.hpp"
#include "tasks/auto_aim/armor_tracker/target.hpp"
#include "tasks/auto_aim/armor_tracker/trackerv3.hpp"
#include "tasks/auto_aim/armor_where/armor_where.hpp"
#include "tasks/auto_aim/debug.hpp"
#include "tasks/type_common.hpp"
#include "tasks/utils/config.hpp"
#include "wust_vl/common/concurrency/queues.hpp"

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
        Impl(
            const std::string& config_path,
            TFConfig::Ptr tf_config,
            const std::pair<cv::Mat, cv::Mat>& camera_info,
            bool debug
        ) {
            tf_config_ = tf_config;
            camera_info_ = camera_info;
            auto_aim_config_parameter_ = wust_vl::common::utils::Parameter::create();
            auto config = YAML::LoadFile(config_path);
            auto_aim_config_parameter_->loadFromFile(config_path);
            auto_exposure_cfg_ = AutoExposureCfg::create();
            very_aimer_ = VeryAimer::create(auto_aim_config_parameter_);
            auto_aim_config_parameter_->registerGroup(*auto_exposure_cfg_);
            auto_aim_config_parameter_->registerGroup(*auto_aim_fsm_cl_.config_);
            tracker_ = Tracker::create(auto_aim_config_parameter_);
            auto_aim_config_parameter_->reloadFromOldPath();

            wust_vl::common::utils::ParameterManager::instance().registerParameter(
                *auto_aim_config_parameter_.get()
            );
            max_detect_armors_ = config["max_detect_armors"].as<int>(10);
            armor_where_ = ArmorWhere::create(config["armor_where"], camera_info_);
            const std::string armor_detect_backend =
                config["armor_detect_backend"].as<std::string>("");
            armor_detector_ = DetectorFactory::createArmorDetector(armor_detect_backend, true);
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
        }
        void start() {
            if (run_flag_) {
                return;
            }
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
                frame.img_frame.src_img.size()
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
            armors.timestamp = frame.img_frame.timestamp;
            armors.id = frame.id;
            Eigen::Vector3d v = Eigen::Vector3d::Zero();
            Eigen::Matrix3d R_gimbal2odom = Eigen::Matrix3d::Identity();
            auto ctx = std::any_cast<VisionCtx>(frame.any_ctx);
            std::pair<double, double> gimbal_py;
            if (ctx.motion_buffer) {
                const auto delay =
                    std::chrono::microseconds(static_cast<int64_t>(ctx.communication_delay_μs));
                const auto t_query = armors.timestamp + delay;
                auto apply_motion = [&](const auto& att) {
                    v << att.data.vx, att.data.vy, att.data.vz;
                    R_gimbal2odom = Eigen::AngleAxisd(att.data.yaw, Eigen::Vector3d::UnitZ())
                        * Eigen::AngleAxisd(-att.data.pitch, Eigen::Vector3d::UnitY())
                        * Eigen::AngleAxisd(att.data.roll, Eigen::Vector3d::UnitX());
                    gimbal_py = std::make_pair(att.data.pitch, att.data.yaw);
                };
                if (auto past_att = ctx.motion_buffer->get_interpolated(t_query)) {
                    apply_motion(*past_att);
                } else if (auto last_att = ctx.motion_buffer->get_last()) {
                    apply_motion(*last_att);
                }
            }
            autoExposureControl(frame.img_frame.src_img, ctx.camera);
            T_camera_to_odom_ = utils::computeCameraToOdomTransform(
                R_gimbal2odom,
                tf_config_->R_camera2gimbal,
                tf_config_->t_camera2gimbal
            );
            armors.armors = armor_where_->where(sorted_objs, T_camera_to_odom_);
            armors.v = v;
            for (auto& armor: armors.armors) {
                armor.timestamp = armors.timestamp;
            }

            armor_queue_->enqueue(armors);
            ++detect_finish_count_;
            if (debug_mode_) {
                std::lock_guard<std::mutex> lock(dbg_mutex_);
                auto& dbg = auto_aim_debug_;

                dbg.img_frame = frame.img_frame;
                dbg.armors = armors;
                dbg.T_camera_to_odom = T_camera_to_odom_;
                dbg.detect_color = frame.detect_color;
                dbg.armor_objs = sorted_objs;
                dbg.expanded = frame.expanded;
                dbg.gimbal_py = gimbal_py;
            }
        }
        void armorsCallback(const Armors& armors) {
            if (armors.timestamp <= tracker_->getLastTime()) {
                WUST_WARN(logger_) << "Received out-of-order armor data, discarded.";
                return;
            }
            Target target = tracker_->track(armors);
            auto_aim_fsm_cl_.update(std::abs(target.target_state_.vyaw()), target.jumped);
            const auto now = std::chrono::steady_clock::now();
            {
                std::lock_guard<std::mutex> lock(target_mutex_);
                target_ = target;
            }
            const auto latency_ms =
                wust_vl::common::utils::time_utils::durationMs(armors.timestamp, now);
            latency_averager_->add(latency_ms);
            auto& dbg = auto_aim_debug_;
            dbg.latency_ms = latency_averager_->average();
            if (debug_mode_) {
                std::lock_guard<std::mutex> lock(dbg_mutex_);
                dbg.target = target;
                dbg.fsm = auto_aim_fsm_cl_.fsm_state_;
            }
        }
        Target getTarget() {
            Target target;
            {
                std::lock_guard<std::mutex> lock(target_mutex_);
                target = target_;
            }
            return target;
        }
        GimbalCmd solve(double bullet_speed) {
            GimbalCmd gimbal_cmd;
            Target target;
            {
                std::lock_guard<std::mutex> lock(target_mutex_);
                target = target_;
            }
            AimTarget aim_target;
            const bool appear = target.checkTargetAppear();
            if (appear && target.target_state_.pos().norm() > 0.1) {
                try {
                    gimbal_cmd =
                        very_aimer_->veryAim(target, bullet_speed, auto_aim_fsm_cl_.fsm_state_);
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
                // bool skip;
                // if (armor_queue_->dequeue_wait(armors, skip)) {
                //     armorsCallback(armors);
                //     tracker_finish_count_++;
                //     if (skip) {
                //         WUST_DEBUG(logger_) << "OrderQueue skip";
                //     }
                // }
                if (armor_queue_->try_dequeue(armors)) {
                    armorsCallback(armors);
                    tracker_finish_count_++;
                } else {
                    std::this_thread::sleep_for(std::chrono::milliseconds(1));
                }
            }
        }
        void doDebug() {
            debug_mode_ = true;
            AutoAimDebug dbg;
            {
                std::lock_guard<std::mutex> lock(dbg_mutex_);
                dbg = auto_aim_debug_;
            }
            drawDebugOverlayShm(dbg, camera_info_, false);
            debuglog(dbg);
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
        VeryAimer::Ptr very_aimer_;
        ArmorWhere::Ptr armor_where_;
        AutoAimFsmController auto_aim_fsm_cl_;
        AutoExposureCfg::Ptr auto_exposure_cfg_;

        cv::Rect expanded_;
        int max_detect_armors_;
        bool run_flag_ = false;
        int detect_finish_count_ = 0;
        int img_recv_count_ = 0;
        int tracker_finish_count_ = 0;
        int fire_count_ = 0;
        int timer_cout_ = 0;
        Target target_;
        bool debug_mode_ = false;
        AutoAimDebug auto_aim_debug_;
        std::unique_ptr<wust_vl::common::concurrency::Averager<double>> latency_averager_;
        TFConfig::Ptr tf_config_;
        std::pair<cv::Mat, cv::Mat> camera_info_;
        Eigen::Matrix4d T_camera_to_odom_;
        std::mutex target_mutex_;
        std::mutex dbg_mutex_;
    };
    AutoAim::AutoAim(
        const std::string& config_path,
        TFConfig::Ptr tf_config,
        const std::pair<cv::Mat, cv::Mat>& camera_info,
        bool debug
    ):
        _impl(std::make_unique<Impl>(config_path, tf_config, camera_info, debug)) {}
    AutoAim::~AutoAim() {
        _impl.reset();
    }

    void AutoAim::start() {
        _impl->start();
    }
    void AutoAim::pushInput(CommonFrame& frame) {
        _impl->pushInput(frame);
    }

    GimbalCmd AutoAim::solve(double bullet_speed) {
        return _impl->solve(bullet_speed);
    }
    wust_vl::common::concurrency::MonitoredThread::Ptr AutoAim::getThread() {
        return _impl->processing_thread_;
    }
    Target AutoAim::getTarget() {
        return _impl->getTarget();
    }
    void AutoAim::doDebug() {
        _impl->doDebug();
    }
    wust_vl::common::utils::Parameter::Ptr AutoAim::getParameter() {
        return _impl->auto_aim_config_parameter_;
    }
    VeryAimer::Ptr AutoAim::getVeryAimer() {
        return _impl->very_aimer_;
    }
} // namespace auto_aim
} // namespace wust_vision