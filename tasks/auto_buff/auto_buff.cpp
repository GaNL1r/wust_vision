#include "auto_buff.hpp"
#include "tasks/auto_buff/debug.hpp"
#include "tasks/auto_buff/rune_control/aimer.hpp"
#include "tasks/auto_buff/rune_detector/rune_detector.hpp"
#include "tasks/auto_buff/rune_tracker/rune_tracker.hpp"
#include "tasks/auto_buff/rune_where/rune_where.hpp"
#include "tasks/type_common.hpp"
#include "tasks/utils/utils.hpp"
namespace wust_vision {
namespace auto_buff {

    struct AutoBuff::Impl {
        ~Impl() {
            run_flag_ = false;
            rune_queue_->stop();
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
            debug_mode_ = debug;
            auto_buff_config_parameter_ = wust_vl::common::utils::Parameter::create();
            auto_buff_config_parameter_->loadFromFile(config_path);
            auto_exposure_cfg_ = AutoExposureCfg::create();
            aimer_ = auto_buff::Aimer::create(auto_buff_config_parameter_);
            rune_tracker_ = RuneTracker::create(auto_buff_config_parameter_);
            auto_buff_config_parameter_->registerGroup(*auto_exposure_cfg_);
            auto_buff_config_parameter_->reloadFromOldPath();
            auto config = YAML::LoadFile(config_path);
            wust_vl::common::utils::ParameterManager::instance().registerParameter(
                *auto_buff_config_parameter_.get()
            );
            tf_config_ = tf_config;
            camera_info_ = camera_info;

            rune_where_ = auto_buff::RuneWhere::create(config["rune_where"], camera_info);

            rune_detector_ = RuneDetectorCV::make_detector(config["rune_detector"]);
            rune_detector_->setCallback(std::bind(
                &AutoBuff::Impl::runeDetectCallback,
                this,
                std::placeholders::_1,
                std::placeholders::_2,
                std::placeholders::_3
            ));
            rune_queue_ =
                std::make_unique<wust_vl::common::concurrency::OrderedQueue<auto_buff::RuneFan>>(
                    50,
                    500
                );

            latency_averager_ =
                std::make_unique<wust_vl::common::concurrency::Averager<double>>(100);
        }
        void start() {
            if (run_flag_) {
                return;
            }
            run_flag_ = true;
            processing_thread_ = wust_vl::common::concurrency::MonitoredThread::create(
                "AutoBuffProcessingThread",
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
            auto bbox = rune_target_.expanded(
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
            rune_detector_->pushInput(frame, debug_mode_);
        }
        void runeDetectCallback(
            const auto_buff::RuneFan& fan,
            const CommonFrame& frame,
            cv::Mat& debug_img
        ) {
            std::lock_guard<std::mutex> lock(callback_mutex_);
            Eigen::Vector3d v = Eigen::Vector3d::Zero();
            Eigen::Matrix3d R_gimbal2odom = Eigen::Matrix3d::Identity();
            auto ctx = std::any_cast<VisionCtx>(frame.any_ctx);
            std::pair<double, double> gimbal_py;
            if (ctx.motion_buffer) {
                const auto delay =
                    std::chrono::microseconds(static_cast<int64_t>(ctx.communication_delay_μs));
                const auto t_query = fan.timestamp + delay;
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
            Eigen::Matrix4d T_camera_to_odom = utils::computeCameraToOdomTransform(
                R_gimbal2odom,
                tf_config_->R_camera2gimbal,
                tf_config_->t_camera2gimbal
            );
            T_camera_to_odom_ = T_camera_to_odom;
            auto_buff::RuneFan copy_fan = rune_where_->where(fan, T_camera_to_odom);
            copy_fan.is_big =
                InfantryMode::toAttackMode(ctx.mode) == InfantryMode::AttackMode::BIG_RUNE;
            rune_queue_->enqueue(copy_fan);
            if (debug_mode_) {
                std::lock_guard<std::mutex> lock(dbg_mutex_);
                auto_buff_debug_.img_frame = frame.img_frame;
                auto_buff_debug_.T_camera_to_odom = T_camera_to_odom_;
                auto_buff_debug_.expanded = frame.expanded;
                auto_buff_debug_.pnp_distance =
                    copy_fan.fans.empty() ? 0.0 : copy_fan.fans[0].pos.norm();
                auto_buff_debug_.gimbal_py = gimbal_py;
            }

            detect_finish_count_++;
        }

        void runeTargetCallback(const auto_buff::RuneFan& fan) {
            if (fan.timestamp <= last_rune_target_time_) {
                WUST_WARN(logger_) << "Received out-of-order auto_buff data, discarded.";
                return;
            }
            last_rune_target_time_ = fan.timestamp;

            auto rune_target = rune_tracker_->track(fan);

            {
                std::lock_guard<std::mutex> lock(target_mutex_);
                rune_target_ = rune_target;
            }
            auto now = std::chrono::steady_clock::now();
            auto latency_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
                                  std::chrono::steady_clock::now() - fan.timestamp
            )
                                  .count();
            auto latency_ms = wust_vl::common::utils::time_utils::durationMs(fan.timestamp, now);
            latency_averager_->add(latency_ms);
            auto_buff_debug_.latency_ms = latency_averager_->average();
            if (debug_mode_) {
                std::lock_guard<std::mutex> lock(dbg_mutex_);
                static double last_unwrapped_roll = 0.0;
                static double last_raw_roll = 0.0;
                const double raw_roll = rune_target.roll();
                const double raw_pred = rune_target.predictAngle(0.5);
                const double obs_angle = last_unwrapped_roll
                    + angles::shortest_angular_distance(last_raw_roll, raw_roll);
                const double pre_angle =
                    obs_angle + angles::shortest_angular_distance(raw_roll, raw_pred);
                last_unwrapped_roll = obs_angle;
                last_raw_roll = raw_roll;
                auto_buff_debug_.obs_v = rune_target.v_roll();
                auto_buff_debug_.fitter_v = rune_target.getFitterSpd(
                    wust_vl::common::utils::time_utils::now()
                    + std::chrono::microseconds(int(0.2 * 1e6))
                );
                auto_buff_debug_.obs_angle = obs_angle;
                auto_buff_debug_.pre_angle = pre_angle;
                auto_buff_debug_.target = rune_target;
                auto_buff_debug_.power_rune = rune_target.getPowerRune();
            }
        }
        GimbalCmd solve(double bullet_speed) {
            GimbalCmd gimbal_cmd;
            auto_buff::RuneTarget rune_target;

            {
                std::lock_guard<std::mutex> lock(target_mutex_);
                rune_target = rune_target_;
            }
            if (rune_target.checkTargetAppear()) {
                gimbal_cmd = aimer_->aim(rune_target, bullet_speed);
            }
            if (gimbal_cmd.fire_advice) {
                fire_count_++;
            }
            if (debug_mode_) {
                std::lock_guard<std::mutex> lock(dbg_mutex_);
                auto_buff_debug_.gimbal_cmd = gimbal_cmd;
                auto_buff_debug_.aim_target = gimbal_cmd.aim_target;
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
                auto_buff::RuneFan auto_buff;
                // bool skip;
                // if (rune_queue_->dequeue_wait(auto_buff, skip)) {
                //     runeTargetCallback(auto_buff);
                //     tracker_finish_count_++;
                //     if (skip) {
                //         WUST_DEBUG(logger_) << "OrderQueue skip";
                //     }
                // }
                if (rune_queue_->try_dequeue(auto_buff)) {
                    runeTargetCallback(auto_buff);
                    tracker_finish_count_++;
                } else {
                    std::this_thread::sleep_for(std::chrono::milliseconds(1));
                }
            }
        }
        void doDebug() {
            debug_mode_ = true;
            AutoBuffDebug dbg;
            {
                std::lock_guard<std::mutex> lock(dbg_mutex_);
                dbg = auto_buff_debug_;
            }
            drawDebugOverlayShm(dbg, camera_info_, false);
            debuglog(dbg);
        }
        void printStats() {
            utils::XSecOnce(
                [&] {
                    WUST_INFO(logger_)
                        << "Rec: " << img_recv_count_ << ", Det: " << detect_finish_count_
                        << ", Fin: " << tracker_finish_count_ << ", Tc: " << timer_cout_
                        << ", Lat: " << auto_buff_debug_.latency_ms << "ms"
                        << ", Fire: " << fire_count_;
                    img_recv_count_ = 0;
                    detect_finish_count_ = 0;
                    fire_count_ = 0;
                    tracker_finish_count_ = 0;
                    timer_cout_ = 0;
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
        std::mutex callback_mutex_;
        RuneDetectorCV::Ptr rune_detector_;
        RuneTracker::Ptr rune_tracker_;
        auto_buff::Aimer::Ptr aimer_;
        RuneWhere::Ptr rune_where_;
        std::string logger_ = "auto_buff";
        std::unique_ptr<wust_vl::common::concurrency::OrderedQueue<auto_buff::RuneFan>> rune_queue_;
        wust_vl::common::concurrency::MonitoredThread::Ptr processing_thread_;
        AutoExposureCfg::Ptr auto_exposure_cfg_;
        cv::Rect expanded_;
        auto_buff::RuneTarget rune_target_;
        bool run_flag_ = false;
        int detect_finish_count_ = 0;
        int img_recv_count_ = 0;
        int tracker_finish_count_ = 0;
        int timer_cout_ = 0;
        int fire_count_;
        std::chrono::steady_clock::time_point last_rune_target_time_;
        bool debug_mode_ = false;
        AutoBuffDebug auto_buff_debug_;
        std::unique_ptr<wust_vl::common::concurrency::Averager<double>> latency_averager_;
        TFConfig::Ptr tf_config_;
        std::pair<cv::Mat, cv::Mat> camera_info_;
        wust_vl::common::utils::Parameter::Ptr auto_buff_config_parameter_;
        Eigen::Matrix4d T_camera_to_odom_;

        std::mutex target_mutex_;
        std::mutex dbg_mutex_;
    };
    AutoBuff::AutoBuff(
        const std::string& config_path,
        TFConfig::Ptr tf_config,
        const std::pair<cv::Mat, cv::Mat>& camera_info,
        bool debug
    ):
        _impl(std::make_unique<Impl>(config_path, tf_config, camera_info, debug)) {}
    AutoBuff::~AutoBuff() {
        _impl.reset();
    }
    void AutoBuff::start() {
        _impl->start();
    }
    void AutoBuff::pushInput(CommonFrame& frame) {
        _impl->pushInput(frame);
    }

    GimbalCmd AutoBuff::solve(double bullet_speed) {
        return _impl->solve(bullet_speed);
    }

    wust_vl::common::concurrency::MonitoredThread::Ptr AutoBuff::getThread() {
        return _impl->processing_thread_;
    }
    void AutoBuff::doDebug() {
        _impl->doDebug();
    }
} // namespace auto_buff
} // namespace wust_vision