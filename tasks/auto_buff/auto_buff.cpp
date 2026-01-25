#include "auto_buff.hpp"
#include "tasks/auto_buff/rune_control/aimer.hpp"
#include "tasks/auto_buff/rune_detector/rune_detector.hpp"
#include "tasks/auto_buff/rune_optimize/ba_solver.hpp"
#include "tasks/auto_buff/rune_tracker/rune_tracker.hpp"
#include "tasks/utils.hpp"
#include "wust_vl/video/camera.hpp"
namespace auto_buff {

struct AutoBuff::Impl {
    ~Impl() {
        run_flag_ = false;
        rune_queue_->stop();
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
        int max_detect_running
    ) {
        config_ = config;
        R_camera2gimbal_ = R_camera2gimbal;
        t_camera2gimbal_ = t_camera2gimbal;
        camera_info_ = camera_info;
        if (config["rune_optimize"]["enable"].as<bool>()) {
            ba_solver_ = auto_buff::BaSolver::create(config["rune_optimize"], camera_info.first);
        }

        rune_detector_ = RuneDetectorCV::make_detector(config["rune_detector"]);
        rune_detector_->setCallback(std::bind(
            &AutoBuff::Impl::runeDetectCallback,
            this,
            std::placeholders::_1,
            std::placeholders::_2,
            std::placeholders::_3
        ));
        rune_queue_ = std::make_unique<OrderedQueue<auto_buff::RuneFan>>(50, 500);
        auto_buff::RuneTargetConfig rune_target_config;
        rune_target_config.loadFromYaml(config["rune_tracker"]);

        rune_tracker_ = RuneTracker::create(config["rune_tracker"]);
        std::string comp_type =
            config["trajectory_compensator"]["compenstator_type"].as<std::string>("ideal");
        double gravity_ = config["trajectory_compensator"]["gravity"].as<double>(10.0);
        double resistance_ = config["trajectory_compensator"]["resistance"].as<double>(0.092);
        int iteration_times_ = config["trajectory_compensator"]["iteration_times"].as<int>(20);
        auto trajectory_compensator = CompensatorFactory::createCompensator(comp_type);
        trajectory_compensator->iteration_times_ = iteration_times_;
        trajectory_compensator->gravity_ = gravity_;
        trajectory_compensator->resistance_ = resistance_;
        aimer_ = auto_buff::Aimer::create(config["aimer"], trajectory_compensator);
        latency_averager_ = std::make_unique<Averager<double>>(100);
        auto_exposure_cfg_.loadFromYaml(config_["auto_exposure"]);
        return true;
    }
    void start() {
        run_flag_ = true;
        processing_thread_ = wust_vl_concurrency::MonitoredThread::create(
            "AutoBuffProcessingThread",
            [this](std::shared_ptr<wust_vl_concurrency::MonitoredThread> self) {
                this->processingLoop(self);
            }
        );
        wust_vl_concurrency::ThreadManager::instance().registerThread(processing_thread_);
    }
    void pushInput(CommonFrame& frame, bool is_big) {
        img_recv_count_++;
        auto bbox = rune_target_.expanded(
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
        rune_detector_->pushInput(frame, is_big, debug_mode_);
    }
    void runeDetectCallback(
        const auto_buff::RuneFan& fan,
        const CommonFrame& frame,
        cv::Mat& debug_img
    ) {
        std::lock_guard<std::mutex> lock(callback_mutex_);
        Eigen::Vector3d v = Eigen::Vector3d::Zero();
        Eigen::Matrix3d R_gimbal2odom = Eigen::Matrix3d::Identity();
        if (shared_->motion_buffer) {
            auto apply_motion = [&](const auto& att) {
                v = Eigen::Vector3d(att.data.vx, att.data.vy, att.data.vz);
                R_gimbal2odom = Eigen::AngleAxisd(att.data.yaw, Eigen::Vector3d::UnitZ())
                    * Eigen::AngleAxisd(-att.data.pitch, Eigen::Vector3d::UnitY())
                    * Eigen::AngleAxisd(att.data.roll, Eigen::Vector3d::UnitX());
            };
            auto delay = std::chrono::microseconds(
                static_cast<int64_t>(std::round(shared_->communication_delay_μs))
            );
            auto t_query = frame.timestamp + delay;
            if (auto past_att = shared_->motion_buffer->get_interpolated(t_query)) {
                apply_motion(*past_att);
            } else if (auto last_att = shared_->motion_buffer->get_last()) {
                apply_motion(*last_att);
            }
        }

        Eigen::Matrix4d T_camera_to_odom =
            utils::computeCameraToOdomTransform(R_gimbal2odom, R_camera2gimbal_, t_camera2gimbal_);
        T_camera_to_odom_ = T_camera_to_odom;
        auto_buff::RuneFan copy_fan = fan;
        const Eigen::Matrix3d R_imu_cam = T_camera_to_odom.block<3, 3>(0, 0);
        double pnp_distance = 0.0;
        for (auto& fan: copy_fan.fans) {
            cv::Mat rvec, tvec;
            cv::solvePnP(
                fan.getObjs(),
                fan.landmarks(),
                camera_info_.first,
                camera_info_.second,
                rvec,
                tvec,
                false,
                cv::SOLVEPNP_IPPE //平移更稳定，（旋转这里纯靠后面优化）

            );
            cv::Mat R_cv;
            cv::Rodrigues(rvec, R_cv);
            Eigen::Matrix3d R = utils::cvToEigen(R_cv);
            Eigen::Vector3d t = utils::cvToEigen(tvec);
            pnp_distance = t.norm();
            if (ba_solver_) {
                R = ba_solver_->solveBa_R(fan, t, R, R_imu_cam);
            }
            fan.ori = Eigen::Quaterniond(R);
            fan.pos = t;
            Eigen::Vector3d pos_camera = fan.pos;
            fan.target_pos = utils::transformPosition(pos_camera, T_camera_to_odom);

            Eigen::Quaterniond q_camera(fan.ori.w(), fan.ori.x(), fan.ori.y(), fan.ori.z());
            Eigen::Quaterniond q_odom = utils::transformOrientation(q_camera, T_camera_to_odom);
            fan.target_ori = q_odom;

            copy_fan.is_valid = true;
        }

        rune_queue_->enqueue(copy_fan);
        if (debug_mode_) {
            std::lock_guard<std::mutex> lock(dbg_mutex_);
            auto_buff_debug_.src_img = { std::move(debug_img), frame.timestamp };
            auto_buff_debug_.T_camera_to_odom = T_camera_to_odom_;
            auto_buff_debug_.expanded = frame.expanded;
            auto_buff_debug_.pnp_distance = pnp_distance;
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
        auto latency_ms = time_utils::durationMs(fan.timestamp, now);
        latency_averager_->add(latency_ms);
        auto_buff_debug_.latency_ms = latency_averager_->average();
        if (debug_mode_) {
            std::lock_guard<std::mutex> lock(dbg_mutex_);
            static double last_unwrapped_roll = 0.0;
            static double last_raw_roll = 0.0;
            const double raw_roll = rune_target.roll();
            const double raw_pred = rune_target.predictAngle(0.5);
            const double obs_angle =
                last_unwrapped_roll + angles::shortest_angular_distance(last_raw_roll, raw_roll);
            const double pre_angle =
                obs_angle + angles::shortest_angular_distance(raw_roll, raw_pred);
            last_unwrapped_roll = obs_angle;
            last_raw_roll = raw_roll;
            auto_buff_debug_.obs_v = rune_target.v_roll();
            auto_buff_debug_.fitter_v = rune_target.getFitterSpd(
                time_utils::now() + std::chrono::microseconds(int(0.2 * 1e6))
            );
            auto_buff_debug_.obs_angle = obs_angle;
            auto_buff_debug_.pre_angle = pre_angle;
            auto_buff_debug_.target = rune_target;
            auto_buff_debug_.power_rune = rune_target.getPowerRune();
        }
    }
    GimbalCmd solve() {
        GimbalCmd gimbal_cmd;
        auto_buff::RuneTarget rune_target;

        {
            std::lock_guard<std::mutex> lock(target_mutex_);
            rune_target = rune_target_;
        }

        if (gimbal_cmd.fire_advice) {
            fire_count_++;
        }
        if (rune_target.checkTargetAppear()) {
            gimbal_cmd = aimer_->aim(rune_target, shared_->bullet_speed);
        }
        if (debug_mode_) {
            std::lock_guard<std::mutex> lock(dbg_mutex_);
            auto_buff_debug_.gimbal_cmd = gimbal_cmd;
            auto_buff_debug_.aim_target = gimbal_cmd.aim_target;
        }
        timer_cout_++;

        return gimbal_cmd;
    }
    void processingLoop(std::shared_ptr<wust_vl_concurrency::MonitoredThread> self) {
        while (!self->isAlive()) {
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
        while (self->isAlive() && run_flag_) {
            if (!self->waitPoint())
                break;
            self->heartbeat();
            printStats();
            auto_buff::RuneFan auto_buff;
            bool skip;
            // if (rune_queue_->dequeue_wait(auto_buff, skip)) {
            //     runeTargetCallback(auto_buff);
            //     tracker_finish_count_++;
            //     if (skip) {
            //         WUST_DEBUG(logger_) << "OrderQueue skip";
            //     }
            // }
            if (!rune_queue_->try_dequeue(auto_buff)) {
                std::this_thread::sleep_for(std::chrono::milliseconds(3));
                continue;
            }
            runeTargetCallback(auto_buff);
            tracker_finish_count_++;
        }
    }
    void setDebug(bool debug) {
        debug_mode_ = debug;
    }
    DebugRune getDebugFrame() {
        std::lock_guard<std::mutex> lock(dbg_mutex_);
        return auto_buff_debug_;
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
    void autoExposureControl(const cv::Mat& frame, std::shared_ptr<wust_vl_video::Camera> camera) {
        const double dt = auto_exposure_cfg_.control_interval_ms / 1000.0;
        utils::XSecOnce(
            [&] {
                if (!auto_exposure_cfg_.enable || frame.empty()) {
                    return;
                }
                if (auto* hik = dynamic_cast<wust_vl_video::HikCamera*>(camera->getDevice())) {
                    cv::Mat i_use = frame(expanded_);
                    if (expanded_.area() < 100 || i_use.empty()) {
                        i_use = frame;
                    }
                    double brightness = utils::computeBrightness(i_use);

                    double diff = brightness - auto_exposure_cfg_.target_brightness;
                    const double exposure_min = auto_exposure_cfg_.exposure_min;
                    const double exposure_max = auto_exposure_cfg_.exposure_max;
                    double exposure_time = hik->getExposureTime();
                    double last_exposure_time = exposure_time;
                    if (std::fabs(diff) > auto_exposure_cfg_.tolerance && exposure_time > 0.0) {
                        exposure_time -= diff * auto_exposure_cfg_.step_gain;
                    } else {
                        exposure_time -= auto_exposure_cfg_.decay_step;
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
    auto_buff::BaSolver::Ptr ba_solver_;
    std::string logger_ = "auto_buff";
    std::unique_ptr<OrderedQueue<auto_buff::RuneFan>> rune_queue_;
    std::shared_ptr<wust_vl_concurrency::MonitoredThread> processing_thread_;
    AutoExposureCfg auto_exposure_cfg_;
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
    DebugRune auto_buff_debug_;
    std::unique_ptr<Averager<double>> latency_averager_;
    Eigen::Matrix3d R_camera2gimbal_;
    Eigen::Vector3d t_camera2gimbal_;
    std::pair<cv::Mat, cv::Mat> camera_info_;
    YAML::Node config_;
    Eigen::Matrix4d T_camera_to_odom_;
    std::shared_ptr<AutoBuffShared> shared_;
    std::mutex target_mutex_;
    std::mutex dbg_mutex_;
    void setShared(std::shared_ptr<AutoBuffShared> shared) {
        shared_ = shared;
    }
};
AutoBuff::AutoBuff(): _impl(std::make_unique<Impl>()) {}
AutoBuff::~AutoBuff() {
    _impl.reset();
}
bool AutoBuff::init(
    const YAML::Node& config,
    int& use_detect_ncnn_count,
    const Eigen::Matrix3d& R_camera2gimbal,
    const Eigen::Vector3d& t_camera2gimbal,
    const std::pair<cv::Mat, cv::Mat>& camera_info,
    int max_detect_running
) {
    return _impl->init(
        config,
        use_detect_ncnn_count,
        R_camera2gimbal,
        t_camera2gimbal,
        camera_info,
        max_detect_running
    );
}
void AutoBuff::start() {
    _impl->start();
}
void AutoBuff::pushInput(CommonFrame& frame, bool is_big) {
    _impl->pushInput(frame, is_big);
}
void AutoBuff::setDebug(bool debug) {
    _impl->setDebug(debug);
}
DebugRune AutoBuff::getDebugFrame() {
    return _impl->getDebugFrame();
}
GimbalCmd AutoBuff::solve() {
    return _impl->solve();
}
void AutoBuff::setShared(std::shared_ptr<AutoBuffShared> shared) {
    _impl->setShared(shared);
}
bool AutoBuff::isActive() {
    if (_impl->processing_thread_->getStatus()
        == wust_vl_concurrency::MonitoredThread::Status::Running) {
        return true;
    } else {
        return false;
    }
}
void AutoBuff::processingWait() {
    _impl->processing_thread_->pause();
}
void AutoBuff::processingUp() {
    _impl->processing_thread_->resume();
}
void AutoBuff::autoExposureControl(
    const cv::Mat& frame,
    std::shared_ptr<wust_vl_video::Camera> camera
) {
    _impl->autoExposureControl(frame, camera);
}
} // namespace auto_buff