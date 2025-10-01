#include "auto_buff.hpp"
#include "tasks/auto_buff/rune_detect/detect_factory.hpp"
#include "tasks/auto_buff/rune_detect/rune_detector_base.hpp"
#include "tasks/auto_buff/rune_detect/rune_detector_cv.hpp"
#include "tasks/utils.hpp"
namespace auto_buff {

struct AutoBuff::Impl {
    ~Impl() {
        run_flag_ = false;
        if (rune_detector_) {
            rune_detector_.reset();
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
        const std::pair<cv::Mat, cv::Mat>& camera_info
    ) {
        config_ = config;
        R_camera2gimbal_ = R_camera2gimbal;
        t_camera2gimbal_ = t_camera2gimbal;
        camera_info_ = camera_info;

        std::string rune_detect_backend = config_["rune_detect_backend"].as<std::string>("");
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
            return "";
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
        if (rune_detect_backend.empty()) {
            throw std::runtime_error("rune_detect_backend not set in config.");
        }
        rune_detector_ = loadRuneDetectorBackend(rune_detect_backend);
        rune_detector_->setCallback(std::bind(
            &AutoBuff::Impl::RuneDetectCallback,
            this,
            std::placeholders::_1,
            std::placeholders::_2
        ));
        WUST_MAIN(logger_) << "Using Rune Detector: " << rune_detect_backend;
        detect_r_tag_ = rune_detect_config_["rune_detector"]["detect_r_tag"].as<bool>();
        use_manual_r_ = rune_detect_config_["rune_detector"]["use_manual_r"].as<bool>();
        rune_binary_thresh_ = rune_detect_config_["rune_detector"]["min_lightness"].as<int>();
        // rune_solver_ = std::make_unique<RuneSolver>(config_, camera_info_);
        // rune_solver_->setStateCallback(std::bind(&AutoBuff::Impl::runeSloverStateCallback, this));
        rune_queue_ = std::make_unique<OrderedQueue<rune::Rune>>(10, 500);
        latency_averager_ = std::make_unique<Averager<double>>(100);
        return true;
    }
    void start() {
        processing_thread_ = wust_vl_concurrency::MonitoredThread::create(
            "AutoBuffProcessingThread",
            [this](std::shared_ptr<wust_vl_concurrency::MonitoredThread> self) {
                this->processingLoop(self);
            }
        );
        wust_vl_concurrency::ThreadManager::instance().registerThread(processing_thread_);
        run_flag_ = true;
    }
    void pushInput(CommonFrame& frame) {
        img_recv_count_++;
        if (rune_detector_) {
            rune_detector_->pushInput(frame);
        }
    }

    void RuneDetectCallback(const std::vector<rune::RuneObject>& objs, const CommonFrame& frame) {
        auto it = std::max_element(
            objs.begin(),
            objs.end(),
            [](const rune::RuneObject& a, const rune::RuneObject& b) { return a.prob < b.prob; }
        );
        rune::RuneFan fan { .is_valid = false };
        cv::Mat debug_img;
        if (debug_mode_)
            debug_img = frame.src_img.clone();
        if (it != objs.end()) {
            const rune::RuneObject& best_obj = *it;
            cv::Point2f r_tag_roi =
                best_obj.pts.r_tag - cv::Point2f(best_obj.box.x, best_obj.box.y);
            cv::Mat roi = frame.src_img(best_obj.box);
            fan = rune_cv_.detect(roi, r_tag_roi,debug_img,debug_mode_);
            fan.base = cv::Point2f(best_obj.box.x, best_obj.box.y);
        }
        std::lock_guard<std::mutex> lock(callback_mutex_);
        static bool last_rune_big = false;
        rune::Rune rune_target { .frame_id = "camera_optical_frame",
                                 .timestamp = frame.timestamp,
                                 .is_big_rune = false,
                                 .is_lost = true };
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
            auto t_query = rune_target.timestamp + delay;
            if (auto past_att = shared_->motion_buffer->get_interpolated(t_query)) {
                apply_motion(*past_att);
            } else if (auto last_att = shared_->motion_buffer->get_last()) {
                apply_motion(*last_att);
            }
        }
        Eigen::Matrix4d T_camera_to_odom =
            utils::computeCameraToOdomTransform(R_gimbal2odom, R_camera2gimbal_, t_camera2gimbal_);
        
        
        if (fan.is_valid) {
            cv::Mat rvec, tvec;
            bool success = cv::solvePnP(
                fan.getObjs(), // 3D 模型点
                fan.landmarks(), // 对应的2D像素点
                camera_info_.first, // 相机内参
                camera_info_.second, // 畸变系数
                rvec,
                tvec, // 输出：旋转向量和平移向量
                false, // 是否使用初始 rvec,tvec
                cv::SOLVEPNP_ITERATIVE
            );
            if (success) {
                Eigen::Vector3d t;
                Eigen::Quaterniond q;
                utils::pnpToEigen(rvec, tvec, t, q);
                auto pos_odom = utils::transformPosition(t, T_camera_to_odom);
                auto q_odom = utils::transformOrientation(q, T_camera_to_odom);
                auto euler_odom = utils::quatToEuler(q_odom, utils::EulerOrder::ZYX);
                fan.drawPoints(debug_img);
                std::cout << std::fixed << std::setprecision(3) // 固定小数位数
                          << std::setw(12) << "t: " << std::setw(10) << t.x() << std::setw(10)
                          << t.y() << std::setw(10) << t.z()

                          << " | pos_odom: " << std::setw(10) << pos_odom.x() << std::setw(10)
                          << pos_odom.y() << std::setw(10) << pos_odom.z()

                          << " | yaw: " << std::setw(8) << euler_odom[0] * 180.0 / M_PI
                          << " pitch: " << std::setw(8) << euler_odom[1] * 180.0 / M_PI
                          << " roll: " << std::setw(8) << euler_odom[2] * 180.0 / M_PI << std::endl;
            }
        }
    

        rune_target.T_camera_to_odom = T_camera_to_odom;
        rune_target.is_big_rune = false;
        rune_queue_->enqueue(rune_target);
        T_camera_to_odom_ = T_camera_to_odom;
        if (debug_mode_) {
            std::lock_guard<std::mutex> lock(dbg_mutex_);
            auto_buff_debug_.src_img = { debug_img, rune_target.timestamp };
            auto_buff_debug_.objs = objs;
        }

        detect_finish_count_++;
    }
    void runeTargetCallback(const rune::Rune rune_target) {
        if (rune_target.timestamp <= last_rune_target_time_) {
            WUST_WARN(logger_) << "Received out-of-order rune data, discarded.";
            return;
        }
        // last_rune_target_time_ = rune_target.timestamp;
        // if (rune_solver_->pnp_solver_ == nullptr) {
        //     return;
        // }
        double observed_angle = 0;
        // if (rune_solver_->tracker_state == RuneSolver::LOST) {
        //     observed_angle = rune_solver_->init(rune_target, rune_target.T_camera_to_odom);
        // } else {
        //     observed_angle = rune_solver_->update(rune_target, rune_target.T_camera_to_odom);
        // }
        auto now = std::chrono::steady_clock::now();
        auto latency_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
                              std::chrono::steady_clock::now() - rune_target.timestamp
        )
                              .count();
        auto latency_ms = time_utils::durationMs(rune_target.timestamp, now);
        latency_averager_->add(latency_ms);
        auto_buff_debug_.latency_ms = latency_averager_->average();
        if (debug_mode_) {
            std::lock_guard<std::mutex> lock(dbg_mutex_);
        }
    }
    GimbalCmd solve() {
        GimbalCmd gimbal_cmd;

        if (gimbal_cmd.fire_advice) {
            fire_count_++;
        }
        if (debug_mode_) {
            std::lock_guard<std::mutex> lock(dbg_mutex_);
            auto_buff_debug_.gimbal_cmd = gimbal_cmd;
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
            rune::Rune rune;

            if (!rune_queue_->try_dequeue(rune)) {
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
                continue;
            }
            self->heartbeat();
            runeTargetCallback(rune);
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
        using namespace std::chrono;
        auto now = steady_clock::now();

        if (last_stat_time_steady_.time_since_epoch().count() == 0) {
            last_stat_time_steady_ = now;
            return;
        }

        auto elapsed = duration_cast<duration<double>>(now - last_stat_time_steady_);
        if (elapsed.count() >= 1.0) {
            WUST_INFO(logger_) << "Rec: " << img_recv_count_ << ", Det: " << detect_finish_count_
                               << ", Fin: " << tracker_finish_count_ << ", Tc: " << timer_cout_
                               << ", Lat: " << auto_buff_debug_.latency_ms << "ms"
                               << ", Fire: " << fire_count_;
            img_recv_count_ = 0;
            detect_finish_count_ = 0;
            fire_count_ = 0;
            tracker_finish_count_ = 0;
            timer_cout_ = 0;
            last_stat_time_steady_ = now;
        }
    }

    std::mutex callback_mutex_;
    std::unique_ptr<RuneDetectorBase> rune_detector_;
    RuneDetectorCV rune_cv_;
    std::string logger_ = "auto_buff";
    std::unique_ptr<OrderedQueue<rune::Rune>> rune_queue_;
    std::shared_ptr<wust_vl_concurrency::MonitoredThread> processing_thread_;
    bool run_flag_ = false;
    int detect_finish_count_ = 0;
    int img_recv_count_ = 0;
    int tracker_finish_count_ = 0;
    int timer_cout_ = 0;
    int fire_count_;
    std::chrono::steady_clock::time_point last_rune_target_time_;
    std::chrono::steady_clock::time_point last_stat_time_steady_ = std::chrono::steady_clock::now();
    bool debug_mode_ = false;
    DebugRune auto_buff_debug_;
    std::mutex dbg_mutex_;
    std::unique_ptr<Averager<double>> latency_averager_;
    Eigen::Matrix3d R_camera2gimbal_;
    Eigen::Vector3d t_camera2gimbal_;
    std::pair<cv::Mat, cv::Mat> camera_info_;
    YAML::Node config_;
    int rune_binary_thresh_ = 0;
    bool detect_r_tag_;
    static std::vector<cv::Point2f> clicked_points_;
    bool manual_r_init_ = false;
    cv::Point2f manual_r_center_;
    std::vector<cv::Point2f> manual_r_box_;
    bool use_manual_r_ = false;
    std::atomic<bool> manual_r_runing_ { false };
    Eigen::Matrix4d T_r_;
    YAML::Node rune_detect_config_;
    Eigen::Matrix4d T_camera_to_odom_;
    std::shared_ptr<AutoBuffShared> shared_;
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
    const std::pair<cv::Mat, cv::Mat>& camera_info
) {
    return _impl
        ->init(config, use_detect_ncnn_count, R_camera2gimbal, t_camera2gimbal, camera_info);
}
void AutoBuff::start() {
    _impl->start();
}
void AutoBuff::pushInput(CommonFrame& frame) {
    _impl->pushInput(frame);
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
} // namespace auto_buff