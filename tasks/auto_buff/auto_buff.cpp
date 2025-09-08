#include "auto_buff.hpp"
#include "rune_solver.hpp"
#include "tasks/auto_buff/rune_detect/detect_factory.hpp"
#include "tasks/auto_buff/rune_detect/rune_detector_base.hpp"
#include "tasks/utils.hpp"
namespace auto_buff {
std::vector<cv::Point2f> clicked_points_;
void onMouse(int event, int x, int y, int, void*) {
    if (event == cv::EVENT_LBUTTONDOWN) {
        clicked_points_.clear();
        clicked_points_.emplace_back(x, y);
        WUST_INFO("Manual R") << "Clicked Point: (" << x << ", " << y << ")";
    }
}
static std::vector<cv::Point3f> R_BOX_POINTS = {
    { 0, 0.05, -0.05 },
    { 0, -0.05, -0.05 },
    { 0, -0.05, 0.05 },
    { 0, 0.05, 0.05 },
};
bool projectRTargetToImage(
    const Eigen::Matrix4d& TRodom,
    const Eigen::Matrix4d& T_camera_to_odom,
    std::vector<cv::Point2f>& manual_r_box,
    const cv::Mat& camera_intrinsic,
    const cv::Mat& camera_distortion
) {
    if (camera_intrinsic.empty() || camera_distortion.empty()) {
        //WUST_ERROR(mono_logger) << "Camera parameters not initialized.";
        return false;
    }

    // 计算TRc: 目标相对于相机
    Eigen::Matrix4d TRc = T_camera_to_odom.inverse() * TRodom;

    // 从TRc提取旋转和平移
    Eigen::Matrix3d R_eigen = TRc.block<3, 3>(0, 0);
    Eigen::Vector3d t_eigen = TRc.block<3, 1>(0, 3);

    cv::Mat rvec, tvec;
    cv::Mat R_cv;
    cv::eigen2cv(R_eigen, R_cv);
    cv::Rodrigues(R_cv, rvec); // 将旋转矩阵转为旋转向量
    cv::eigen2cv(t_eigen, tvec);

    // R_BOX_POINTS：目标3D点，cv::Point3f格式的vector
    cv::projectPoints(R_BOX_POINTS, rvec, tvec, camera_intrinsic, camera_distortion, manual_r_box);

    return true;
}
bool calcRTarget(
    const std::vector<cv::Point2f>& manual_r_box,
    Eigen::Matrix4d& TRodom,
    const Eigen::Matrix4d& T_camera_to_odom,
    const cv::Mat& camera_intrinsic,
    const cv::Mat& camera_distortion
) {
    if (camera_intrinsic.empty() || camera_distortion.empty()) {
        //WUST_ERROR(mono_logger) << "Camera parameters not initialized.";
        return false;
    }

    // OpenCV solvePnP
    cv::Mat rvec, tvec;
    bool res = cv::solvePnP(
        R_BOX_POINTS,
        manual_r_box,
        camera_intrinsic,
        camera_distortion,
        rvec,
        tvec,
        false,
        cv::SOLVEPNP_IPPE
    );

    if (!res || !cv::checkRange(rvec) || !cv::checkRange(tvec)) {
        return false;
    }

    // Rodrigues -> rotation matrix
    cv::Mat R_cv;
    cv::Rodrigues(rvec, R_cv);

    // 转为 Eigen
    Eigen::Matrix3d R_eigen;
    Eigen::Vector3d t_eigen;
    cv::cv2eigen(R_cv, R_eigen);
    cv::cv2eigen(tvec, t_eigen);

    // 构造 Target 相对于 Camera 的齐次变换矩阵
    Eigen::Matrix4d TRc = Eigen::Matrix4d::Identity();
    TRc.block<3, 3>(0, 0) = R_eigen;
    TRc.block<3, 1>(0, 3) = t_eigen;

    // 计算 Target 相对于 Odom 的变换
    TRodom = T_camera_to_odom * TRc;

    return true;
}
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
        rune_solver_ = std::make_unique<RuneSolver>(config_, camera_info_);
        rune_solver_->setStateCallback(std::bind(&AutoBuff::Impl::runeSloverStateCallback, this));
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
        if (use_manual_r_) {
            cv::Point2f center(frame.src_img.cols / 2.0, frame.src_img.rows / 2.0);
            calculationManualR(center);
        }
        if (rune_detector_) {
            rune_detector_->pushInput(frame);
        }
    }
    void RuneDetectCallback(std::vector<rune::RuneObject>& objs, const CommonFrame& frame) {
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
                // R_gimbal2odom = att.q.toRotationMatrix();
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
        cv::Mat debug_img;
        if (debug_mode_)
            debug_img = frame.src_img.clone();

        if (!objs.empty()) {
            std::sort(objs.begin(), objs.end(), [](auto& a, auto& b) { return a.prob > b.prob; });

            cv::Point2f r_tag;
            cv::Mat binary_roi(1, 1, CV_8UC3, cv::Scalar(0));

            auto detectRTagAt = [&](const cv::Point2f& center, bool manual) {
                return rune_detector_
                    ->detectRTag(frame.src_img, rune_binary_thresh_, center, manual);
            };

            if (use_manual_r_ && manual_r_init_) {
                projectRTargetToImage(
                    T_r_,
                    T_camera_to_odom,
                    manual_r_box_,
                    camera_info_.first,
                    camera_info_.second
                );

                manual_r_center_ = utils::computeCenter(manual_r_box_);
                r_tag = manual_r_center_;
                if (detect_r_tag_ && !frame.src_img.empty())
                    std::tie(r_tag, binary_roi) = detectRTagAt(manual_r_center_, true);
            } else if (detect_r_tag_ && !frame.src_img.empty()) {
                std::tie(r_tag, binary_roi) = detectRTagAt(objs.front().pts.r_center, false);
            } else {
                r_tag = std::accumulate(
                    objs.begin(),
                    objs.end(),
                    cv::Point2f(0, 0),
                    [n = float(objs.size())](cv::Point2f p, auto& o) {
                        return p + o.pts.r_center / n;
                    }
                );
            }

            for (auto& o: objs)
                o.pts.r_center = r_tag;

            if (debug_mode_ && !debug_img.empty()) {
                cv::Rect roi(debug_img.cols - binary_roi.cols, 0, binary_roi.cols, binary_roi.rows);
                binary_roi.copyTo(debug_img(roi));
                cv::rectangle(debug_img, roi, cv::Scalar(150, 150, 150), 2);
            }

            auto target_it = std::find_if(
                objs.begin(),
                objs.end(),
                [c = EnemyColor(frame.detect_color)](auto& o) {
                    return o.type == rune::RuneType::INACTIVATED && o.color == c;
                }
            );

            if (target_it != objs.end()) {
                rune_target.is_lost = false;
                auto& p = target_it->pts;
                rune_target.pts[0].x = p.r_center.x;
                rune_target.pts[0].y = p.r_center.y;
                rune_target.pts[1].x = p.bottom_left.x;
                rune_target.pts[1].y = p.bottom_left.y;
                rune_target.pts[2].x = p.top_left.x;
                rune_target.pts[2].y = p.top_left.y;
                rune_target.pts[3].x = p.top_right.x;
                rune_target.pts[3].y = p.top_right.y;
                rune_target.pts[4].x = p.bottom_right.x;
                rune_target.pts[4].y = p.bottom_right.y;
            }
        }
        if (shared_) {
            if (shared_->is_rune_big) {
                rune_target.is_big_rune = true;
            } else {
                rune_target.is_big_rune = false;
            }
        } else {
            rune_target.is_big_rune = false;
        }

        rune_target.id = frame.id;

        rune_target.T_camera_to_odom = T_camera_to_odom;
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
        last_rune_target_time_ = rune_target.timestamp;
        if (rune_solver_->pnp_solver_ == nullptr) {
            return;
        }
        double observed_angle = 0;
        if (rune_solver_->tracker_state == RuneSolver::LOST) {
            observed_angle = rune_solver_->init(rune_target, rune_target.T_camera_to_odom);
        } else {
            observed_angle = rune_solver_->update(rune_target, rune_target.T_camera_to_odom);
        }
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
            auto_buff_debug_.debug_text = rune_solver_->curve_fitter_->getDebugText();
            auto_buff_debug_.manual_r_box = manual_r_box_;
            auto_buff_debug_.predict_angle =
                rune_solver_->last_pre_angle - rune_solver_->last_observed_angle_;
            auto_buff_debug_.obs_angle = rune_solver_->last_observed_angle_;
            auto_buff_debug_.pre_angle = rune_solver_->last_pre_angle;
            auto_buff_debug_.fitter_v = rune_solver_->curve_fitter_->getFittingParam()[0];
        }
    }
    RuneSolver::StateResult runeSloverStateCallback() {
        auto result = RuneSolver::StateResult();
        if (shared_->motion_buffer) {
            auto last_att = shared_->motion_buffer->get_last();
            if (last_att) {
                // result.rpy[0] = last_att->roll;
                // result.rpy[1] = last_att->pitch;
                // result.rpy[2] = last_att->yaw;
            }
        }
        result.bullet_speed = shared_->bullet_speed;
        result.controller_delay = shared_->controller_delay;

        return result;
    }
    GimbalCmd solve() {
        GimbalCmd gimbal_cmd;
        try {
            gimbal_cmd = rune_solver_->solve();
        } catch (const std::exception& e) {
            gimbal_cmd = rune_solver_->returnDefaultCmd();
        }

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
    std::unique_ptr<RuneSolver> rune_solver_;
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
    void calculationManualR(const cv::Point2f center) {
        manual_r_runing_ = true;
        const int half_size = 5;
        float x = center.x;
        float y = center.y;
        manual_r_center_ = { x, y };
        manual_r_box_ = { {
            { x - half_size, y - half_size }, // 左上 → 对应点0
            { x - half_size, y + half_size }, // 左下 → 对应点1
            { x + half_size, y + half_size }, // 右下 → 对应点2
            { x + half_size, y - half_size } // 右上 → 对应点3
        } };
        calcRTarget(
            manual_r_box_,
            T_r_,
            T_camera_to_odom_,
            camera_info_.first,
            camera_info_.second
        );

        manual_r_runing_ = false;
    }
    void calculationManualR(const cv::Mat& src_img) {
        manual_r_runing_ = true;
        clicked_points_.clear();
        cv::Mat img_show = src_img.clone();
        cv::cvtColor(img_show, img_show, cv::COLOR_BGR2RGB);
        cv::namedWindow("Manual R Box", cv::WINDOW_NORMAL);
        cv::resizeWindow("Manual R Box", 1280, 960);
        cv::setMouseCallback("Manual R Box", onMouse, nullptr);
        const int half_size = 5;
        while (true) {
            cv::Mat temp = img_show.clone();
            if (!clicked_points_.empty()) {
                manual_r_center_ = clicked_points_.front();
                float x = std::clamp(
                    manual_r_center_.x,
                    float(half_size),
                    float(src_img.cols - half_size - 1)
                );
                float y = std::clamp(
                    manual_r_center_.y,
                    float(half_size),
                    float(src_img.rows - half_size - 1)
                );
                manual_r_center_ = { x, y };
                manual_r_box_ = { {
                    { x - half_size, y - half_size }, // 左上 → 对应点0
                    { x - half_size, y + half_size }, // 左下 → 对应点1
                    { x + half_size, y + half_size }, // 右下 → 对应点2
                    { x + half_size, y - half_size } // 右上 → 对应点3
                } };
                cv::circle(temp, manual_r_center_, 3, cv::Scalar(0, 255, 0), -1);
                for (int i = 0; i < 4; ++i)
                    cv::line(
                        temp,
                        manual_r_box_[i],
                        manual_r_box_[(i + 1) % 4],
                        cv::Scalar(255, 0, 0),
                        1
                    );
            }
            cv::imshow("Manual R Box", temp);
            int key = cv::waitKey(30);
            if (key == 27) { // ESC 退出，不提交
                WUST_INFO("Manual R") << "Manual box canceled.";
                manual_r_init_ = false;
                break;
            }
            if (key == 13 || key == 10) { // 回车提交
                if (!clicked_points_.empty()) {
                    manual_r_init_ = true;
                    WUST_INFO("Manual R") << "Manual center: (" << manual_r_center_.x << ", "
                                          << manual_r_center_.y << ")";
                    WUST_INFO("Manual R") << "Manual R Box Points Saved.";
                } else {
                    WUST_ERROR("Manual R") << "No point to submit.";
                    manual_r_init_ = false;
                }
                break;
            }
            if (key == 'b' || key == 8) {
                clicked_points_.clear();
                WUST_INFO("Manual R") << "Manual point cleared.";
            }
        }

        calcRTarget(
            manual_r_box_,
            T_r_,
            T_camera_to_odom_,
            camera_info_.first,
            camera_info_.second
        );

        cv::destroyWindow("Manual R Box");
        cv::destroyWindow("Manual R Box");
        cv::destroyWindow("Manual R Box");
        manual_r_runing_ = false;
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