#include "auto_buff.hpp"
#include "tasks/auto_buff/rune_control/aimer.hpp"
#include "tasks/auto_buff/rune_detector/scut_robot_detector.hpp"
#include "tasks/auto_buff/rune_optimize/ba_solver.hpp"
#include "tasks/auto_buff/rune_tracker/rune_tracker.hpp"
#include "tasks/utils.hpp"
namespace auto_buff {

struct AutoBuff::Impl {
    ~Impl() {
        run_flag_ = false;
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

        std::array<double, 9> camera_matrix;
        CV_Assert(camera_info.first.rows == 3 && camera_info.first.cols == 3);
        CV_Assert(camera_info.first.type() == CV_64F);

        for (int i = 0; i < 3; ++i)
            for (int j = 0; j < 3; ++j)
                camera_matrix[i * 3 + j] = camera_info.first.at<double>(i, j);

 
        scut_detector_ = ScutRobotDetector::make_detector(camera_info_, config);
        rune_queue_ = std::make_unique<OrderedQueue<rune::RuneFan>>(10, 500);
        rune::RuneTargetConfig rune_target_config;
        rune_target_config.loadFromYaml(config["rune_tracker"]);

        rune_tracker_ = std::make_unique<RuneTracker>(
            config["rune_tracker"]["tracking_thres"].as<int>(),
            config["rune_tracker"]["lost_time_thres"].as<double>(),
            config["rune_tracker"]["max_dis_diff"].as<double>(),
            rune_target_config
        );
        std::string comp_type =
            config["trajectory_compensator"]["compenstator_type"].as<std::string>("ideal");
        double gravity_ = config["trajectory_compensator"]["gravity"].as<double>(10.0);
        double resistance_ = config["trajectory_compensator"]["resistance"].as<double>(0.092);
        int iteration_times_ = config["trajectory_compensator"]["iteration_times"].as<int>(20);
        auto trajectory_compensator = CompensatorFactory::createCompensator(comp_type);
        trajectory_compensator->iteration_times_ = iteration_times_;
        trajectory_compensator->gravity_ = gravity_;
        trajectory_compensator->resistance_ = resistance_;
        aimer_ = std::make_unique<rune::Aimer>(config["aimer"], trajectory_compensator);
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
    void pushInput(CommonFrame& frame,bool is_big) {
        img_recv_count_++;
        frame.id = current_id_++;
        Eigen::Vector3d v = Eigen::Vector3d::Zero();
        Eigen::Matrix3d R_gimbal2odom = Eigen::Matrix3d::Identity();
        Eigen::Vector3d gimbal = Eigen::Vector3d::Zero();
        if (shared_->motion_buffer) {
            auto apply_motion = [&](const auto& att) {
                v = Eigen::Vector3d(att.data.vx, att.data.vy, att.data.vz);
                R_gimbal2odom = Eigen::AngleAxisd(att.data.yaw, Eigen::Vector3d::UnitZ())
                    * Eigen::AngleAxisd(-att.data.pitch, Eigen::Vector3d::UnitY())
                    * Eigen::AngleAxisd(att.data.roll, Eigen::Vector3d::UnitX());
                gimbal[0] = att.data.yaw;
                gimbal[1] = att.data.pitch;
                gimbal[2] = att.data.roll;
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
        cv::Mat debug_img;

        {
            std::lock_guard<std::mutex> lock(callback_mutex_);
            auto fan =
                scut_detector_->detect(frame, gimbal, T_camera_to_odom, debug_mode_, debug_img);
            fan.is_big = is_big;
            rune_queue_->enqueue(fan);
        }
        T_camera_to_odom_ = T_camera_to_odom;
        if (debug_mode_) {
            std::lock_guard<std::mutex> lock(dbg_mutex_);
            auto_buff_debug_.src_img = { std::move(debug_img), frame.timestamp };
            auto_buff_debug_.T_camera_to_odom = T_camera_to_odom;
        }

        detect_finish_count_++;
    }

    void runeTargetCallback(const rune::RuneFan& fan) {
        if (fan.timestamp <= last_rune_target_time_) {
            WUST_WARN(logger_) << "Received out-of-order rune data, discarded.";
            return;
        }
        last_rune_target_time_ = fan.timestamp;
        auto rune_target = rune_tracker_->track(fan);
        {
            std::lock_guard<std::mutex> lock(rune_target_mutex_);
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
            static double last_unwrapped_roll = 0.0;
            static double last_raw_roll = 0.0;
            std::lock_guard<std::mutex> lock(dbg_mutex_);
            const double raw_roll = rune_target.roll();
            const double raw_pred = rune_target.predictAngle(0.5);
            const double obs_angle =
                last_unwrapped_roll + angles::shortest_angular_distance(last_raw_roll, raw_roll);
            const double pre_angle =
                obs_angle + angles::shortest_angular_distance(raw_roll, raw_pred);
            last_unwrapped_roll = obs_angle;
            last_raw_roll = raw_roll;
            auto_buff_debug_.obs_v = rune_target.v_roll();
            auto_buff_debug_.fitter_v = rune_target.getFitterSpd(time_utils::now()+std::chrono::microseconds(int(0.2 * 1e6)));
            auto_buff_debug_.obs_angle = obs_angle;
            auto_buff_debug_.pre_angle = pre_angle;
            auto_buff_debug_.target = rune_target;
            auto_buff_debug_.power_rune = rune_target.getPowerRune();
        }
    }
    GimbalCmd solve() {
        GimbalCmd gimbal_cmd;
        rune::RuneTarget rune_target;
        {
            std::lock_guard<std::mutex> lock(rune_target_mutex_);
            rune_target = rune_target_;
        }
        if (gimbal_cmd.fire_advice) {
            fire_count_++;
        }
        if (rune_target.checkTargetAppear()) {
            gimbal_cmd =
                aimer_->aim(rune_target, shared_->bullet_speed, std::chrono::steady_clock::now());
        }
        gimbal_cmd.appera = rune_target.checkTargetAppear();
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
        while (self->isAlive()) {
            if (!self->waitPoint())
                break;
            printStats();
            rune::RuneFan rune;

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
    int current_id_ = 0;
    std::unique_ptr<ScutRobotDetector> scut_detector_;
    std::unique_ptr<RuneTracker> rune_tracker_;
    std::unique_ptr<rune::Aimer> aimer_;
    std::string logger_ = "auto_buff";
    std::unique_ptr<OrderedQueue<rune::RuneFan>> rune_queue_;
    std::shared_ptr<wust_vl_concurrency::MonitoredThread> processing_thread_;
    rune::RuneTarget rune_target_;
    std::mutex rune_target_mutex_;
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
void AutoBuff::pushInput(CommonFrame& frame,bool is_big) {
    _impl->pushInput(frame,is_big);
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