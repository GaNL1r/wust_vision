// Copyright 2025 Xiaojian Wu
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
#include "wust_vision.hpp"
#include "common/calculation.hpp"
#include "common/debug/matplottools.hpp"
#include "common/debug/tools.hpp"
#include "common/debug/toolsgobal.hpp"
#include "common/gobal.hpp"
#include "common/logger.hpp"
#include "common/tf.hpp"
#include "control/armor_solver.hpp"
#include "type/type.hpp"
#include <csignal>
#include <functional>
#include <iostream>
#include <mutex>
#include <opencv2/core/mat.hpp>
#include <opencv2/highgui.hpp>
#include <ostream>
#include <string>
#include <unistd.h>
WustVision::WustVision() {}
WustVision::~WustVision() {}
void WustVision::stop() {
    gobal::is_inited_ = false;

    if (!only_nav_enable) {
        auto_labeler_.reset();
        if (use_video) {
            video_player_->stop();
        } else {
            if (camera_) {
                camera_->stopCamera();
                camera_.reset();
            }
        }

        stopTimer();
        std::this_thread::sleep_for(std::chrono::seconds(1));

        armor_detector_.reset();
        rune_detector_.reset();
#ifdef USE_NCNN
        if (use_rune_detect_ncnn || use_armor_detect_ncnn) {
            ncnn::destroy_gpu_instance();
        }

#endif
        gobal::measure_tool_.reset();

        if (thread_pool_) {
            thread_pool_->waitUntilEmpty();
            thread_pool_.reset();
        }
        if (toolsgobal::robot_cmd_plot_thread_.joinable()) {
            toolsgobal::robot_cmd_plot_thread_.join();
        }
    }
    if (serial_) {
        serial_->stopThread();
        serial_.reset();
    }

    WUST_MAIN(vision_logger) << "WustVision shutdown complete.";
}
void WustVision::stopTimer() {
    {
        std::lock_guard<std::mutex> lk(timer_mtx_);
        timer_running_ = false;
    }
    timer_cv_.notify_one();
    if (timer_thread_.joinable()) {
        timer_thread_.join();
    }
}
void WustVision::init() {
    WUST_MAIN(vision_logger) << "WustVision init start";
#ifdef USE_OPENVINO
    gobal::config = YAML::LoadFile("/home/hy/wust_vision/config/config_openvino.yaml");
#elif defined(USE_TRT)
    gobal::config = YAML::LoadFile("/home/hy/wust_vision/config/config_trt.yaml");
#elif defined(USE_NCNN_ONLY)
    gobal::config = YAML::LoadFile("/home/hy/wust_vision/config/config_ncnn.yaml");
#else
    static_assert(false, "No backend defined: USE_OPENVINO or USE_TRT");
#endif

    std::string log_level_ = gobal::config["logger"]["log_level"].as<std::string>("INFO");
    std::string log_path_ = gobal::config["logger"]["log_path"].as<std::string>("wust_log");
    bool use_logcli = gobal::config["logger"]["use_logcli"].as<bool>();
    bool use_logfile = gobal::config["logger"]["use_logfile"].as<bool>();
    bool use_simplelog = gobal::config["logger"]["use_simplelog"].as<bool>();
    initLogger(log_level_, log_path_, use_logcli, use_logfile, use_simplelog);
    gobal::control_rate = gobal::config["control"]["control_rate"].as<int>();
    initSerial();
    only_nav_enable = gobal::config["common"]["only_nav_enable"].as<bool>();
    if (!only_nav_enable) {
        gobal::attack_mode = gobal::config["common"]["init_attack_mode"].as<int>();
        gobal::debug_mode_ = gobal::config["debug"]["debug_mode"].as<bool>();
        toolsgobal::debug_w = gobal::config["debug"]["debug_w"].as<int>(640);
        toolsgobal::debug_h = gobal::config["debug"]["debug_h"].as<int>(480);
        debug_show_dt_ = gobal::config["debug"]["debug_show_dt"].as<double>(0.05);
        toolsgobal::debug_fps = gobal::config["debug"]["debug_fps"].as<double>(30);
        gobal::use_calculation_ = gobal::config["common"]["use_calculation"].as<bool>();

        use_video = gobal::config["camera"]["video_player"]["use"].as<bool>(false);
        if (use_video) {
            std::string video_play_path =
                gobal::config["camera"]["video_player"]["path"].as<std::string>("");
            int video_play_fps = gobal::config["camera"]["video_player"]["fps"].as<int>(30);
            int start_frame = gobal::config["camera"]["video_player"]["start_frame"].as<int>(0);
            bool loop = gobal::config["camera"]["video_player"]["loop"].as<bool>(false);
            video_player_ =
                std::make_unique<VideoPlayer>(video_play_path, video_play_fps, start_frame, loop);
            video_player_->setCallback([this](const ImageFrame& frame) {
                static bool first_is_inited = false;

                if (gobal::is_inited_) {
                    Eigen::Matrix3d R_gimbal2odom;
                    R_gimbal2odom = Eigen::AngleAxisd(gobal::last_yaw, Eigen::Vector3d::UnitZ())
                        * Eigen::AngleAxisd(gobal::last_pitch, Eigen::Vector3d::UnitY())
                        * Eigen::AngleAxisd(gobal::last_roll, Eigen::Vector3d::UnitX());
                    thread_pool_->enqueue([frame = std::move(frame), R_gimbal2odom, this]() {
                        processImage(frame, R_gimbal2odom);
                    });
                } else {
                    return;
                }
            });

        } else {
            camera_ = std::make_unique<HikCamera>();
            if (!camera_->initializeCamera()) {
                WUST_ERROR(vision_logger) << "Camera initialization failed.";
                return;
            }

            camera_->setParameters(
                gobal::config["camera"]["acquisition_frame_rate"].as<int>(),
                gobal::config["camera"]["exposure_time"].as<int>(),
                gobal::config["camera"]["gain"].as<double>(),
                gobal::config["camera"]["adc_bit_depth"].as<std::string>(),
                gobal::config["camera"]["pixel_format"].as<std::string>(),
                gobal::config["camera"]["acquisitionFrameRateEnable"].as<bool>()
            );
            camera_->setFrameCallback(
                [this](const ImageFrame& frame, Eigen::Matrix3d R_gimbal2odom) {
                    static bool first_is_inited = false;

                    if (gobal::is_inited_) {
                        thread_pool_->enqueue([frame = std::move(frame), R_gimbal2odom, this]() {
                            processImage(frame, R_gimbal2odom);
                        });
                    } else {
                        return;
                    }
                }
            );
        }
        use_auto_labeler = gobal::config["common"]["use_auto_labeler"].as<bool>(false);
        if (use_auto_labeler) {
            auto_labeler_ = std::make_unique<Labeler>();
        }

        const std::string camera_info_path =
            gobal::config["camera"]["camera_info_path"].as<std::string>();
        YAML::Node config_camera_info = YAML::LoadFile(camera_info_path);
        std::vector<double> camera_k =
            config_camera_info["camera_matrix"]["data"].as<std::vector<double>>();
        std::vector<double> camera_d =
            config_camera_info["distortion_coefficients"]["data"].as<std::vector<double>>();

        // 检查大小正确性
        assert(camera_k.size() == 9);
        assert(camera_d.size() == 5);

        // 将 vector 转为 Mat（安全地拷贝）
        cv::Mat K(3, 3, CV_64F);
        std::memcpy(K.data, camera_k.data(), 9 * sizeof(double));

        cv::Mat D(1, 5, CV_64F);
        std::memcpy(D.data, camera_d.data(), 5 * sizeof(double));

        // 存储
        gobal::camera_intrinsic_ = K.clone();
        gobal::camera_distortion_ = D.clone();

        gobal::measure_tool_ = std::make_unique<MonoMeasureTool>();
        armor_pose_estimator_ = std::make_unique<ArmorPoseEstimator>();
        initTF();
        initTracker(gobal::config["armor_tracker"]);
        gobal::detect_color_ = gobal::config["common"]["detect_color"].as<int>(0);
        max_infer_running_ = gobal::config["common"]["max_infer_running"].as<int>(4);
        bool use_armor_detect_opencv =
            gobal::config["common"]["use_armor_detect_opencv"].as<bool>(false);
        bool ncnn_runeinited = false;
        bool ncnn_armorinited = false;
        use_armor_detect_ncnn = gobal::config["common"]["use_armor_detect_ncnn"].as<bool>(false);
        use_rune_detect_ncnn = gobal::config["common"]["use_rune_detect_ncnn"].as<bool>(false);

#ifdef USE_OPENVINO
    #ifdef USE_NCNN

        if (use_armor_detect_ncnn) {
            auto ncnn_config = YAML::LoadFile("/home/hy/wust_vision/config/detect_ncnn.yaml");
            armor_detector_ = DetectorFactory::createArmorDetector("ncnn", ncnn_config);
            ncnn_armorinited = true;
            WUST_MAIN(vision_logger) << "Using Armor Detector: ncnn";
        }

        if (use_rune_detect_ncnn) {
            auto ncnn_config = YAML::LoadFile("/home/hy/wust_vision/config/detect_ncnn.yaml");
            rune_detector_ = DetectorFactory::createRuneDetector("ncnn", ncnn_config);
            ncnn_runeinited = true;
            WUST_MAIN(vision_logger) << "Using Rune Detector: ncnn";
        }
    #endif
        if (!ncnn_armorinited) {
            if (use_armor_detect_opencv) {
                auto opencv_config =
                    YAML::LoadFile("/home/hy/wust_vision/config/armor_detect_opencv.yaml");
                armor_detector_ = DetectorFactory::createArmorDetector("opencv", opencv_config);
                WUST_MAIN(vision_logger) << "Using Armor Detector: opencv";
            } else {
                armor_detector_ = DetectorFactory::createArmorDetector("openvino", gobal::config);
                WUST_MAIN(vision_logger) << "Using Armor Detector: openvino";
            }
        }
        if (!ncnn_runeinited) {
            rune_detector_ = DetectorFactory::createRuneDetector("openvino", gobal::config);
            WUST_MAIN(vision_logger) << "Using Rune Detector: openvino";
        }

#elif defined(USE_TRT)
    #ifdef USE_NCNN

        if (use_armor_detect_ncnn) {
            auto ncnn_config = YAML::LoadFile("/home/hy/wust_vision/config/detect_ncnn.yaml");
            armor_detector_ = DetectorFactory::createArmorDetector("ncnn", ncnn_config);
            ncnn_armorinited = true;
            WUST_MAIN(vision_logger) << "Using Armor Detector: ncnn";
        }

        if (use_rune_detect_ncnn) {
            auto ncnn_config = YAML::LoadFile("/home/hy/wust_vision/config/detect_ncnn.yaml");
            rune_detector_ = DetectorFactory::createRuneDetector("ncnn", ncnn_config);
            ncnn_runeinited = true;
            WUST_MAIN(vision_logger) << "Using Rune Detector: ncnn";
        }
    #endif
        if (!ncnn_armorinited) {
            if (use_armor_detect_opencv) {
                auto opencv_config =
                    YAML::LoadFile("/home/hy/wust_vision/config/armor_detect_opencv.yaml");
                armor_detector_ = DetectorFactory::createArmorDetector("opencv", opencv_config);
                WUST_MAIN(vision_logger) << "Using Armor Detector: opencv";
            } else {
                armor_detector_ = DetectorFactory::createArmorDetector("tensorrt", gobal::config);
                WUST_MAIN(vision_logger) << "Using Armor Detector: tensorrt";
            }
        }
        if (!ncnn_runeinited) {
            rune_detector_ = DetectorFactory::createRuneDetector("tensorrt", gobal::config);
            WUST_MAIN(vision_logger) << "Using Rune Detector: tensorrt";
        }

#elif defined(USE_NCNN_ONLY)
        if (use_armor_detect_opencv) {
            auto opencv_config =
                YAML::LoadFile("/home/hy/wust_vision/config/armor_detect_opencv.yaml");
            armor_detector_ = DetectorFactory::createArmorDetector("opencv", opencv_config);
            WUST_MAIN(vision_logger) << "Using Armor Detector: opencv";
        } else {
            armor_detector_ = DetectorFactory::createArmorDetector("ncnn", gobal::config);
            WUST_MAIN(vision_logger) << "Using Armor Detector: ncnn";
            use_armor_detect_ncnn = true;
        }
        rune_detector_ = DetectorFactory::createRuneDetector("ncnn", gobal::config);
        use_rune_detect_ncnn = true;

        WUST_MAIN(vision_logger) << "Using Rune Detector: ncnn";

#else
        static_assert(false, "No backend defined: USE_OPENVINO or USE_TRT");
#endif

        armor_detector_->setCallback(std::bind(
            &WustVision::ArmorDetectCallback,
            this,
            std::placeholders::_1,
            std::placeholders::_2,
            std::placeholders::_3,
            std::placeholders::_4
        ));
        rune_detector_->setCallback(std::bind(
            &WustVision::RuneDetectCallback,
            this,
            std::placeholders::_1,
            std::placeholders::_2,
            std::placeholders::_3,
            std::placeholders::_4
        ));
        max_detect_armors_ = gobal::config["common"]["max_detect_armors"].as<int>(10);

        initRune(camera_info_path);

        thread_pool_ = std::make_unique<ThreadPool>(std::thread::hardware_concurrency(), 100);
        armor_solver_ = std::make_unique<ArmorSolver>(gobal::config);

    } else {
        WUST_MAIN(vision_logger) << "only nav mode";
    }

    gobal::is_inited_ = true;
    WUST_MAIN(vision_logger) << "WustVision init success";
}
void WustVision::run() {
    WUST_MAIN(vision_logger) << "WustVision run start";
    if (serial_) {
        bool if_use_nav = gobal::config["control"]["use_nav"].as<bool>(false);
        gobal::use_serial = gobal::config["control"]["use_serial"].as<bool>();
        serial_->startThread(gobal::use_serial, if_use_nav);
    }
    if (!only_nav_enable) {
        if (video_player_ && use_video) {
            video_player_->start();
        }
        if (camera_ && !use_video) {
            bool if_recorder = gobal::config["camera"]["recorder"].as<bool>(false);

            camera_->startCamera(if_recorder);
        }
        startTimer();
        toolsgobal::robot_cmd_plot_thread_ = std::thread(&robotCmdLoggerThread);
    }

    WUST_MAIN(vision_logger) << "WustVision run success";
}

void WustVision::initRune(const std::string& camera_info_path) {
    detect_r_tag_ = gobal::config["rune_detector"]["detect_r_tag"].as<bool>();
    use_manual_r = gobal::config["rune_detector"]["use_manual_r"].as<bool>();
    rune_binary_thresh_ = gobal::config["rune_detector"]["min_lightness"].as<int>();
    auto rune_solver_params = RuneSolver::RuneSolverParams {
        .compensator_type = gobal::config["rune_solver"]["compensator_type"].as<std::string>(),
        .gravity = gobal::config["rune_solver"]["gravity"].as<double>(9.8),
        .bullet_speed = gobal::config["rune_solver"]["bullet_speed"].as<double>(25.0),
        .angle_offset_thres = gobal::config["rune_solver"]["angle_offset_thres"].as<double>(0.78),
        .lost_time_thres = gobal::config["rune_solver"]["lost_time_thres"].as<double>(0.5),
        .auto_type_determined = gobal::config["rune_solver"]["auto_type_determined"].as<bool>(true),
    };
    rune_solver_ = std::make_unique<RuneSolver>(rune_solver_params);
    bool use_ypd = gobal::config["rune_solver"]["ekf"]["use_ypd"].as<bool>();
    rune_solver_->use_ypd = use_ypd;
    rune_solver_->predict_offset_ = gobal::config["rune_solver"]["predict_offset"].as<double>(0.0);
    rune_solver_->pnp_solver = std::make_unique<PnPSolver>();
    rune_solver_->pnp_solver->setObjectPoints("rune", RUNE_OBJECT_POINTS);
    std::vector<OffsetEntry> entries;
    if (gobal::config["rune_solver"]["trajectory_offset"]) {
        for (const auto& node: gobal::config["rune_solver"]["trajectory_offset"]) {
            OffsetEntry e;
            e.d_min = node["d_min"].as<double>();
            e.d_max = node["d_max"].as<double>();
            e.h_min = node["h_min"].as<double>();
            e.h_max = node["h_max"].as<double>();
            e.pitch_off = node["pitch_off"].as<double>();
            e.yaw_off = node["yaw_off"].as<double>();
            entries.push_back(e);
        }
    }
    rune_solver_->manual_compensator->updateMapFlow(entries);
    // EKF for filtering the position of R tag
    // state: x, y, z, yaw
    // measurement: x, y, z, yaw
    // f - Process function
    auto f = rune_motion_model::Predict();
    auto yf = ypdrune_motion_model::Predict();
    // h - Observation function
    auto h = rune_motion_model::Measure();
    auto yh = ypdrune_motion_model::Measure();
    // update_Q - process noise covariance matrix
    std::vector<double> q_vec =
        gobal::config["rune_solver"]["ekf"]["q_xyzyaw"].as<std::vector<double>>();

    auto u_q = [q_vec]() {
        Eigen::Matrix<double, rune_motion_model::X_N, rune_motion_model::X_N> q =
            Eigen::MatrixXd::Zero(4, 4);
        q.diagonal() << q_vec[0], q_vec[1], q_vec[2], q_vec[3];
        return q;
    };
    std::vector<double> yq_vec =
        gobal::config["rune_solver"]["ekf"]["q_ypdyaw"].as<std::vector<double>>();
    auto yu_q = [yq_vec]() {
        Eigen::Matrix<double, ypdrune_motion_model::X_N, ypdrune_motion_model::X_N> q =
            Eigen::MatrixXd::Zero(4, 4);
        q.diagonal() << yq_vec[0], yq_vec[1], yq_vec[2], yq_vec[3];
        return q;
    };
    // update_R - measurement noise covariance matrix
    std::vector<double> r_vec =
        gobal::config["rune_solver"]["ekf"]["r_xyzyaw"].as<std::vector<double>>();
    auto u_r = [r_vec](const Eigen::Matrix<double, rune_motion_model::Z_N, 1>& z) {
        Eigen::Matrix<double, rune_motion_model::Z_N, rune_motion_model::Z_N> r =
            Eigen::MatrixXd::Zero(4, 4);
        r.diagonal() << r_vec[0], r_vec[1], r_vec[2], r_vec[3];
        return r;
    };
    std::vector<double> yr_vec =
        gobal::config["rune_solver"]["ekf"]["r_ypdyaw"].as<std::vector<double>>();
    auto yu_r = [yr_vec](const Eigen::Matrix<double, ypdrune_motion_model::Z_N, 1>& z) {
        Eigen::Matrix<double, ypdrune_motion_model::Z_N, ypdrune_motion_model::Z_N> r =
            Eigen::MatrixXd::Zero(4, 4);
        r.diagonal() << yr_vec[0], yr_vec[1], yr_vec[2], yr_vec[3];
        return r;
    };
    // P - error estimate covariance matrix
    Eigen::MatrixXd p0 = Eigen::MatrixXd::Identity(4, 4);
    Eigen::MatrixXd yp0 = Eigen::MatrixXd::Identity(4, 4);
    int iteration_num = gobal::config["rune_solver"]["ekf"]["iteration_num"].as<int>(1);
    if (use_ypd) {
        rune_solver_->ekf_ypd =
            std::make_unique<ypdrune_motion_model::RuneCenterEKF>(yf, yh, yu_q, yu_r, yp0);
        rune_solver_->ekf_ypd->setAngleDims({ 0, 3 });
        rune_solver_->ekf_ypd->setIterationNum(iteration_num);
    } else {
        rune_solver_->ekf_xyz =
            std::make_unique<rune_motion_model::RuneCenterEKF>(f, h, u_q, u_r, p0);
        rune_solver_->ekf_xyz->setAngleDims({ 3 });
        rune_solver_->ekf_xyz->setIterationNum(iteration_num);
    }
}

void WustVision::startTimer() {
    if (timer_running_)
        return;
    WUST_INFO(vision_logger) << "starting timer";

    timer_running_ = true;

    double us_interval = 1e6 / static_cast<double>(gobal::control_rate);
    auto interval = std::chrono::microseconds(static_cast<int64_t>(us_interval));

    constexpr auto spin_margin = std::chrono::microseconds(200);

    timer_thread_ = std::thread([this, interval, spin_margin]() {
        auto next_time = std::chrono::steady_clock::now() + interval;
        auto last_time = std::chrono::steady_clock::now();

        while (true) {
            {
                std::unique_lock<std::mutex> lk(timer_mtx_);
                auto sleep_until = next_time - spin_margin;
                timer_cv_.wait_until(lk, sleep_until, [this]() { return !timer_running_; });
                if (!timer_running_)
                    break;
            }

            while (std::chrono::steady_clock::now() < next_time) {
                // busy‐wait
            }

            auto now = std::chrono::steady_clock::now();
            double dt_ms = std::chrono::duration<double, std::milli>(now - last_time).count();
            last_time = now;

            this->timerCallback(dt_ms);
            next_time += interval;
        }
    });
}

void WustVision::initTF() {
    gimbal2camera_x_ = gobal::config["tf"]["gimbal2camera_x"].as<double>(0.0);
    gimbal2camera_y_ = gobal::config["tf"]["gimbal2camera_y"].as<double>(0.0);
    gimbal2camera_z_ = gobal::config["tf"]["gimbal2camera_z"].as<double>(0.0);
    gimbal2camera_roll_ = gobal::config["tf"]["gimbal2camera_roll"].as<double>(0.0);
    gimbal2camera_pitch_ = gobal::config["tf"]["gimbal2camera_pitch"].as<double>(0.0);
    gimbal2camera_yaw_ = gobal::config["tf"]["gimbal2camera_yaw"].as<double>(0.0);
    // odom 是世界坐标系的根节点
    gobal::tf_tree_.setTransform("", "odom", createTf(0, 0, 0, tf::Quaternion(0, 0, 0, 1)), true);

    // camera 相对于 odom，设置 odom -> camera 的变换
    gobal::tf_tree_
        .setTransform("odom", "gimbal_odom", createTf(0, 0, 0, tf::Quaternion(0, 0, 0, 1)), true);

    gobal::tf_tree_.setTransform(
        "gimbal_odom",
        "gimbal_link",
        createTf(0, 0, 0, tf::Quaternion(0, 0, 0, 1)),
        false
    );
    gobal::gimbal2camera_roll = gimbal2camera_roll_ * M_PI / 180;
    gobal::gimbal2camera_pitch = gimbal2camera_pitch_ * M_PI / 180;
    gobal::gimbal2camera_yaw = gimbal2camera_yaw_ * M_PI / 180;
    tf::Quaternion origimbal2camera;
    origimbal2camera
        .setRPY(gobal::gimbal2camera_roll, gobal::gimbal2camera_pitch, gobal::gimbal2camera_yaw);
    gobal::tf_tree_.setTransform(
        "gimbal_link",
        "camera",
        createTf(gimbal2camera_x_, gimbal2camera_y_, gimbal2camera_z_, origimbal2camera),
        true
    );

    t_gimbal_to_camera = Eigen::Vector3d(gimbal2camera_x_, gimbal2camera_y_, gimbal2camera_z_);

    // 转换为旋转矩阵使用
    R_gimbal_camera << 0, 0, 1, -1, 0, 0, 0, -1, 0;

    // camera_optical_frame 相对于 camera，设置 camera -> camera_optical_frame
    // 的旋转变换
    double yaw = M_PI / 2;
    double roll = -M_PI / 2;
    double pitch = 0.0;

    tf::Quaternion orientation;
    orientation.setRPY(roll, pitch, yaw);

    gobal::tf_tree_
        .setTransform("camera", "camera_optical_frame", createTf(0, 0, 0, orientation), true);
}
void WustVision::initSerial() {
    SerialPortConfig cfg { /*baud*/ 115200,
                           /*csize*/ 8,
                           boost::asio::serial_port_base::parity::none,
                           boost::asio::serial_port_base::stop_bits::one,
                           boost::asio::serial_port_base::flow_control::none };

    std::string device_name = gobal::config["control"]["device_name"].as<std::string>();
    serial_ = std::make_unique<serial::Serial>();
    serial_->init(device_name, cfg);
    serial_->alpha_yaw = gobal::config["control"]["alpha_yaw"].as<double>();
    serial_->alpha_pitch = gobal::config["control"]["alpha_pitch"].as<double>();
    serial_->max_yaw_change = gobal::config["control"]["max_yaw_change"].as<double>();
    serial_->max_pitch_change = gobal::config["control"]["max_pitch_change"].as<double>();
    gobal::communication_delay_μs = gobal::config["control"]["communication_delay"].as<double>();
    int order;
    double alpha, kff, jump;
    order = gobal::config["control"]["control_filter"]["poly_order"].as<int>();
    future_window = gobal::config["control"]["control_filter"]["future_window"].as<int>();
    alpha = gobal::config["control"]["control_filter"]["base_alpha"].as<double>();
    kff = gobal::config["control"]["control_filter"]["base_k_ff"].as<double>();
    jump = gobal::config["control"]["control_filter"]["jump_threshold"].as<double>();
    control_filter_ = std::make_unique<ControlFilter>(order, future_window, alpha, kff, jump);
}

void WustVision::initTracker(const YAML::Node& config) {
    tracker_manager_ = std::make_unique<TrackerManager>(config);
}

void WustVision::runeTargetCallback(const Rune rune_target, Eigen::Matrix4d T_camera_to_odom) {
    // rune_solver_->pnp_solver is nullptr when camera_info is not received
    if (rune_solver_->pnp_solver == nullptr) {
        return;
    }

    // Keep the last detected target
    if (!rune_target.is_lost) {
        last_rune_target_ = rune_target;
    }
    double observed_angle = 0;
    if (rune_solver_->tracker_state == RuneSolver::LOST) {
        observed_angle = rune_solver_->init(rune_target, T_camera_to_odom);
    } else {
        observed_angle = rune_solver_->update(rune_target, T_camera_to_odom);
    }

    auto now = std::chrono::steady_clock::now();
    auto latency_nano =
        std::chrono::duration_cast<std::chrono::nanoseconds>(now - rune_target.timestamp).count();
    toolsgobal::latency_ms = static_cast<double>(latency_nano) / 1e6;
}

void WustVision::armorsCallback(Armors armors_, const cv::Mat& src_img) {
    if (armors_.timestamp <= tracker_manager_->last_time_) {
        // WUST_WARN(vision_logger) << "Received out-of-order armor data,
        // discarded.";
        return;
    }

    if (gobal::debug_mode_) {
        std::lock_guard<std::mutex> target_lock(img_mutex_);
        imgframe_.img = src_img.clone();
        imgframe_.timestamp = armors_.timestamp;
        armors_gobal = armors_;
    }
    if (gobal::use_calculation_) {
        command_callbackypd(armors_);
        // return;
    }
    Target target_;
    std::vector<OneTarget> one_targets_;
    auto time = armors_.timestamp;
    target_.timestamp = time;
    target_.frame_id = "gimbal_odom";
    tracker_manager_->update(target_, one_targets_, armors_, time);

    armor_target = target_;
    one_armor_targets = one_targets_;
    auto now = std::chrono::steady_clock::now();

    auto latency_nano =
        std::chrono::duration_cast<std::chrono::nanoseconds>(now - target_.timestamp).count();
    toolsgobal::latency_ms = static_cast<double>(latency_nano) / 1e6;
}

Armors WustVision::visualizeTargetProjection(
    Target armor_target_,
    std::vector<OneTarget> one_armor_targets_
) {
    Armors armor_data;
    armor_data.frame_id = "gimbal_odom";
    armor_data.timestamp = armor_target_.timestamp;

    if (armor_target_.tracking) {
        double yaw = armor_target_.yaw, r1 = armor_target_.radius_1, r2 = armor_target_.radius_2;
        float xc = armor_target_.position_.x, yc = armor_target_.position_.y,
              zc = armor_target_.position_.z;
        double d_za = armor_target_.d_za, d_zc = armor_target_.d_zc;
        xc = xc + armor_target_.velocity_.x * debug_show_dt_;
        yc = yc + armor_target_.velocity_.y * debug_show_dt_;
        zc = zc + armor_target_.velocity_.z * debug_show_dt_;
        yaw = yaw + armor_target_.v_yaw * debug_show_dt_;

        bool is_current_pair = true;

        armor_data.armors.clear();

        size_t a_n = armor_target_.armors_num;

        armor_data.armors.reserve(a_n);

        for (size_t i = 0; i < a_n; ++i) {
            double tmp_yaw = yaw + i * (2 * M_PI / a_n);
            double cos_yaw = std::cos(tmp_yaw);
            double sin_yaw = std::sin(tmp_yaw);

            tf::Position pos;
            if (a_n == 4) {
                double r = is_current_pair ? r1 : r2;
                pos.z = zc + d_zc + (is_current_pair ? 0 : d_za);
                pos.x = xc - r * cos_yaw;
                pos.y = yc - r * sin_yaw;
                is_current_pair = !is_current_pair;
            } else {
                pos.z = zc;
                pos.x = xc - r1 * cos_yaw;
                pos.y = yc - r1 * sin_yaw;
            }

            tf::Quaternion ori;
            ori.setRPY(
                M_PI / 2,
                armor_target_.id == ArmorNumber::OUTPOST ? -0.2618 : 0.2618,
                tmp_yaw
            );

            armor_data.armors.emplace_back(Armor { .type = armor_target_.type,
                                                   .pos = pos,
                                                   .ori = ori,
                                                   .is_ok = true,
                                                   .distance_to_image_center = 0.0f });
        }
    }
    for (auto one_armor_target_: one_armor_targets_) {
        if (one_armor_target_.tracking) {
            tf::Position pos;
            pos.x = one_armor_target_.position_.x + one_armor_target_.velocity_.x * debug_show_dt_;
            pos.y = one_armor_target_.position_.y + one_armor_target_.velocity_.y * debug_show_dt_;
            pos.z = one_armor_target_.position_.z + one_armor_target_.velocity_.z * debug_show_dt_;
            double tmp_yaw = one_armor_target_.yaw + one_armor_target_.v_yaw * debug_show_dt_;
            tf::Quaternion ori;
            ori.setRPY(
                M_PI / 2,
                one_armor_target_.id == ArmorNumber::OUTPOST ? -0.2618 : 0.2618,
                tmp_yaw
            );

            armor_data.armors.emplace_back(Armor { .type = one_armor_target_.type,
                                                   .pos = pos,
                                                   .ori = ori,
                                                   .is_ok = false,
                                                   .distance_to_image_center = 0.0f });
        }
    }

    return armor_data;
}
void WustVision::ArmorDetectCallback(
    const std::vector<ArmorObject>& objs,
    std::chrono::steady_clock::time_point timestamp,
    const cv::Mat& src_img,
    Eigen::Matrix4d T_camera_to_odom
) {
    std::lock_guard<std::mutex> lock(callback_mutex_);
    detect_finish_count_++;
    if (objs.size() >= max_detect_armors_) {
        WUST_WARN(vision_logger) << "Detected " << objs.size() << " objects"
                                 << "too much";
        infer_running_count_--;
        return;
    }
    if (gobal::measure_tool_ == nullptr) {
        WUST_WARN(vision_logger) << "NO camera info";
        return;
    }
    Armors armors;
    armors.timestamp = timestamp;
    armors.frame_id = "camera_optical_frame";

    armors.armors = armor_pose_estimator_->extractArmorPoses(objs, T_camera_to_odom);

    gobal::measure_tool_
        ->processDetectedArmors(objs, gobal::detect_color_, armors, T_camera_to_odom);

    infer_running_count_--;
    if (use_auto_labeler) {
        static int save_counter = 0; // 静态计数器记录保存次数

        for (const auto& obj: objs) {
            std::vector<float> csv_data;

            int number_ = formArmorNumber(obj.number);
            int color_ = formArmorColor(obj.color);

            const auto& pts = obj.is_ok ? obj.pts_binary : obj.pts;

            for (int i = 0; i < 4; ++i) {
                csv_data.push_back(pts[i].x);
                csv_data.push_back(pts[i].y);
            }

            csv_data.push_back(number_);
            csv_data.push_back(color_);

            save_counter++; // 每次处理一个对象后计数器加一
            if (save_counter % 10 == 0) { // 每10次执行一次保存
                cv::Mat img_save;
                cv::cvtColor(src_img, img_save, cv::COLOR_RGB2BGR);
                auto_labeler_->save(img_save, csv_data);
            }
        }
    }

    armorsCallback(armors, src_img);
}
void WustVision::RuneDetectCallback(
    std::vector<RuneObject>& objs,
    std::chrono::steady_clock::time_point timestamp,
    const cv::Mat& src_img,
    Eigen::Matrix4d T_camera_to_odom
) {
    std::lock_guard<std::mutex> lock(callback_mutex_);
    static bool last_rune_big = false; //Always small rune first
    detect_finish_count_++;
    // Used to draw debug info
    cv::Mat debug_img;
    if (gobal::debug_mode_) {
        debug_img = src_img.clone();
    }
    Rune rune_target;
    rune_target.frame_id = "camera_optical_frame";
    rune_target.timestamp = timestamp;
    rune_target.is_big_rune = false;

    if (!objs.empty()) {
        // Sort by probability
        std::sort(objs.begin(), objs.end(), [](const RuneObject& a, const RuneObject& b) {
            return a.prob > b.prob;
        });

        cv::Point2f r_tag;
        cv::Mat binary_roi = cv::Mat::zeros(1, 1, CV_8UC3);
        if (use_manual_r && manual_r_init) {
            gobal::measure_tool_->projectRTargetToImage(T_r, T_camera_to_odom, manual_r_box);
            manual_r_center = computeCenter(manual_r_box);
            r_tag = manual_r_center;
            if (detect_r_tag_ && !src_img.empty()) {
                std::tie(r_tag, binary_roi) =
                    rune_detector_->detectRTag(src_img, rune_binary_thresh_, manual_r_center, true);
            }
        } else {
            if (detect_r_tag_ && !src_img.empty()) {
                // Detect R tag using traditional method
                std::tie(r_tag, binary_roi) =
                    rune_detector_
                        ->detectRTag(src_img, rune_binary_thresh_, objs.at(0).pts.r_center, false);
            } else {
                // Use the average center of all objects as the center of the R tag
                r_tag = std::accumulate(
                    objs.begin(),
                    objs.end(),
                    cv::Point2f(0, 0),
                    [n = static_cast<float>(objs.size())](cv::Point2f p, auto& o) {
                        return p + o.pts.r_center / n;
                    }
                );
            }
        }

        // Assign the center of the R tag to all objects
        std::for_each(objs.begin(), objs.end(), [r = r_tag](RuneObject& obj) {
            obj.pts.r_center = r;
        });

        // Draw binary roi
        if (gobal::debug_mode_ && !debug_img.empty()) {
            cv::Rect roi =
                cv::Rect(debug_img.cols - binary_roi.cols, 0, binary_roi.cols, binary_roi.rows);
            binary_roi.copyTo(debug_img(roi));
            cv::rectangle(debug_img, roi, cv::Scalar(150, 150, 150), 2);
        }

        // The final target is the inactivated rune with the highest probability
        auto result_it = std::find_if(
            objs.begin(),
            objs.end(),
            [c = static_cast<EnemyColor>(gobal::detect_color_)](const auto& obj) -> bool {
                return obj.type == RuneType::INACTIVATED && obj.color == c;
            }
        );

        if (result_it != objs.end()) {
            rune_target.is_lost = false;
            rune_target.pts[0].x = result_it->pts.r_center.x;
            rune_target.pts[0].y = result_it->pts.r_center.y;
            rune_target.pts[1].x = result_it->pts.bottom_left.x;
            rune_target.pts[1].y = result_it->pts.bottom_left.y;
            rune_target.pts[2].x = result_it->pts.top_left.x;
            rune_target.pts[2].y = result_it->pts.top_left.y;
            rune_target.pts[3].x = result_it->pts.top_right.x;
            rune_target.pts[3].y = result_it->pts.top_right.y;
            rune_target.pts[4].x = result_it->pts.bottom_right.x;
            rune_target.pts[4].y = result_it->pts.bottom_right.y;
        } else {
            // All runes are activated
            rune_target.is_lost = true;
        }
    } else {
        // All runes are not the target color
        rune_target.is_lost = true;
    }
    infer_running_count_--;
    AttackMode mode = toAttackMode(gobal::attack_mode);
    switch (mode) {
        case AttackMode::ARMOR: {
            rune_target.is_big_rune = last_rune_big;
        } break;
        case AttackMode::BIG_RUNE: {
            rune_target.is_big_rune = true;
            last_rune_big = true;
        } break;
        case AttackMode::SMALL_RUNE: {
            rune_target.is_big_rune = false;
            last_rune_big = false;
        } break;
        case AttackMode::UNKNOWN: {
            rune_target.is_big_rune = last_rune_big;
        }
    }

    runeTargetCallback(rune_target, T_camera_to_odom);
    auto now = std::chrono::steady_clock::now();

    auto latency_nano =
        std::chrono::duration_cast<std::chrono::nanoseconds>(now - rune_target.timestamp).count();
    toolsgobal::latency_ms = static_cast<double>(latency_nano) / 1e6;

    if (gobal::debug_mode_) {
        std::lock_guard<std::mutex> target_lock(img_mutex_);
        imgframe_.img = debug_img.clone();
        imgframe_.timestamp = rune_target.timestamp;
        rune_gobal = rune_target;
        rune_objects_ = objs;
    }
}
void WustVision::calculation_manual_r(const cv::Mat& src_img) {
    manual_r_runing = true;
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
            manual_r_center = clicked_points_.front();

            float x = std::clamp(
                manual_r_center.x,
                float(half_size),
                float(src_img.cols - half_size - 1)
            );
            float y = std::clamp(
                manual_r_center.y,
                float(half_size),
                float(src_img.rows - half_size - 1)
            );
            manual_r_center = { x, y };

            manual_r_box = { {
                { x - half_size, y - half_size }, // 左上 → 对应点0
                { x - half_size, y + half_size }, // 左下 → 对应点1
                { x + half_size, y + half_size }, // 右下 → 对应点2
                { x + half_size, y - half_size } // 右上 → 对应点3
            } };

            cv::circle(temp, manual_r_center, 3, cv::Scalar(0, 255, 0), -1);
            for (int i = 0; i < 4; ++i)
                cv::line(
                    temp,
                    manual_r_box[i],
                    manual_r_box[(i + 1) % 4],
                    cv::Scalar(255, 0, 0),
                    1
                );
        }

        cv::imshow("Manual R Box", temp);
        int key = cv::waitKey(30);

        if (key == 27) { // ESC 退出，不提交
            WUST_INFO("Manual R") << "Manual box canceled.";
            manual_r_init = false;
            break;
        }

        if (key == 13 || key == 10) { // 回车提交
            if (!clicked_points_.empty()) {
                manual_r_init = true;
                WUST_INFO("Manual R")
                    << "Manual center: (" << manual_r_center.x << ", " << manual_r_center.y << ")";
                WUST_INFO("Manual R") << "Manual R Box Points Saved.";
            } else {
                WUST_ERROR("Manual R") << "No point to submit.";
                manual_r_init = false;
            }
            break;
        }

        if (key == 'b' || key == 8) {
            clicked_points_.clear();
            WUST_INFO("Manual R") << "Manual point cleared.";
        }
    }

    gobal::measure_tool_->calcRTarget(manual_r_box, T_r, T_camera_to_odom_);
    cv::destroyWindow("Manual R Box");
    cv::destroyWindow("Manual R Box");
    cv::destroyWindow("Manual R Box");
    manual_r_runing = false;
}

std::vector<cv::Point2f> WustVision::clicked_points_;
void WustVision::onMouse(int event, int x, int y, int, void*) {
    if (event == cv::EVENT_LBUTTONDOWN) {
        clicked_points_.clear(); // 总是只保留一个点
        clicked_points_.emplace_back(x, y);
        WUST_INFO("Manual R") << "Clicked Point: (" << x << ", " << y << ")";
    }
}

void WustVision::timerCallback(double dt_ms) {
    if (!gobal::is_inited_)
        return;
    timer_count_++;

    Target target;

    target = armor_target;

    std::vector<OneTarget> one_targets;
    one_targets = one_armor_targets;

    Rune rune;

    rune = rune_gobal;
    Tracker::State state;
    bool appear;
    bool one_appear = false;
    for (auto& one_target: one_targets) {
        if (one_target.tracking) {
            one_appear = true;
        }
    }
    if (target.tracking || one_appear) {
        appear = true;
        state = Tracker::TRACKING;

    } else {
        appear = false;
        state = Tracker::LOST;
    }
    auto now = std::chrono::steady_clock::now();
    AttackMode mode = toAttackMode(gobal::attack_mode);

    GimbalCmd gimbal_cmd;

    if (target.tracking || one_appear || rune_solver_->tracker_state == Tracker::State::TRACKING) {
        try {
            switch (mode) {
                case AttackMode::ARMOR: {
                    auto cmds =
                        armor_solver_->solve_vector(target, one_targets, now, future_window, dt_ms);
                    std::vector<double> future_yaw;
                    std::vector<double> future_pitch;
                    for (auto cmd: cmds) {
                        future_yaw.push_back(cmd.yaw);
                        future_pitch.push_back(cmd.pitch);
                    }
                    auto cmd = armor_solver_->solve(target, one_targets, now);
                    double raw_pitch, raw_yaw;
                    raw_pitch = cmd.pitch;
                    raw_yaw = cmd.yaw;
                    auto [filtered_yaw, filtered_pitch] =
                        control_filter_->update(raw_yaw, raw_pitch, future_yaw, future_pitch, now);
                    cmd.pitch = filtered_pitch;
                    cmd.yaw = filtered_yaw;
                    gimbal_cmd = cmd;

                } break;
                case AttackMode::SMALL_RUNE: {
                    gimbal_cmd = rune_solver_->solve();
                } break;
                case AttackMode::BIG_RUNE: {
                    gimbal_cmd = rune_solver_->solve();
                } break;
                case AttackMode::UNKNOWN: {
                    gimbal_cmd = armor_solver_->solve(target, one_targets, now);
                } break;
            }
            last_cmd_ = gimbal_cmd;
            if (gimbal_cmd.fire_advice) {
                fire_count_++;
            }
            serial_->transformGimbalCmd(gimbal_cmd, appear);
        } catch (...) {
            WUST_ERROR(vision_logger) << "solver error";
            serial_->transformGimbalCmd(last_cmd_, appear);
        }
    } else {
        serial_->transformGimbalCmd(last_cmd_, appear);
    }

    if (gobal::debug_mode_) {
        cv::Mat src;
        {
            std::lock_guard<std::mutex> lock(img_mutex_);
            src = imgframe_.img.clone();
        }

        Armors armors;

        armors = armors_gobal;

        if (mode == AttackMode::ARMOR) {
            Armors armor_data = visualizeTargetProjection(target, one_targets);

            for (auto& armor: armor_data.armors) {
                try {
                    Eigen::Vector4d pos_odom;
                    pos_odom << armor.pos.x, armor.pos.y, armor.pos.z, 1.0;

                    Eigen::Matrix4d T_odom_to_camera = T_camera_to_odom_.inverse();
                    Eigen::Vector4d pos_camera = T_odom_to_camera * pos_odom;

                    armor.target_pos.x = pos_camera.x();
                    armor.target_pos.y = pos_camera.y();
                    armor.target_pos.z = pos_camera.z();

                    Eigen::Quaterniond q_odom(armor.ori.w, armor.ori.x, armor.ori.y, armor.ori.z);
                    Eigen::Matrix3d R_odom = q_odom.normalized().toRotationMatrix();
                    Eigen::Matrix3d R_odom_to_camera = T_odom_to_camera.block<3, 3>(0, 0);

                    Eigen::Matrix3d R_camera = R_odom_to_camera * R_odom;
                    Eigen::Quaterniond q_camera(R_camera);

                    armor.target_ori.x = q_camera.x();
                    armor.target_ori.y = q_camera.y();
                    armor.target_ori.z = q_camera.z();
                    armor.target_ori.w = q_camera.w();

                } catch (const std::exception& e) {
                    WUST_ERROR(vision_logger)
                        << "Transform from odom to camera_optical_frame failed: " << e.what();
                    continue;
                }
            }

            Target_info target_info;
            target_info.select_id = gimbal_cmd.select_id;

            if (!gobal::measure_tool_->reprojectArmorsCorners(armor_data, target_info))
                return;
            write_target_log_to_json(target);
            try {
                draw_debug_overlaywrite(
                    imgframe_,
                    &armors,
                    &target_info,
                    &target,
                    state,
                    last_cmd_
                );
            } catch (const std::exception& e) {
                std::cerr << "draw_debug_overlaywrite failed: " << e.what() << '\n';
            }

        } else {
            double predict_angle =
                rune_solver_->last_pre_angle - rune_solver_->last_observed_angle_;

            try {
                drawRuneandprewrite(
                    src,
                    rune_objects_,
                    imgframe_.timestamp,
                    predict_angle,
                    last_cmd_,
                    manual_r_box
                );
            } catch (const std::exception& e) {
                std::cerr << "drawRuneandprewrite failed: " << e.what() << '\n';
            }
        }

        auto now = std::chrono::steady_clock::now();
        double t = std::chrono::duration<double>(now - toolsgobal::start_time_).count();

        {
            std::lock_guard<std::mutex> lock(toolsgobal::robot_cmd_mutex_);
            toolsgobal::time_log_.push_back(t);
            toolsgobal::cmd_yaw_log_.push_back(last_cmd_.yaw);
            toolsgobal::cmd_pitch_log_.push_back(last_cmd_.pitch);
            double rune_obs = rune_solver_->last_observed_angle_;
            double rune_pre = rune_solver_->last_pre_angle;
            toolsgobal::rune_obs_log_.push_back(rune_obs);
            toolsgobal::rune_pre_log_.push_back(rune_pre);
            if (!armors.armors.empty()) {
                std::vector<Armor> ok_armors;
                for (const auto& armor: armors.armors) {
                    if (armor.is_ok && armor.number != ArmorNumber::OUTPOST) {
                        ok_armors.push_back(armor);
                    }
                }

                if (!ok_armors.empty()) {
                    auto min_armor_it = std::min_element(
                        ok_armors.begin(),
                        ok_armors.end(),
                        [](const Armor& a, const Armor& b) {
                            return a.distance_to_image_center < b.distance_to_image_center;
                        }
                    );

                    const Armor& min_armor = *min_armor_it;
                    last_armor_ = min_armor;

                    last_distance = std::sqrt(
                        min_armor.target_pos.x * min_armor.target_pos.x
                        + min_armor.target_pos.y * min_armor.target_pos.y
                        + min_armor.target_pos.z * min_armor.target_pos.z
                    );
                    double armor_yaw = last_armor_.yaw;
                    armor_yaw = this->last_armor_yaw
                        + angles::shortest_angular_distance(this->last_armor_yaw, armor_yaw);
                    this->last_armor_yaw = armor_yaw;
                    double ypd_y = std::atan2(last_armor_.target_pos.y, last_armor_.target_pos.x);

                    ypd_y = this->last_ypd_y
                        + angles::shortest_angular_distance(this->last_ypd_y, ypd_y);
                    this->last_ypd_y = ypd_y;
                    last_ypd_p = std::atan2(
                        last_armor_.target_pos.z,
                        std::sqrt(
                            last_armor_.target_pos.x * last_armor_.target_pos.x
                            + last_armor_.target_pos.y * last_armor_.target_pos.y
                        )
                    );
                    toolsgobal::armor_yaw_log_.push_back((last_armor_yaw / M_PI * 180));
                    toolsgobal::armor_x_log_.push_back(last_armor_.target_pos.x);
                    toolsgobal::armor_y_log_.push_back(last_armor_.target_pos.y);
                    toolsgobal::armor_z_log_.push_back(last_armor_.target_pos.z);
                    toolsgobal::ypd_y_log_.push_back(last_ypd_y);
                    toolsgobal::ypd_p_log_.push_back(last_ypd_p);

                    toolsgobal::armor_dis_log_.push_back(last_distance);
                } else {
                    toolsgobal::armor_yaw_log_.push_back((last_armor_yaw / M_PI * 180));
                    toolsgobal::armor_x_log_.push_back(last_armor_.target_pos.x);
                    toolsgobal::armor_y_log_.push_back(last_armor_.target_pos.y);
                    toolsgobal::armor_z_log_.push_back(last_armor_.target_pos.z);
                    toolsgobal::ypd_y_log_.push_back(last_ypd_y);
                    toolsgobal::ypd_p_log_.push_back(last_ypd_p);
                    toolsgobal::armor_dis_log_.push_back(last_distance);
                }

            } else {
                toolsgobal::armor_yaw_log_.push_back((last_armor_yaw / M_PI * 180));
                toolsgobal::armor_x_log_.push_back(last_armor_.target_pos.x);
                toolsgobal::armor_y_log_.push_back(last_armor_.target_pos.y);
                toolsgobal::armor_z_log_.push_back(last_armor_.target_pos.z);
                toolsgobal::ypd_y_log_.push_back(last_ypd_y);
                toolsgobal::ypd_p_log_.push_back(last_ypd_p);
                toolsgobal::armor_dis_log_.push_back(last_distance);
            }

            if (toolsgobal::time_log_.size() > 1000) {
                toolsgobal::time_log_.erase(toolsgobal::time_log_.begin());
                toolsgobal::cmd_yaw_log_.erase(toolsgobal::cmd_yaw_log_.begin());
                toolsgobal::cmd_pitch_log_.erase(toolsgobal::cmd_pitch_log_.begin());
                toolsgobal::armor_dis_log_.erase(toolsgobal::armor_dis_log_.begin());
                toolsgobal::armor_yaw_log_.erase(toolsgobal::armor_yaw_log_.begin());
                toolsgobal::armor_x_log_.erase(toolsgobal::armor_x_log_.begin());
                toolsgobal::armor_y_log_.erase(toolsgobal::armor_y_log_.begin());
                toolsgobal::armor_z_log_.erase(toolsgobal::armor_z_log_.begin());
                toolsgobal::ypd_y_log_.erase(toolsgobal::ypd_y_log_.begin());
                toolsgobal::ypd_p_log_.erase(toolsgobal::ypd_p_log_.begin());
                toolsgobal::rune_obs_log_.erase(toolsgobal::rune_obs_log_.begin());
                toolsgobal::rune_pre_log_.erase(toolsgobal::rune_pre_log_.begin());
            }
        }
    }
}
void WustVision::processImage(const ImageFrame& frame, Eigen::Matrix3d R_gimbal2odom) {
    img_recv_count_++;
    if (infer_running_count_.load() >= max_infer_running_) {
        return;
    }
    cv::Mat img;
    if (!use_video) {
        img = convertToMatrgb(frame);
    } else {
        img = convertToMatbgr(frame);
    }

    // Step 1: gimbal → odom
    Eigen::Matrix4d T_gimbal_to_odom = Eigen::Matrix4d::Identity();
    T_gimbal_to_odom.block<3, 3>(0, 0) = R_gimbal2odom;

    // Step 2: camera → gimbal （取 R 的转置）
    Eigen::Matrix3d R_camera_to_gimbal = R_gimbal_camera;
    Eigen::Vector3d t_camera_to_gimbal = -R_camera_to_gimbal * t_gimbal_to_camera;

    Eigen::Matrix4d T_camera_to_gimbal = Eigen::Matrix4d::Identity();
    T_camera_to_gimbal.block<3, 3>(0, 0) = R_camera_to_gimbal;
    T_camera_to_gimbal.block<3, 1>(0, 3) = t_camera_to_gimbal;

    // Step 3: camera → odom
    Eigen::Matrix4d T_camera_to_odom = T_gimbal_to_odom * T_camera_to_gimbal;
    T_camera_to_odom_ = T_camera_to_odom;

    infer_running_count_++;
    printStats();

    AttackMode mode = toAttackMode(gobal::attack_mode);
    switch (mode) {
        case AttackMode::ARMOR: {
            armor_detector_->pushInput(img, frame.timestamp, T_camera_to_odom);
        } break;
        case AttackMode::SMALL_RUNE: {
            if (use_manual_r && !manual_r_init && !manual_r_runing) {
                calculation_manual_r(img);
                detect_finish_count_++;
                infer_running_count_--;
                return;
            }
            rune_detector_->pushInput(img, frame.timestamp, T_camera_to_odom);
        } break;
        case AttackMode::BIG_RUNE: {
            rune_detector_->pushInput(img, frame.timestamp, T_camera_to_odom);
        }
        case AttackMode::UNKNOWN: {
            armor_detector_->pushInput(img, frame.timestamp, T_camera_to_odom);
        } break;
    }
}

void WustVision::printStats() {
    static int timer_check_count = 0;
    using namespace std::chrono;

    auto now = steady_clock::now();

    if (last_stat_time_steady_.time_since_epoch().count() == 0) {
        last_stat_time_steady_ = now;
        return;
    }

    auto elapsed = duration_cast<duration<double>>(now - last_stat_time_steady_);
    if (elapsed.count() >= 1.0) {
        if (timer_count_ < gobal::control_rate / 10) {
            timer_check_count++;
        }
        if (timer_check_count > 5) {
            stopTimer();
            startTimer();
            timer_check_count = 0;
        }
        WUST_INFO(vision_logger) << "Rec: " << img_recv_count_ << ", Det: " << detect_finish_count_
                                 << ", Fps: " << detect_finish_count_ / elapsed.count()
                                 << ", Lat: " << toolsgobal::latency_ms << "ms"
                                 << ", Fire: " << fire_count_ << ", Tc: " << timer_count_;
        timer_count_ = 0;
        img_recv_count_ = 0;
        detect_finish_count_ = 0;
        fire_count_ = 0;
        last_stat_time_steady_ = now;
    }
}
