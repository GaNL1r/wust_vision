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
#include "common/debug/tools.hpp"
#include "common/debug/toolsgobal.hpp"
#include "common/gobal.hpp"
#include "common/logger.hpp"
#include "common/tf.hpp"
#include "common/utils.hpp"
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

    if (!only_nav_enable_) {
        auto_labeler_.reset();
        if (use_video_) {
            video_player_->stop();
        } else {
            if (camera_) {
                camera_->stopCamera();
                camera_.reset();
            }
        }
        if (use_omni_) {
            omni_manager_->stop();
            omni_manager_.reset();
        }
        if (timer_) {
            timer_->stop();
            timer_.reset();
        }

        armor_detector_.reset();
        rune_detector_.reset();
#ifdef USE_TRT
        cudaDeviceSynchronize();
        cudaDeviceReset();
#endif
#ifdef USE_NCNN
        if (gobal::use_detect_ncnn_count > 0) {
            ncnn::destroy_gpu_instance();
        }
#endif
        gobal::measure_tool.reset();

        if (gobal::thread_pool) {
            gobal::thread_pool->waitUntilEmpty();
            gobal::thread_pool.reset();
        }
        if (toolsgobal::debug_thread_.joinable()) {
            toolsgobal::debug_thread_.join();
        }
    }
    if (serial_) {
        serial_->stopThread();
        serial_.reset();
    }

    WUST_MAIN(vision_logger_) << "WustVision shutdown complete.";
}
bool WustVision::init() {
    WUST_MAIN(vision_logger_) << "WustVision init start";
    gobal::config = YAML::LoadFile(ROOT_CONFIG);

    std::string log_level_ = gobal::config["logger"]["log_level"].as<std::string>("INFO");
    std::string log_path_ = gobal::config["logger"]["log_path"].as<std::string>("wust_log");
    bool use_logcli = gobal::config["logger"]["use_logcli"].as<bool>();
    bool use_logfile = gobal::config["logger"]["use_logfile"].as<bool>();
    bool use_simplelog = gobal::config["logger"]["use_simplelog"].as<bool>();
    initLogger(log_level_, log_path_, use_logcli, use_logfile, use_simplelog);
    gobal::control_rate = gobal::config["control"]["control_rate"].as<int>();
    initSerial();
    only_nav_enable_ = gobal::config["common"]["only_nav_enable"].as<bool>();
    if (!only_nav_enable_) {
        gobal::attack_mode = gobal::config["common"]["init_attack_mode"].as<int>();
        gobal::debug_mode = gobal::config["debug"]["debug_mode"].as<bool>();
        toolsgobal::debug_w = gobal::config["debug"]["debug_w"].as<int>(640);
        toolsgobal::debug_h = gobal::config["debug"]["debug_h"].as<int>(480);
        debug_show_dt_ = gobal::config["debug"]["debug_show_dt"].as<double>(0.05);
        toolsgobal::debug_fps = gobal::config["debug"]["debug_fps"].as<double>(30);
        use_calculation_ = gobal::config["common"]["use_calculation"].as<bool>();

        use_video_ = gobal::config["camera"]["video_player"]["use"].as<bool>(false);
        if (use_video_) {
            std::string video_play_path =
                gobal::config["camera"]["video_player"]["path"].as<std::string>("");
            int video_play_fps = gobal::config["camera"]["video_player"]["fps"].as<int>(30);
            int start_frame = gobal::config["camera"]["video_player"]["start_frame"].as<int>(0);
            bool loop = gobal::config["camera"]["video_player"]["loop"].as<bool>(false);
            video_alpha_ = gobal::config["camera"]["video_player"]["alpha"].as<double>(1.0);
            video_beta_ = gobal::config["camera"]["video_player"]["beta"].as<int>(0);
            video_player_ =
                std::make_unique<VideoPlayer>(video_play_path, video_play_fps, start_frame, loop);
            video_player_->enablehighPriorityAndCpuidPriority(
                gobal::config["camera"]["video_player"]["use_high_priority"].as<bool>(false),
                gobal::config["camera"]["video_player"]["high_priority_cpu_id"].as<int>(0),
                gobal::config["camera"]["video_player"]["high_priority_cpu_priority"].as<int>(0),
                gobal::config["camera"]["video_player"]["use_sched_fifo"].as<bool>(false)
            );
            video_player_->setCallback([this](ImageFrame& frame) {
                static bool first_is_inited = false;

                if (gobal::is_inited_) {
                    img_recv_count_++;
                    if (infer_running_count_.load() >= max_infer_running_) {
                        return;
                    }

                    frame.R_gimbal2odom =
                        Eigen::AngleAxisd(gobal::last_yaw, Eigen::Vector3d::UnitZ())
                        * Eigen::AngleAxisd(gobal::last_pitch, Eigen::Vector3d::UnitY())
                        * Eigen::AngleAxisd(gobal::last_roll, Eigen::Vector3d::UnitX());
                    frame.v = Eigen::Vector3d(gobal::last_v_x, gobal::last_v_y, gobal::last_v_z);
                    gobal::thread_pool->enqueue(
                        [frame = std::move(frame), this]() {
                            infer_running_count_++;
                            processImage(frame);
                            // auto t0 =time_utils::now();
                            // double time_used = time_utils::durationMs(frame.timestamp, t0);
                            // std::cout << "time used: " << time_used << std::endl;
                            detect_finish_count_++;
                            infer_running_count_--;
                        },
                        -1
                    );

                } else {
                    return;
                }
            });

        } else {
            camera_ = std::make_unique<HikCamera>();
            std::string target_sn = gobal::config["camera"]["target_sn"].as<std::string>();
            if (!camera_->initializeCamera(target_sn)) {
                WUST_ERROR(vision_logger_) << "Camera initialization failed.";
                return false;
            }

            camera_->setParameters(
                gobal::config["camera"]["acquisition_frame_rate"].as<int>(),
                gobal::config["camera"]["exposure_time"].as<int>(),
                gobal::config["camera"]["gain"].as<double>(),
                gobal::config["camera"]["gamma"].as<double>(),
                gobal::config["camera"]["adc_bit_depth"].as<std::string>(),
                gobal::config["camera"]["pixel_format"].as<std::string>(),
                gobal::config["camera"]["acquisition_frame_rate_enable"].as<bool>(),
                gobal::config["camera"]["reverse_x"].as<bool>(false),
                gobal::config["camera"]["reverse_y"].as<bool>(false)
            );
            camera_->enablehighPriorityAndCpuidPriority(
                gobal::config["camera"]["use_high_priority"].as<bool>(false),
                gobal::config["camera"]["high_priority_cpu_id"].as<int>(0),
                gobal::config["camera"]["high_priority_cpu_priority"].as<int>(0),
                gobal::config["camera"]["use_sched_fifo"].as<bool>(false)
            );
            camera_->setFrameCallback([this](const ImageFrame& frame) {
                static bool first_is_inited = false;

                if (gobal::is_inited_) {
                    img_recv_count_++;
                    if (infer_running_count_.load() >= max_infer_running_) {
                        return;
                    }
                    gobal::thread_pool->enqueue(
                        [frame = std::move(frame), this]() {
                            infer_running_count_++;
                            processImage(frame);
                            detect_finish_count_++;
                            infer_running_count_--;
                        },
                        -1
                    );

                } else {
                    return;
                }
            });
        }
        use_auto_labeler_ = gobal::config["common"]["use_auto_labeler"].as<bool>(false);
        if (use_auto_labeler_) {
            auto_labeler_ = std::make_unique<Labeler>();
        }

        const std::string camera_info_path =
            gobal::config["camera"]["camera_info_path"].as<std::string>();
        YAML::Node config_camera_info = YAML::LoadFile(camera_info_path);
        std::vector<double> camera_k =
            config_camera_info["camera_matrix"]["data"].as<std::vector<double>>();
        std::vector<double> camera_d =
            config_camera_info["distortion_coefficients"]["data"].as<std::vector<double>>();

        assert(camera_k.size() == 9);
        assert(camera_d.size() == 5);

        cv::Mat K(3, 3, CV_64F);
        std::memcpy(K.data, camera_k.data(), 9 * sizeof(double));

        cv::Mat D(1, 5, CV_64F);
        std::memcpy(D.data, camera_d.data(), 5 * sizeof(double));

        gobal::camera_intrinsic = K.clone();
        gobal::camera_distortion = D.clone();

        gobal::measure_tool = std::make_unique<MonoMeasureTool>();
        armor_pose_estimator_ = std::make_unique<ArmorPoseEstimator>();
        bool use_ba = gobal::config["common"]["use_ba"].as<bool>(false);
        if (use_ba) {
            armor_pose_estimator_->enableBA(true);
        } else {
            armor_pose_estimator_->enableBA(false);
        }
        initTF();
        initTracker(gobal::config["armor_tracker"]);
        gobal::detect_color = gobal::config["common"]["detect_color"].as<int>(0);
        max_infer_running_ = gobal::config["common"]["max_infer_running"].as<int>(4);
        initDetector();
        initRune(camera_info_path); //无论是否使用仍然初始化rune_solver
        max_detect_armors_ = gobal::config["common"]["max_detect_armors"].as<int>(10);
        int thread_multiplier = gobal::config["common"]["thread_multiplier"].as<int>(1);
        gobal::thread_pool =
            std::make_unique<ThreadPool>(std::thread::hardware_concurrency() * thread_multiplier);
        armor_solver_ = std::make_unique<ArmorSolver>(gobal::config);
        use_omni_ = gobal::config["common"]["use_omni"].as<bool>(false);
        if (use_omni_) {
            hit_omni_dt_ = gobal::config["common"]["hit_omni_dt"].as<double>(0.1);
            receive_omni_dt_ = gobal::config["common"]["receive_omni_dt"].as<double>(0.1);
            auto omni_config = YAML::LoadFile(OMNI_CONFIG);
            omni_manager_ = std::make_unique<OmniManager>(omni_config);
        }
        timer_ = std::make_unique<Timer>();
    } else {
        WUST_MAIN(vision_logger_) << "only nav mode";
    }

    gobal::is_inited_ = true;
    WUST_MAIN(vision_logger_) << "WustVision init success";
    return true;
}
void WustVision::start() {
    WUST_MAIN(vision_logger_) << "WustVision run start";
    if (serial_) {
        bool if_use_nav = gobal::config["control"]["use_nav"].as<bool>(false);
        gobal::use_serial = gobal::config["control"]["use_serial"].as<bool>();
        serial_->startThread(gobal::use_serial, if_use_nav);
    }
    if (!only_nav_enable_) {
        if (video_player_ && use_video_) {
            video_player_->start();
        }
        if (camera_ && !use_video_) {
            bool if_recorder = gobal::config["camera"]["recorder"].as<bool>(false);
            camera_->startCamera(if_recorder);
        }
        if (use_omni_ && omni_manager_) {
            omni_manager_->start();
        }
        if (timer_) {
            auto timercallback = std::bind(&WustVision::timerCallback, this, std::placeholders::_1);
            double rate_hz = static_cast<double>(gobal::control_rate);
            timer_->start(rate_hz, timercallback);
        }
    }
    if (gobal::debug_mode) {
        toolsgobal::debug_thread_ = std::thread([this]() { this->debugThread(); });
    }

    WUST_MAIN(vision_logger_) << "WustVision run success";
}
void WustVision::initDetector() {
    std::string armor_detect_backend =
        gobal::config["common"]["armor_detect_backend"].as<std::string>("");
    std::string rune_detect_backend =
        gobal::config["common"]["rune_detect_backend"].as<std::string>("");

    auto isBackendEnabled = [](const std::string& backend) -> bool {
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
            gobal::use_detect_ncnn_count++;
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
        armor_detect_config_ = YAML::LoadFile(config_path);
        return DetectorFactory::createArmorDetector(backend, armor_detect_config_, true);
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

    if (armor_detect_backend.empty()) {
        throw std::runtime_error("armor_detect_backend not set in config.");
    }
    armor_detector_ = loadArmorDetectorBackend(armor_detect_backend);
    WUST_MAIN(vision_logger_) << "Using Armor Detector: " << armor_detect_backend;

    armor_detector_->setCallback(std::bind(
        &WustVision::ArmorDetectCallback,
        this,
        std::placeholders::_1,
        std::placeholders::_2
    ));
#ifdef USE_RUNE
    if (rune_detect_backend.empty()) {
        throw std::runtime_error("rune_detect_backend not set in config.");
    }
    rune_detector_ = loadRuneDetectorBackend(rune_detect_backend);
    WUST_MAIN(vision_logger_) << "Using Rune Detector: " << rune_detect_backend;
    rune_detector_->setCallback(std::bind(
        &WustVision::RuneDetectCallback,
        this,
        std::placeholders::_1,
        std::placeholders::_2
    ));

#endif
}
void WustVision::initRune(const std::string& camera_info_path) {
#ifdef USE_RUNE
    detect_r_tag_ = rune_detect_config_["rune_detector"]["detect_r_tag"].as<bool>();
    use_manual_r_ = rune_detect_config_["rune_detector"]["use_manual_r"].as<bool>();
    rune_binary_thresh_ = rune_detect_config_["rune_detector"]["min_lightness"].as<int>();
#endif

    rune_solver_ = std::make_unique<RuneSolver>(gobal::config);
}

void WustVision::initTF() {
    gimbal2camera_x_ = gobal::config["tf"]["gimbal2camera_x"].as<double>(0.0);
    gimbal2camera_y_ = gobal::config["tf"]["gimbal2camera_y"].as<double>(0.0);
    gimbal2camera_z_ = gobal::config["tf"]["gimbal2camera_z"].as<double>(0.0);
    gimbal2camera_roll_ = gobal::config["tf"]["gimbal2camera_roll"].as<double>(0.0);
    gimbal2camera_pitch_ = gobal::config["tf"]["gimbal2camera_pitch"].as<double>(0.0);
    gimbal2camera_yaw_ = gobal::config["tf"]["gimbal2camera_yaw"].as<double>(0.0);

    t_gimbal_to_camera_ = Eigen::Vector3d(gimbal2camera_x_, gimbal2camera_y_, gimbal2camera_z_);

    // 转换为旋转矩阵使用
    R_camera2gimbal_ << 0, 0, 1, -1, 0, 0, 0, -1, 0;
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
    serial_->alpha_yaw_ = gobal::config["control"]["alpha_yaw"].as<double>();
    serial_->alpha_pitch_ = gobal::config["control"]["alpha_pitch"].as<double>();
    serial_->max_yaw_change_ = gobal::config["control"]["max_yaw_change"].as<double>();
    serial_->max_pitch_change_ = gobal::config["control"]["max_pitch_change"].as<double>();
    gobal::communication_delay_μs = gobal::config["control"]["communication_delay"].as<double>();
    jump_yaw = gobal::config["control"]["jump_yaw"].as<double>();
}

void WustVision::initTracker(const YAML::Node& config) {
    tracker_manager_ = std::make_unique<TrackerManager>(config);
}

void WustVision::runeTargetCallback(
    const rune::Rune rune_target,
    Eigen::Matrix4d T_camera_to_odom
) {
    // rune_solver_->pnp_solver is nullptr when camera_info is not received
    if (rune_solver_->pnp_solver_ == nullptr) {
        return;
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

void WustVision::armorsCallback(
    armor::Armors armors_,
    const cv::Mat& src_img,
    const Eigen::Matrix3d& R_gimbal2odom,
    const Eigen::Vector3d& v
) {
    if (armors_.timestamp <= tracker_manager_->last_time_) {
        // WUST_WARN(vision_logger) << "Received out-of-order armor data,
        // discarded.";
        return;
    }

    if (gobal::debug_mode) {
        std::lock_guard<std::mutex> target_lock(img_mutex_);
        imgframe_.img = src_img.clone();
        imgframe_.timestamp = armors_.timestamp;
        armors_gobal_ = armors_;
    }
    if (use_calculation_) {
        commandCallbackYpd(armors_);
    }
    armor::Target target_;
    std::vector<armor::OneTarget> one_targets_;
    auto time = armors_.timestamp;

    tracker_manager_->update(target_, one_targets_, armors_, time, R_gimbal2odom, v);

    armor_target_ = target_;
    one_armor_targets_ = one_targets_;
    auto now = std::chrono::steady_clock::now();

    auto latency_nano =
        std::chrono::duration_cast<std::chrono::nanoseconds>(now - target_.timestamp).count();
    toolsgobal::latency_averager.add(latency_nano);
    toolsgobal::latency_ms = toolsgobal::latency_averager.average_ms();
}

void WustVision::ArmorDetectCallback(
    const std::vector<armor::ArmorObject>& objs,
    const CommonFrame& frame
) {
    std::lock_guard<std::mutex> lock(callback_mutex_);

    std::vector<armor::ArmorObject> sorted_objs = objs;

    if (sorted_objs.size() > max_detect_armors_) {
        WUST_WARN(vision_logger_) << "Detected " << sorted_objs.size()
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
    armors.frame_id = "camera_optical_frame";

    armors.armors = armor_pose_estimator_->extractArmorPoses(
        sorted_objs,
        frame.T_camera_to_odom,
        gobal::camera_intrinsic,
        gobal::camera_distortion
    );

    gobal::measure_tool->processDetectedArmors(
        sorted_objs,
        armors,
        frame.T_camera_to_odom,
        gobal::camera_intrinsic,
        gobal::camera_distortion
    );

    if (use_auto_labeler_) {
        saveAutoLabelData(objs, frame);
    }
    Eigen::Matrix3d R_gimbal2odom =
        utils::getRGimbalToOdom(T_camera_to_odom_, R_camera2gimbal_, t_gimbal_to_camera_);

    armorsCallback(armors, frame.src_img, R_gimbal2odom, frame.v);
    T_camera_to_odom_ = frame.T_camera_to_odom;
}

void WustVision::RuneDetectCallback(std::vector<rune::RuneObject>& objs, const CommonFrame& frame) {
    std::lock_guard<std::mutex> lock(callback_mutex_);
    static bool last_rune_big = false; //Always small rune first

    // Used to draw debug info
    cv::Mat debug_img;
    if (gobal::debug_mode) {
        debug_img = frame.src_img.clone();
    }
    rune::Rune rune_target;
    rune_target.frame_id = "camera_optical_frame";
    rune_target.timestamp = frame.timestamp;
    rune_target.is_big_rune = false;

    if (!objs.empty()) {
        // Sort by probability
        std::sort(
            objs.begin(),
            objs.end(),
            [](const rune::RuneObject& a, const rune::RuneObject& b) { return a.prob > b.prob; }
        );

        cv::Point2f r_tag;
        cv::Mat binary_roi = cv::Mat::zeros(1, 1, CV_8UC3);
        if (use_manual_r_ && manual_r_init_) {
            gobal::measure_tool->projectRTargetToImage(
                T_r_,
                frame.T_camera_to_odom,
                manual_r_box_,
                gobal::camera_intrinsic,
                gobal::camera_distortion
            );
            manual_r_center_ = utils::computeCenter(manual_r_box_);
            r_tag = manual_r_center_;
            if (detect_r_tag_ && !frame.src_img.empty()) {
                std::tie(r_tag, binary_roi) =
                    rune_detector_
                        ->detectRTag(frame.src_img, rune_binary_thresh_, manual_r_center_, true);
            }
        } else {
            if (detect_r_tag_ && !frame.src_img.empty()) {
                // Detect R tag using traditional method
                std::tie(r_tag, binary_roi) = rune_detector_->detectRTag(
                    frame.src_img,
                    rune_binary_thresh_,
                    objs.at(0).pts.r_center,
                    false
                );
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
        std::for_each(objs.begin(), objs.end(), [r = r_tag](rune::RuneObject& obj) {
            obj.pts.r_center = r;
        });

        // Draw binary roi
        if (gobal::debug_mode && !debug_img.empty()) {
            cv::Rect roi =
                cv::Rect(debug_img.cols - binary_roi.cols, 0, binary_roi.cols, binary_roi.rows);
            binary_roi.copyTo(debug_img(roi));
            cv::rectangle(debug_img, roi, cv::Scalar(150, 150, 150), 2);
        }

        // The final target is the inactivated rune with the highest probability
        auto result_it = std::find_if(
            objs.begin(),
            objs.end(),
            [c = static_cast<EnemyColor>(gobal::detect_color)](const auto& obj) -> bool {
                return obj.type == rune::RuneType::INACTIVATED && obj.color == c;
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

    runeTargetCallback(rune_target, frame.T_camera_to_odom);
    T_camera_to_odom_ = frame.T_camera_to_odom;
    auto now = std::chrono::steady_clock::now();

    auto latency_nano =
        std::chrono::duration_cast<std::chrono::nanoseconds>(now - rune_target.timestamp).count();
    toolsgobal::latency_averager.add(latency_nano);
    toolsgobal::latency_ms = toolsgobal::latency_averager.average_ms();

    if (gobal::debug_mode) {
        std::lock_guard<std::mutex> target_lock(img_mutex_);
        imgframe_.img = debug_img.clone();
        imgframe_.timestamp = rune_target.timestamp;
        rune_gobal_ = rune_target;
        rune_objects_ = objs;
    }
}

GimbalCmd WustVision::solveByMode(
    AttackMode mode,
    const ArmorSloverTarget& armor_slover_target,
    const std::chrono::steady_clock::time_point& now
) {
    switch (mode) {
        case AttackMode::ARMOR: {
            auto cmd = armor_solver_->solve(armor_slover_target, now);
            auto next_time =
                now + std::chrono::microseconds(static_cast<int64_t>(1e6 / gobal::control_rate));
            auto next_cmd = armor_solver_->solve(armor_slover_target, next_time);
            if (std::abs(cmd.yaw - next_cmd.yaw) > jump_yaw
                || std::abs(cmd.yaw - gobal::last_cmd.yaw) > jump_yaw)
                cmd.fire_advice = false;
            return cmd;
        }
        case AttackMode::SMALL_RUNE:
        case AttackMode::BIG_RUNE:
            return rune_solver_->solve();
        case AttackMode::UNKNOWN:
        default:
            return armor_solver_->solve(armor_slover_target, now);
    }
}
void WustVision::timerCallback(double dt_ms) {
    if (!gobal::is_inited_)
        return;

    auto now = std::chrono::steady_clock::now();
    timer_count_++;

    armor::Target target = armor_target_;
    std::vector<armor::OneTarget> one_targets = one_armor_targets_;

    if (std::chrono::duration<double>(now - last_track_target_).count() > hit_omni_dt_) {
        for (const auto& omni: gobal::omni_targets) {
            if (std::abs(std::chrono::duration<double>(now - omni.timestamp).count())
                <= receive_omni_dt_) {
                one_targets.push_back(omni);
            }
        }
    }

    bool appear = utils::checkTargetAppear(target, one_targets);
    Tracker::State state = appear ? Tracker::TRACKING : Tracker::LOST;
    if (appear)
        last_track_target_ = now;
    AttackMode mode = toAttackMode(gobal::attack_mode);

    GimbalCmd gimbal_cmd;

    if (appear || rune_solver_->tracker_state == Tracker::TRACKING) {
        if (appear || rune_solver_->tracker_state == Tracker::TRACKING) {
            try {
                ArmorSloverTarget armor_slover_target;
                armor_slover_target.one_targets = one_targets;
                armor_slover_target.target = target;
                gimbal_cmd = solveByMode(mode, armor_slover_target, now);
                gobal::last_cmd = gimbal_cmd;
                if (gimbal_cmd.fire_advice)
                    fire_count_++;
            } catch (const std::exception& e) {
                std::cerr << "solver error: " << e.what() << '\n';
                gimbal_cmd = gobal::last_cmd;
            }
        } else {
            gimbal_cmd = gobal::last_cmd;
        }

        serial_->transformGimbalCmd(gimbal_cmd, appear);
    }

    if (gobal::debug_mode) {
        //debuglog();
    }
}
void WustVision::processImage(const ImageFrame& frame) {
    CommonFrame common_frame;
    common_frame.timestamp = frame.timestamp;
    common_frame.v = frame.v;
    auto t1 = time_utils::now();
    if (!use_video_) {
        common_frame.src_img = convertToMatrgb(frame);
    } else {
        common_frame.src_img = convertToMatbgr(frame);
        common_frame.src_img.convertTo(common_frame.src_img, -1, video_alpha_, video_beta_);
    }
    common_frame.T_camera_to_odom = utils::computeCameraToOdomTransform(
        frame.R_gimbal2odom,
        R_camera2gimbal_,
        t_gimbal_to_camera_
    );
    auto t2 = time_utils::now();
    printStats();

    AttackMode mode = toAttackMode(gobal::attack_mode);
    switch (mode) {
        case AttackMode::ARMOR: {
            if (armor_detector_) {
                armor_detector_->pushInput(common_frame);
            }
        } break;
        case AttackMode::SMALL_RUNE: {
            // if (use_manual_r_ && !manual_r_init_ && !manual_r_runing_) {
            //     cv::Point2f center(common_frame.src_img.cols/2.0,common_frame.src_img.rows/2.0);
            //     calculationManualR(center);
            //     return;
            // }
            if (use_manual_r_ && gobal::if_manual_reset) {
                cv::Point2f center(
                    common_frame.src_img.cols / 2.0,
                    common_frame.src_img.rows / 2.0
                );
                calculationManualR(center);
            }
            if (rune_detector_) {
                rune_detector_->pushInput(common_frame);
            }

        } break;
        case AttackMode::BIG_RUNE: {
            // if (use_manual_r_ && !manual_r_init_ && !manual_r_runing_) {
            //     calculationManualR(common_frame.src_img);
            //     return;
            // }
            if (use_manual_r_ && gobal::if_manual_reset) {
                cv::Point2f center(
                    common_frame.src_img.cols / 2.0,
                    common_frame.src_img.rows / 2.0
                );
                calculationManualR(center);
            }

            if (rune_detector_) {
                rune_detector_->pushInput(common_frame);
            }
        } break;
        case AttackMode::UNKNOWN: {
            if (armor_detector_) {
                armor_detector_->pushInput(common_frame);
            }
        } break;
    }
    auto t3 = time_utils::now();
    //std::cout<<"time: "<<time_utils::durationMs(t2,t1)<<" "<<time_utils::durationMs(t3,t2)<<std::endl;
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
            if (timer_) {
                auto timercallback =
                    std::bind(&WustVision::timerCallback, this, std::placeholders::_1);
                double rate_hz = static_cast<double>(gobal::control_rate);
                timer_->start(rate_hz, timercallback);
            }
            timer_check_count = 0;
        }
        WUST_INFO(vision_logger_) << "Rec: " << img_recv_count_ << ", Det: " << detect_finish_count_
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
void WustVision::debugThread() {
    using namespace std::chrono;

    double us_interval = 1e6 / static_cast<double>(toolsgobal::debug_fps);
    auto kInterval = std::chrono::microseconds(static_cast<int64_t>(us_interval));
    while (gobal::is_inited_ && gobal::debug_mode) {
        auto start_time = steady_clock::now();

        debugvisualize(false);
        debuglog();
        writeCmdLogToJson();
        reloadConfig();
        auto elapsed = steady_clock::now() - start_time;
        if (elapsed < kInterval) {
            std::this_thread::sleep_for(kInterval - elapsed);
        }
    }
}
void WustVision::debuglog() {
    auto now = std::chrono::steady_clock::now();
    armor::Armors armors;
    armors = armors_gobal_;
    double t = std::chrono::duration<double>(now - toolsgobal::start_time_).count();
    armor::Target target = armor_target_;
    writeTargetLogToJson(target);
    {
        std::lock_guard<std::mutex> lock(toolsgobal::robot_cmd_mutex_);

        double armor_yaw = 0.0, ypd_y = 0.0, ypd_p = 0.0, armor_distance = 0.0;
        if (!armors.armors.empty()) {
            std::vector<armor::Armor> ok_armors;
            for (const auto& armor: armors.armors) {
                if (armor.number != armor::ArmorNumber::OUTPOST)
                    ok_armors.push_back(armor);
            }

            if (!ok_armors.empty()) {
                const armor::Armor& min_armor = *std::min_element(
                    ok_armors.begin(),
                    ok_armors.end(),
                    [](const armor::Armor& a, const armor::Armor& b) {
                        return a.distance_to_image_center < b.distance_to_image_center;
                    }
                );

                last_armor_ = min_armor;

                armor_distance = std::hypot(
                    min_armor.target_pos.x,
                    min_armor.target_pos.y,
                    min_armor.target_pos.z
                );

                armor_yaw = last_armor_yaw_
                    + angles::shortest_angular_distance(last_armor_yaw_, min_armor.yaw);
                last_armor_yaw_ = armor_yaw;

                ypd_y = std::atan2(min_armor.target_pos.y, min_armor.target_pos.x);
                ypd_y = last_ypd_y_ + angles::shortest_angular_distance(last_ypd_y_, ypd_y);
                last_ypd_y_ = ypd_y;

                ypd_p = std::atan2(
                    min_armor.target_pos.z,
                    std::hypot(min_armor.target_pos.x, min_armor.target_pos.y)
                );
                last_ypd_p_ = ypd_p;

                last_distance_ = armor_distance;
            }
        }

        DebugLogs& log = toolsgobal::debug_logs_;

        log.time_log.push_back(t);
        log.cmd_yaw_log.push_back(gobal::last_cmd.yaw);
        log.cmd_pitch_log.push_back(gobal::last_cmd.pitch);
        log.rune_obs_log.push_back(rune_solver_->last_observed_angle_);
        log.rune_pre_log.push_back(rune_solver_->last_pre_angle);
        log.rune_v_log.push_back(rune_solver_->curve_fitter_->getFittingParam()[0]);
        log.armor_yaw_log.push_back(armor_yaw * 180.0 / M_PI);
        log.armor_x_log.push_back(last_armor_.target_pos.x);
        log.armor_y_log.push_back(last_armor_.target_pos.y);
        log.armor_z_log.push_back(last_armor_.target_pos.z);
        log.ypd_y_log.push_back(last_ypd_y_ * 180.0 / M_PI);
        log.ypd_p_log.push_back(last_ypd_p_ * 180.0 / M_PI);
        log.armor_dis_log.push_back(last_distance_);

        // 控制长度不超过 1000
        auto trim = [](std::vector<double>& v) {
            if (v.size() > 1000)
                v.erase(v.begin());
        };

        trim(log.time_log);
        trim(log.cmd_yaw_log);
        trim(log.cmd_pitch_log);
        trim(log.rune_obs_log);
        trim(log.rune_pre_log);
        trim(log.rune_v_log);
        trim(log.armor_yaw_log);
        trim(log.armor_x_log);
        trim(log.armor_y_log);
        trim(log.armor_z_log);
        trim(log.ypd_y_log);
        trim(log.ypd_p_log);
        trim(log.armor_dis_log);
    }
}
void WustVision::debugvisualize(bool auto_fps) {
    auto now = std::chrono::steady_clock::now();
    armor::Armors armors = armors_gobal_;
    armor::Target target = armor_target_;
    std::vector<armor::OneTarget> one_targets = one_armor_targets_;
    AttackMode mode = toAttackMode(gobal::attack_mode);
    bool appear = utils::checkTargetAppear(target, one_targets);
    Tracker::State state = appear ? Tracker::TRACKING : Tracker::LOST;
    GimbalCmd gimbal_cmd = gobal::last_cmd;

    cv::Mat src;
    {
        std::lock_guard<std::mutex> lock(img_mutex_);
        src = imgframe_.img.clone();
    }

    if (mode == AttackMode::ARMOR) {
        armor::Armors armor_data = visualizeTargetProjection(target, one_targets);
        utils::transformArmorData(armor_data, T_camera_to_odom_.inverse());
        Target_info target_info;
        target_info.select_id = gimbal_cmd.select_id;

        if (!gobal::measure_tool->reprojectArmorsCorners(
                armor_data,
                target_info,
                gobal::camera_intrinsic,
                gobal::camera_distortion
            ))
            return;
        try {
            DebugArmor dbg;
            dbg.src_img = imgframe_;
            dbg.target = target;
            dbg.target_info = target_info;
            dbg.armors = armors;
            dbg.gimbal_cmd = gobal::last_cmd;
            dbg.tracker_state = state;
            drawDebugOverlayShm(dbg, auto_fps);
        } catch (const std::exception& e) {
            std::cerr << "drawDebugArmor failed: " << e.what() << '\n';
        }

    } else {
        double predict_angle = rune_solver_->last_pre_angle - rune_solver_->last_observed_angle_;

        try {
            DebugRune dbg;
            dbg.src_img = imgframe_;
            dbg.objs = rune_objects_;
            dbg.predict_angle = predict_angle;
            dbg.gimbal_cmd = gobal::last_cmd;
            dbg.manual_r_box = manual_r_box_;
            dbg.debug_text = rune_solver_->curve_fitter_->getDebugText();
            drawDebugOverlayShm(dbg, auto_fps);
        } catch (const std::exception& e) {
            std::cerr << "drawRuneAndPre failed: " << e.what() << '\n';
        }
    }
}
void WustVision::calculationManualR(const cv::Point2f center) {
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
    gobal::measure_tool->calcRTarget(
        manual_r_box_,
        T_r_,
        T_camera_to_odom_,
        gobal::camera_intrinsic,
        gobal::camera_distortion
    );
    manual_r_runing_ = false;
}
void WustVision::calculationManualR(const cv::Mat& src_img) {
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
    gobal::measure_tool->calcRTarget(
        manual_r_box_,
        T_r_,
        T_camera_to_odom_,
        gobal::camera_intrinsic,
        gobal::camera_distortion
    );
    cv::destroyWindow("Manual R Box");
    cv::destroyWindow("Manual R Box");
    cv::destroyWindow("Manual R Box");
    manual_r_runing_ = false;
}

std::vector<cv::Point2f> WustVision::clicked_points_;
void WustVision::onMouse(int event, int x, int y, int, void*) {
    if (event == cv::EVENT_LBUTTONDOWN) {
        clicked_points_.clear();
        clicked_points_.emplace_back(x, y);
        WUST_INFO("Manual R") << "Clicked Point: (" << x << ", " << y << ")";
    }
}
void WustVision::saveAutoLabelData(
    const std::vector<armor::ArmorObject>& objs,
    const CommonFrame& frame
) {
    static int save_counter = 0;

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

        save_counter++;
        if (save_counter % 10 == 0) {
            cv::Mat img_save;
            cv::cvtColor(frame.src_img, img_save, cv::COLOR_RGB2BGR);
            auto_labeler_->save(img_save, csv_data);
        }
    }
}
armor::Armors WustVision::visualizeTargetProjection(
    armor::Target armor_target_,
    std::vector<armor::OneTarget> one_armor_targets_
) {
    armor::Armors armor_data;
    armor_data.frame_id = "gimbal_odom";
    armor_data.timestamp = armor_target_.timestamp;

    if (armor_target_.tracking) {
        tf::Position pos = armor_target_.position_;
        tf::Position vel = armor_target_.velocity_;
        utils::addVelFromAccDt(vel, armor_target_.acceleration_, debug_show_dt_);
        utils::addPosFromVelDt(pos, vel, debug_show_dt_);
        if (pos.norm() > 0.5) {
            double yaw = armor_target_.yaw + armor_target_.v_yaw * debug_show_dt_;
            double r1 = armor_target_.radius_1;
            double r2 = armor_target_.radius_2;
            double d_za = armor_target_.d_za;
            double d_zc = armor_target_.d_zc;
            float xc = pos.x;
            float yc = pos.y;
            float zc = pos.z;
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
                    armor_target_.id == armor::ArmorNumber::OUTPOST ? -0.2618 : 0.2618,
                    tmp_yaw
                );

                armor_data.armors.emplace_back(armor::Armor { .type = armor_target_.type,
                                                              .pos = pos,
                                                              .ori = ori,
                                                              .is_ok = true,
                                                              .distance_to_image_center = 0.0f });
            }
        }
    }
    for (auto one_armor_target_: one_armor_targets_) {
        if (one_armor_target_.tracking) {
            tf::Position pos = one_armor_target_.position_;
            tf::Position vel = one_armor_target_.velocity_;
            utils::addVelFromAccDt(vel, one_armor_target_.acceleration_, debug_show_dt_);
            utils::addPosFromVelDt(pos, vel, debug_show_dt_);
            if (pos.norm() > 0.5) {
                double tmp_yaw = one_armor_target_.yaw + one_armor_target_.v_yaw * debug_show_dt_;
                tf::Quaternion ori;
                ori.setRPY(
                    M_PI / 2,
                    one_armor_target_.id == armor::ArmorNumber::OUTPOST ? -0.2618 : 0.2618,
                    tmp_yaw
                );

                armor_data.armors.emplace_back(armor::Armor { .type = one_armor_target_.type,
                                                              .pos = pos,
                                                              .ori = ori,
                                                              .is_ok = false,
                                                              .distance_to_image_center = 0.0f });
            }
        }
    }

    return armor_data;
}

void WustVision::reloadConfig(
) { //只有这个函数可以使用utils::tryGetValue，初始化必须保证参数完全加载
    using namespace std::chrono;
    static steady_clock::time_point last_reload_time = steady_clock::now() - seconds(2);
    static std::unordered_map<std::string, size_t> section_hashes;
    static int count = 0;

    auto now = steady_clock::now();
    if (duration_cast<seconds>(now - last_reload_time).count() < 2) {
        return;
    }
    last_reload_time = now;

    auto new_config = YAML::LoadFile(ROOT_CONFIG);
    if (!new_config) {
        std::cerr << "Failed to load config file or file empty." << std::endl;
        return;
    }

    auto compute_hash = [](const YAML::Node& node) -> size_t {
        if (!node || node.IsNull())
            return 0;
        YAML::Emitter emitter;
        emitter << node;
        std::hash<std::string> hasher;
        return hasher(emitter.c_str());
    };
    auto camera_config = new_config["camera"];
    size_t new_camera_hash = compute_hash(camera_config);
    if (new_camera_hash != section_hashes["camera"]) {
        if (camera_config && camera_ && count != 0) {
            int acquisition_frame_rate;
            utils::tryGetValue<int>(
                camera_config,
                "acquisition_frame_rate",
                acquisition_frame_rate
            );
            int exposure_time;
            utils::tryGetValue<int>(camera_config, "exposure_time", exposure_time);
            double gain;
            utils::tryGetValue<double>(camera_config, "gain", gain);
            double gamma;
            utils::tryGetValue<double>(camera_config, "gamma", gamma);
            std::string adc_bit_depth;
            utils::tryGetValue<std::string>(camera_config, "adc_bit_depth", adc_bit_depth);
            std::string pixel_format;
            utils::tryGetValue<std::string>(camera_config, "pixel_format", pixel_format);
            bool acquisition_frame_rate_enable;
            utils::tryGetValue<bool>(
                camera_config,
                "acquisition_frame_rate_enable",
                acquisition_frame_rate_enable
            );
            bool reverse_x;
            utils::tryGetValue<bool>(camera_config, "reverse_x", reverse_x);
            bool reverse_y;
            utils::tryGetValue<bool>(camera_config, "reverse_y", reverse_y);
            camera_->setParameters(
                acquisition_frame_rate,
                exposure_time,
                gain,
                gamma,
                adc_bit_depth,
                pixel_format,
                acquisition_frame_rate_enable,
                reverse_x,
                reverse_y
            );
        }
        section_hashes["camera"] = new_camera_hash;
    }

    auto shoot_config = new_config["shoot"];
    size_t new_shoot_hash = compute_hash(shoot_config);
    if (new_shoot_hash != section_hashes["shoot"]) {
        if (shoot_config && count != 0) {
            utils::tryGetValue<double>(shoot_config, "bullet_speed", gobal::velocity);
        }
        section_hashes["shoot"] = new_shoot_hash;
    }

    auto tracker_config = new_config["armor_tracker"];
    size_t new_tracker_hash = compute_hash(tracker_config);
    if (new_tracker_hash != section_hashes["armor_tracker"] && tracker_manager_) {
        if (tracker_config && tracker_manager_ && count != 0) {
            auto ekf_config = tracker_config["ekf"];
            if (ekf_config) {
                utils::tryGetValue<double>(ekf_config, "ys2qx_a", tracker_manager_->ys2qx_a_);
                utils::tryGetValue<double>(ekf_config, "ys2qy_a", tracker_manager_->ys2qy_a_);
                utils::tryGetValue<double>(ekf_config, "ys2qz_a", tracker_manager_->ys2qz_a_);
                utils::tryGetValue<double>(ekf_config, "ys2qyaw_a", tracker_manager_->ys2qyaw_a_);
                utils::tryGetValue<double>(ekf_config, "ys2qr_a", tracker_manager_->ys2qr_a_);
                utils::tryGetValue<double>(ekf_config, "ys2qd_zc_a", tracker_manager_->ys2qd_zc_a_);

                utils::tryGetValue<double>(ekf_config, "yr_y_a", tracker_manager_->yr_y_a_);
                utils::tryGetValue<double>(ekf_config, "yr_p_a", tracker_manager_->yr_p_a_);
                utils::tryGetValue<double>(
                    ekf_config,
                    "yr_d_front_a",
                    tracker_manager_->yr_d_front_a_
                );
                utils::tryGetValue<double>(
                    ekf_config,
                    "yr_d_side_a",
                    tracker_manager_->yr_d_side_a_
                );
                utils::tryGetValue<double>(
                    ekf_config,
                    "yr_yaw_front_a",
                    tracker_manager_->yr_yaw_front_a_
                );
                utils::tryGetValue<double>(
                    ekf_config,
                    "yr_yaw_side_a",
                    tracker_manager_->yr_yaw_side_a_
                );

                utils::tryGetValue<double>(ekf_config, "ys2qx_c", tracker_manager_->ys2qx_c_);
                utils::tryGetValue<double>(ekf_config, "ys2qy_c", tracker_manager_->ys2qy_c_);
                utils::tryGetValue<double>(ekf_config, "ys2qz_c", tracker_manager_->ys2qz_c_);
                utils::tryGetValue<double>(ekf_config, "ys2qyaw_c", tracker_manager_->ys2qyaw_c_);
                utils::tryGetValue<double>(ekf_config, "ys2qr_c", tracker_manager_->ys2qr_c_);
                utils::tryGetValue<double>(ekf_config, "ys2qd_zc_c", tracker_manager_->ys2qd_zc_c_);

                utils::tryGetValue<double>(ekf_config, "yr_y_c", tracker_manager_->yr_y_c_);
                utils::tryGetValue<double>(ekf_config, "yr_p_c", tracker_manager_->yr_p_c_);
                utils::tryGetValue<double>(
                    ekf_config,
                    "yr_d_front_c",
                    tracker_manager_->yr_d_front_c_
                );
                utils::tryGetValue<double>(
                    ekf_config,
                    "yr_d_side_c",
                    tracker_manager_->yr_d_side_c_
                );
                utils::tryGetValue<double>(
                    ekf_config,
                    "yr_yaw_front_c",
                    tracker_manager_->yr_yaw_front_c_
                );
                utils::tryGetValue<double>(
                    ekf_config,
                    "yr_yaw_side_c",
                    tracker_manager_->yr_yaw_side_c_
                );

                utils::tryGetValue<double>(ekf_config, "oys2qx", tracker_manager_->oys2qx_);
                utils::tryGetValue<double>(ekf_config, "oys2qy", tracker_manager_->oys2qy_);
                utils::tryGetValue<double>(ekf_config, "oys2qz", tracker_manager_->oys2qz_);
                utils::tryGetValue<double>(ekf_config, "oys2qyaw", tracker_manager_->oys2qyaw_);

                utils::tryGetValue<double>(ekf_config, "oyr_y", tracker_manager_->oyr_y_);
                utils::tryGetValue<double>(ekf_config, "oyr_p", tracker_manager_->oyr_p_);
                utils::tryGetValue<double>(
                    ekf_config,
                    "oyr_d_front",
                    tracker_manager_->oyr_d_front_
                );
                utils::tryGetValue<double>(ekf_config, "oyr_d_side", tracker_manager_->oyr_d_side_);
                utils::tryGetValue<double>(
                    ekf_config,
                    "oyr_yaw_front",
                    tracker_manager_->oyr_yaw_front_
                );
                utils::tryGetValue<double>(
                    ekf_config,
                    "oyr_yaw_side",
                    tracker_manager_->oyr_yaw_side_
                );

                utils::tryGetValue<double>(ekf_config, "r_v", tracker_manager_->r_v);
                utils::tryGetValue<double>(ekf_config, "q_v", tracker_manager_->q_v);
                utils::tryGetValue<double>(ekf_config, "q_a", tracker_manager_->q_a);
            }
        }
        section_hashes["armor_tracker"] = new_tracker_hash;
    }

    auto armor_solver_config = new_config["armor_solver"];
    size_t new_armor_solver_hash = compute_hash(armor_solver_config);
    if (new_armor_solver_hash != section_hashes["armor_solver"]) {
        if (armor_solver_config && armor_solver_ && count != 0) {
            utils::tryGetValue<double>(
                armor_solver_config,
                "small_shooting_range_w",
                armor_solver_->small_shooting_range_w_
            );
            utils::tryGetValue<double>(
                armor_solver_config,
                "small_shooting_range_h",
                armor_solver_->small_shooting_range_h_
            );
            utils::tryGetValue<double>(
                armor_solver_config,
                "big_shooting_range_w",
                armor_solver_->big_shooting_range_w_
            );
            utils::tryGetValue<double>(
                armor_solver_config,
                "big_shooting_range_h",
                armor_solver_->big_shooting_range_h_
            );
            utils::tryGetValue<double>(
                armor_solver_config,
                "max_tracking_v_yaw",
                armor_solver_->max_tracking_v_yaw_
            );
            utils::tryGetValue<double>(
                armor_solver_config,
                "prediction_delay",
                armor_solver_->prediction_delay_
            );
            utils::tryGetValue<double>(
                armor_solver_config,
                "side_angle",
                armor_solver_->side_angle_
            );
            utils::tryGetValue<double>(
                armor_solver_config,
                "min_switching_v_yaw",
                armor_solver_->min_switching_v_yaw_
            );
            utils::tryGetValue<double>(
                armor_solver_config,
                "gravity",
                armor_solver_->trajectory_compensator_->gravity_
            );
            utils::tryGetValue<double>(
                armor_solver_config,
                "resistance",
                armor_solver_->trajectory_compensator_->resistance_
            );
            utils::tryGetValue<int>(
                armor_solver_config,
                "iteration_times",
                armor_solver_->trajectory_compensator_->iteration_times_
            );
            utils::tryGetValue<double>(
                armor_solver_config,
                "oneswitch_position_thres",
                armor_solver_->oneswitch_position_thres_
            );
            utils::tryGetValue<double>(
                armor_solver_config,
                "oneswitch_angle_thres",
                armor_solver_->oneswitch_angle_thres_
            );
            utils::tryGetValue<double>(
                armor_solver_config,
                "oneswitch_hold_time",
                armor_solver_->oneswitch_hold_time_
            );
            //armor_solver_->manual_compensator_->updateMapFlow(utils::getOffsetEntry(armor_solver_config));
        }
        section_hashes["armor_solver"] = new_armor_solver_hash;
    }

    auto armor_optimize_config = new_config["armor_optimize"];
    size_t new_armor_optimize_hash = compute_hash(armor_optimize_config);
    if (new_armor_optimize_hash != section_hashes["armor_optimize"] && armor_pose_estimator_) {
        if (armor_optimize_config && armor_pose_estimator_ && count != 0) {
            utils::tryGetValue<int>(
                armor_optimize_config,
                "max_iter_R",
                armor_pose_estimator_->ba_solver_->max_iter_R_
            );
            utils::tryGetValue<int>(
                armor_optimize_config,
                "max_iter_t",
                armor_pose_estimator_->ba_solver_->max_iter_t_
            );
            utils::tryGetValue<int>(
                armor_optimize_config,
                "step_R",
                armor_pose_estimator_->ba_solver_->step_R_
            );
            utils::tryGetValue<int>(
                armor_optimize_config,
                "step_t",
                armor_pose_estimator_->ba_solver_->step_t_
            );
            utils::tryGetValue<double>(
                armor_optimize_config,
                "min_error_R",
                armor_pose_estimator_->ba_solver_->min_error_R_
            );
            utils::tryGetValue<double>(
                armor_optimize_config,
                "min_error_t",
                armor_pose_estimator_->ba_solver_->min_error_t_
            );
            utils::tryGetValue<double>(
                armor_optimize_config,
                "distance_fix_a2",
                armor_pose_estimator_->distance_fix_a2_
            );
        }
        section_hashes["armor_optimize"] = new_armor_optimize_hash;
    }

    auto rune_solver_config = new_config["rune_solver"];
    size_t new_rune_solver_hash = compute_hash(rune_solver_config);
    if (new_rune_solver_hash != section_hashes["rune_solver"] && rune_solver_) {
        if (rune_solver_config && rune_solver_ && count != 0) {
            utils::tryGetValue<double>(
                rune_solver_config,
                "gravity",
                rune_solver_->trajectory_compensator_->gravity_
            );
            utils::tryGetValue<double>(
                rune_solver_config,
                "resistance",
                rune_solver_->trajectory_compensator_->resistance_
            );
            utils::tryGetValue<int>(
                rune_solver_config,
                "iteration_times",
                rune_solver_->trajectory_compensator_->iteration_times_
            );
            //rune_solver_->manual_compensator_->updateMapFlow(utils::getOffsetEntry(rune_solver_config));
            auto ekf_config = rune_solver_config["ekf"];
            if (ekf_config) {
                utils::tryGetValue<std::vector<double>>(
                    ekf_config,
                    "q_ypdyaw",
                    rune_solver_->yq_vec_
                );
                utils::tryGetValue<std::vector<double>>(
                    ekf_config,
                    "r_ypdyaw",
                    rune_solver_->yr_vec_
                );
            }
        }
        section_hashes["rune_solver"] = new_rune_solver_hash;
    }

    gobal::config = new_config;
    count++;
}