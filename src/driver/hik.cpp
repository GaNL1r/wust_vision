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
#include "driver/hik.hpp"
#include "common/gobal.hpp"
#include "common/logger.hpp"
#include <Eigen/Dense>
#include <algorithm>
#include <chrono>
#include <filesystem>
#include <iostream>
#include <opencv2/highgui.hpp>
#include <pwd.h>
#include <regex>
#include <stdexcept>
#include <unistd.h>

HikCamera::HikCamera(): camera_handle_(nullptr), fail_count_(0) {}

HikCamera::~HikCamera() {
    stopCamera();
    if (recorder_ != nullptr) {
        recorder_->stop();
        WUST_INFO(hik_logger) << "Recorder stopped! Video file " << recorder_->path.string()
                              << " has been saved";
    }
    if (capture_thread_.joinable()) {
        capture_thread_.join();
    }
    if (camera_handle_) {
        MV_CC_StopGrabbing(camera_handle_);
        MV_CC_CloseDevice(camera_handle_);
        MV_CC_DestroyHandle(&camera_handle_);
    }

    WUST_INFO(hik_logger) << "Camera destroyed!";
}

// 初始化相机：枚举设备、创建句柄、打开设备、获取图像信息等
bool HikCamera::initializeCamera(const std::string& target_sn) {
    last_target_sn_ = target_sn;

    while (true) {
        MV_CC_DEVICE_INFO_LIST device_list = { 0 };

        int n_ret = MV_CC_EnumDevices(MV_USB_DEVICE, &device_list);
        if (n_ret != MV_OK) {
            WUST_ERROR(hik_logger) << "MV_CC_EnumDevices failed, error code: " << n_ret;
            std::this_thread::sleep_for(std::chrono::seconds(1));
            continue;
        }

        if (device_list.nDeviceNum == 0) {
            WUST_ERROR(hik_logger) << "No USB cameras found";
            std::this_thread::sleep_for(std::chrono::seconds(1));
            continue;
        }

        WUST_INFO(hik_logger) << "Found " << device_list.nDeviceNum << " USB camera(s):";
        for (unsigned int i = 0; i < device_list.nDeviceNum; ++i) {
            auto info = device_list.pDeviceInfo[i];
            const char* sn =
                reinterpret_cast<const char*>(info->SpecialInfo.stUsb3VInfo.chSerialNumber);
            WUST_INFO(hik_logger) << "  [" << i << "] SN = " << sn;
        }

        int sel = -1;
        for (unsigned int i = 0; i < device_list.nDeviceNum; ++i) {
            auto info = device_list.pDeviceInfo[i];
            const char* sn =
                reinterpret_cast<const char*>(info->SpecialInfo.stUsb3VInfo.chSerialNumber);
            if (target_sn == sn) {
                sel = i;
                break;
            }
        }

        if (sel < 0) {
            WUST_ERROR(hik_logger) << "Camera with serial " << target_sn << " not found";
            std::this_thread::sleep_for(std::chrono::seconds(1));
            continue;
        }

        WUST_INFO(hik_logger) << "Selecting camera at index " << sel << " (SN=" << target_sn << ")";

        n_ret = MV_CC_CreateHandle(&camera_handle_, device_list.pDeviceInfo[sel]);
        if (n_ret != MV_OK) {
            WUST_ERROR(hik_logger) << "MV_CC_CreateHandle failed: " << n_ret;
            std::this_thread::sleep_for(std::chrono::seconds(1));
            continue;
        }

        n_ret = MV_CC_OpenDevice(camera_handle_);
        if (n_ret != MV_OK) {
            WUST_ERROR(hik_logger) << "MV_CC_OpenDevice failed: " << n_ret;
            MV_CC_DestroyHandle(camera_handle_);
            std::this_thread::sleep_for(std::chrono::seconds(1));
            continue;
        }

        n_ret = MV_CC_GetImageInfo(camera_handle_, &img_info_);
        if (n_ret != MV_OK) {
            WUST_ERROR(hik_logger) << "MV_CC_GetImageInfo failed: " << n_ret;
            MV_CC_CloseDevice(camera_handle_);
            MV_CC_DestroyHandle(camera_handle_);
            std::this_thread::sleep_for(std::chrono::seconds(1));
            continue;
        }

        convert_param_.nWidth = img_info_.nWidthValue;
        convert_param_.nHeight = img_info_.nHeightValue;
        convert_param_.enDstPixelType = PixelType_Gvsp_RGB8_Packed;

        disableTrigger();

        WUST_INFO(hik_logger) << "Camera initialized successfully";
        return true;
    }
}

bool HikCamera::enableTrigger(TriggerType type, const std::string& source, int64_t activation) {
    trigger_type_ = type;
    trigger_source_ = source;
    trigger_activation_ = activation;

    if (MV_CC_SetEnumValueByString(camera_handle_, "TriggerMode", "On") != MV_OK)
        return false;
    if (MV_CC_SetEnumValueByString(camera_handle_, "TriggerSource", source.c_str()) != MV_OK)
        return false;

    const char* act = (activation == 1 ? "RisingEdge" : "FallingEdge");
    if (MV_CC_SetEnumValueByString(camera_handle_, "TriggerActivation", act) != MV_OK)
        return false;

    WUST_INFO(hik_logger) << "Trigger enabled: src=" << source << ", act=" << act;
    return true;
}

void HikCamera::disableTrigger() {
    trigger_type_ = TriggerType::None;
    MV_CC_SetEnumValueByString(camera_handle_, "TriggerMode", "Off");
    WUST_INFO(hik_logger) << "Trigger disabled, continuous mode.";
}
// 设置相机参数：帧率、曝光、增益、ADC位深及像素格式（这里硬编码参数，可按需修改）
void HikCamera::setParameters(
    double acquisition_frame_rate,
    double exposure_time,
    double gain,
    double gamma,
    const std::string& adc_bit_depth,
    const std::string& pixel_format,
    bool acquisitionFrameRateEnable,
    bool reverse_x,
    bool reverse_y
) {
    MVCC_FLOATVALUE f_value;
    // 设置像素格式
    int status = MV_CC_SetEnumValueByString(camera_handle_, "PixelFormat", pixel_format.c_str());
    if (status == MV_OK) {
        WUST_INFO(hik_logger) << "Pixel Format set to " << pixel_format;
    } else {
        WUST_ERROR(hik_logger) << "Failed to set Pixel Format, status = " << status;
    }

    // 设置 ADC 位深
    status = MV_CC_SetEnumValueByString(camera_handle_, "ADCBitDepth", adc_bit_depth.c_str());
    if (status == MV_OK) {
        WUST_INFO(hik_logger) << "ADC Bit Depth set to " << adc_bit_depth;
    } else {
        WUST_ERROR(hik_logger) << "Failed to set ADC Bit Depth, status = " << status;
    }

    // 设置采集帧率
    MV_CC_SetBoolValue(camera_handle_, "AcquisitionFrameRateEnable", true);
    MV_CC_SetFloatValue(camera_handle_, "AcquisitionFrameRate", acquisition_frame_rate);
    WUST_INFO(hik_logger) << "Acquisition frame rate: " << acquisition_frame_rate;

    // 设置曝光时间（单位：微秒）
    MV_CC_SetFloatValue(camera_handle_, "ExposureTime", exposure_time);
    WUST_INFO(hik_logger) << "Exposure time: " << exposure_time;

    // 设置增益
    if (int ret = MV_CC_GetFloatValue(camera_handle_, "Gain", &f_value); ret == MV_OK) {
        double clamped =
            std::clamp(gain, static_cast<double>(f_value.fMin), static_cast<double>(f_value.fMax));
        MV_CC_SetFloatValue(camera_handle_, "Gain", clamped);
        WUST_INFO(hik_logger) << "Gain: " << clamped;
    } else {
        WUST_ERROR(hik_logger) << "Failed to set Gain, status = " << ret;
    }

    int ret = MV_CC_SetBoolValue(camera_handle_, "GammaEnable", true);

    if (ret == MV_OK) {
        WUST_INFO(hik_logger) << "Set GammaEnable success";
    } else {
        WUST_ERROR(hik_logger) << "Failed to set GammaEnable, status = " << ret;
    }
    // 设置gamma
    if (int ret = MV_CC_GetFloatValue(camera_handle_, "Gamma", &f_value); ret == MV_OK) {
        double clamped =
            std::clamp(gamma, static_cast<double>(f_value.fMin), static_cast<double>(f_value.fMax));
        MV_CC_SetFloatValue(camera_handle_, "Gamma", clamped);
        WUST_INFO(hik_logger) << "Set Gamma to " << clamped;
    } else {
        WUST_ERROR(hik_logger) << "Failed to set Gamma, status = " << ret;
    }

    if (!acquisitionFrameRateEnable) {
        MV_CC_SetBoolValue(camera_handle_, "AcquisitionFrameRateEnable", false);
    }
    MV_CC_SetBoolValue(camera_handle_, "ReverseX", reverse_x);
    WUST_INFO(hik_logger) << "ReverseX set to " << reverse_x;

    MV_CC_SetBoolValue(camera_handle_, "ReverseY", reverse_y);
    WUST_INFO(hik_logger) << "ReverseY set to " << reverse_y;
    last_frame_rate_ = acquisition_frame_rate;
    last_exposure_time_ = exposure_time;
    last_gain_ = gain;
    last_gamma_ = gamma;
    last_adc_bit_depth_ = adc_bit_depth;
    last_pixel_format_ = pixel_format;
    last_acquisitionFrameRateEnable = acquisitionFrameRateEnable;
    last_reverse_x_ = reverse_x;
    last_reverse_y_ = reverse_y;
    WUST_INFO(hik_logger) << "Camera parameters set successfully!";
}

// 启动图像采集，采集线程不断获取图像帧并推入队列
void HikCamera::startCamera(bool if_recorder) {
    int n_ret = MV_CC_StartGrabbing(camera_handle_);
    if (n_ret != MV_OK) {
        WUST_ERROR(hik_logger) << "Failed to start camera grabbing!";
    }
    MVCC_INTVALUE stParam = { 0 };
    if (MV_CC_GetIntValue(camera_handle_, "Width", &stParam) == MV_OK) {
        expected_width_ = stParam.nCurValue;
    }
    if (MV_CC_GetIntValue(camera_handle_, "Height", &stParam) == MV_OK) {
        expected_height_ = stParam.nCurValue;
    }
    if (trigger_type_ != TriggerType::Software) {
        capture_thread_ = std::thread(&HikCamera::hikCaptureLoop, this);
    }

    if (if_recorder) {
        const char* home = nullptr;

        // 尝试从 SUDO_USER 获取真实用户 home
        const char* sudo_user = std::getenv("SUDO_USER");
        if (sudo_user) {
            struct passwd* pw = getpwnam(sudo_user);
            if (pw) {
                home = pw->pw_dir;
            }
        }

        // 如果不是 sudo，使用 getuid 获取 home
        if (!home) {
            struct passwd* pw = getpwuid(getuid());
            if (pw) {
                home = pw->pw_dir;
            }
        }

        if (!home) {
            throw std::runtime_error("HOME environment variable not set.");
        }

        namespace fs = std::filesystem;
        std::filesystem::path video_path_ = fs::path(home) / "wust_data/video/"
            / std::string(std::to_string(std::time(nullptr)) + ".avi");

        recorder_ = std::make_unique<Recorder>(
            video_path_,
            last_frame_rate_,
            cv::Size(expected_width_, expected_height_)
        );
        recorder_->start();
    }
}

bool HikCamera::restartCamera() {
    WUST_WARN(hik_logger) << "Restarting camera from scratch...";

    MV_CC_StopGrabbing(camera_handle_);
    MV_CC_CloseDevice(camera_handle_);
    MV_CC_DestroyHandle(&camera_handle_);
    camera_handle_ = nullptr;

    std::this_thread::sleep_for(std::chrono::seconds(1));

    if (!initializeCamera(last_target_sn_)) {
        WUST_ERROR(hik_logger) << "Failed to re-initialize camera.";
        return false;
    }

    setParameters(
        last_frame_rate_,
        last_exposure_time_,
        last_gain_,
        last_gamma_,
        last_adc_bit_depth_,
        last_pixel_format_,
        last_acquisitionFrameRateEnable,
        last_reverse_x_,
        last_reverse_y_
    );

    int n_ret = MV_CC_StartGrabbing(camera_handle_);
    if (n_ret != MV_OK) {
        WUST_ERROR(hik_logger) << "Failed to start grabbing after restart.";
        return false;
    }

    WUST_INFO(hik_logger) << "Camera restarted successfully!";
    return true;
}

void HikCamera::hikCaptureLoop() {
    MV_FRAME_OUT out_frame;
    WUST_INFO(hik_logger) << "Starting image capture loop!";

    auto fail_start_time = std::chrono::steady_clock::now();
    bool in_fail_state = false;

    in_low_frame_rate_state_ = false;
    auto last_frame_rate_check = std::chrono::steady_clock::now();
    int frame_counter = 0;

    try {
        while (!stop_signal_) {
            int n_ret = MV_CC_GetImageBuffer(camera_handle_, &out_frame, 1);
            if (n_ret == MV_OK) {
                in_fail_state = false;
                ++frame_counter;

                ImageFrame frame;
                frame.width = out_frame.stFrameInfo.nWidth;
                frame.height = out_frame.stFrameInfo.nHeight;
                frame.step = frame.width * 3;

                frame.data.resize(frame.width * frame.height * 3);

                convert_param_.pDstBuffer = frame.data.data();
                convert_param_.nDstBufferSize = static_cast<int>(frame.data.size());
                convert_param_.pSrcData = out_frame.pBufAddr;
                convert_param_.nSrcDataLen = out_frame.stFrameInfo.nFrameLen;
                convert_param_.enSrcPixelType = out_frame.stFrameInfo.enPixelType;

                MV_CC_ConvertPixelType(camera_handle_, &convert_param_);
                auto current_time = std::chrono::steady_clock::now();
                frame.timestamp = current_time;

                if (on_frame_callback_) {
                    std::chrono::microseconds delay =
                        std::chrono::microseconds(static_cast<int64_t>(
                            std::round((last_exposure_time_ / 2.0) - gobal::communication_delay_μs)
                        ));
                    auto t_query = std::chrono::steady_clock::now() - delay;
                    auto past_att = gobal::attitude_buffer.get_interpolated(t_query);
                    if (past_att) {
                        double yaw = past_att->yaw;
                        double pitch = past_att->pitch;
                        double roll = past_att->roll;
                        Eigen::Vector3d v(past_att->vx, past_att->vy, past_att->vz);
                        Eigen::Matrix3d R_gimbal2odom;
                        R_gimbal2odom = Eigen::AngleAxisd(
                                            yaw + gobal::gimbal2camera_yaw,
                                            Eigen::Vector3d::UnitZ()
                                        )
                            * Eigen::AngleAxisd(
                                            -pitch - gobal::gimbal2camera_pitch,
                                            Eigen::Vector3d::UnitY()
                            )
                            * Eigen::AngleAxisd(
                                            roll + gobal::gimbal2camera_roll,
                                            Eigen::Vector3d::UnitX()
                            );
                        on_frame_callback_(frame, R_gimbal2odom, v);

                    } else {
                        Eigen::Matrix3d R_gimbal2odom;
                        R_gimbal2odom = Eigen::AngleAxisd(
                                            gobal::last_yaw + gobal::gimbal2camera_yaw,
                                            Eigen::Vector3d::UnitZ()
                                        )
                            * Eigen::AngleAxisd(
                                            -gobal::last_pitch - gobal::gimbal2camera_pitch,
                                            Eigen::Vector3d::UnitY()
                            )
                            * Eigen::AngleAxisd(
                                            gobal::last_roll + gobal::gimbal2camera_roll,
                                            Eigen::Vector3d::UnitX()
                            );
                        Eigen::Vector3d v(gobal::last_v_x, gobal::last_v_y, gobal::last_v_z);
                        on_frame_callback_(frame, R_gimbal2odom, v);
                    }
                }
                if (recorder_ != nullptr) {
                    recorder_->addFrame(frame.data);
                }

                MV_CC_FreeImageBuffer(camera_handle_, &out_frame);
                continue;
                if (std::chrono::duration_cast<std::chrono::seconds>(
                        current_time - last_frame_rate_check
                    )
                        .count()
                    >= 1)
                {
                    float actual_fps = static_cast<float>(frame_counter);
                    frame_counter = 0;
                    last_frame_rate_check = current_time;

                    if (actual_fps < last_frame_rate_ * 0.5f) {
                        if (!in_low_frame_rate_state_) {
                            low_frame_rate_start_time_ = current_time;
                            in_low_frame_rate_state_ = true;
                            WUST_WARN(hik_logger) << "Low FPS detected: " << actual_fps;
                        } else if (std::chrono::duration_cast<std::chrono::seconds>(
                           current_time - low_frame_rate_start_time_)
                           .count() >= 5)
                        {
                            WUST_ERROR(hik_logger)
                                << "Low FPS persisted for 5s. Restarting camera...";
                            if (restartCamera()) {
                                in_low_frame_rate_state_ = false;
                                WUST_INFO(hik_logger) << "Camera restarted successfully";
                            } else {
                                WUST_ERROR(hik_logger) << "Restart failed, exiting capture loop.";
                                break;
                            }
                        }
                    } else if (in_low_frame_rate_state_) {
                        in_low_frame_rate_state_ = false;
                        WUST_INFO(hik_logger) << "FPS recovered to normal: " << actual_fps;
                    }
                }

            } else {
                if (!in_fail_state) {
                    fail_start_time = std::chrono::steady_clock::now();
                    in_fail_state = true;
                }

                if (std::chrono::duration_cast<std::chrono::seconds>(
                        std::chrono::steady_clock::now() - fail_start_time
                    )
                        .count()
                    > 5)
                {
                    if (!restartCamera()) {
                        WUST_ERROR(hik_logger)
                            << "Failed to restart camera after hardware failure.";
                        gobal::exit_flag = true;
                        break;
                    }
                    fail_start_time = std::chrono::steady_clock::now();
                    in_fail_state = false;
                }
            }
        }
    } catch (const std::exception& e) {
        WUST_ERROR(hik_logger) << "Exception in capture loop: " << e.what();
        stop_signal_ = true;
    } catch (...) {
        WUST_ERROR(hik_logger) << "Unknown exception in capture loop!";
        stop_signal_ = true;
    }
    WUST_INFO(hik_logger) << "Exiting image capture loop.";
}

void HikCamera::stopCamera() {
    stop_signal_ = true;
}
bool HikCamera::read() {
    if (trigger_type_ == TriggerType::None) {
        WUST_WARN(hik_logger) << "read() called in non-trigger mode. Ignored.";
        return false;
    }

    if (trigger_type_ == TriggerType::Software) {
        MV_CC_SetCommandValue(camera_handle_, "TriggerSoftware");
    }

    MV_FRAME_OUT out_frame;
    int n_ret = MV_CC_GetImageBuffer(camera_handle_, &out_frame, 1000); // 1s timeout
    if (n_ret != MV_OK) {
        WUST_ERROR(hik_logger) << "Failed to get image buffer in read()";
        return false;
    }

    ImageFrame frame;
    frame.width = out_frame.stFrameInfo.nWidth;
    frame.height = out_frame.stFrameInfo.nHeight;
    frame.step = frame.width * 3;
    frame.data.resize(frame.width * frame.height * 3);

    convert_param_.pDstBuffer = frame.data.data();
    convert_param_.nDstBufferSize = static_cast<int>(frame.data.size());
    convert_param_.pSrcData = out_frame.pBufAddr;
    convert_param_.nSrcDataLen = out_frame.stFrameInfo.nFrameLen;
    convert_param_.enSrcPixelType = out_frame.stFrameInfo.enPixelType;

    MV_CC_ConvertPixelType(camera_handle_, &convert_param_);

    frame.timestamp = std::chrono::steady_clock::now();

    if (on_frame_callback_) {
        std::chrono::microseconds delay = std::chrono::microseconds(static_cast<int64_t>(
            std::round((last_exposure_time_ / 2.0) - gobal::communication_delay_μs)
        ));
        auto t_query = std::chrono::steady_clock::now() - delay;
        auto past_att = gobal::attitude_buffer.get_interpolated(t_query);
        if (past_att) {
            double yaw = past_att->yaw;
            double pitch = past_att->pitch;
            double roll = past_att->roll;
            Eigen::Vector3d v(past_att->vx, past_att->vy, past_att->vz);
            Eigen::Matrix3d R_gimbal2odom;
            R_gimbal2odom =
                Eigen::AngleAxisd(yaw + gobal::gimbal2camera_yaw, Eigen::Vector3d::UnitZ())
                * Eigen::AngleAxisd(-pitch - gobal::gimbal2camera_pitch, Eigen::Vector3d::UnitY())
                * Eigen::AngleAxisd(roll + gobal::gimbal2camera_roll, Eigen::Vector3d::UnitX());
            on_frame_callback_(frame, R_gimbal2odom, v);

        } else {
            Eigen::Matrix3d R_gimbal2odom;
            R_gimbal2odom = Eigen::AngleAxisd(
                                gobal::last_yaw + gobal::gimbal2camera_yaw,
                                Eigen::Vector3d::UnitZ()
                            )
                * Eigen::AngleAxisd(
                                -gobal::last_pitch - gobal::gimbal2camera_pitch,
                                Eigen::Vector3d::UnitY()
                )
                * Eigen::AngleAxisd(
                                gobal::last_roll + gobal::gimbal2camera_roll,
                                Eigen::Vector3d::UnitX()
                );
            Eigen::Vector3d v(gobal::last_v_x, gobal::last_v_y, gobal::last_v_z);
            on_frame_callback_(frame, R_gimbal2odom, v);
        }

        if (recorder_ != nullptr) {
            recorder_->addFrame(frame.data);
        }

        MV_CC_FreeImageBuffer(camera_handle_, &out_frame);
        return true;
    }
    return false;
}
ImageFrame HikCamera::readImage() {
    if (trigger_type_ == TriggerType::None) {
        WUST_WARN(hik_logger) << "read() called in non-trigger mode. Ignored.";
        return ImageFrame();
    }

    if (trigger_type_ == TriggerType::Software) {
        MV_CC_SetCommandValue(camera_handle_, "TriggerSoftware");
    }

    MV_FRAME_OUT out_frame;
    int n_ret = MV_CC_GetImageBuffer(camera_handle_, &out_frame, 1000); // 1s timeout
    if (n_ret != MV_OK) {
        WUST_ERROR(hik_logger) << "Failed to get image buffer in read()";
        return ImageFrame();
    }

    ImageFrame frame;
    frame.width = out_frame.stFrameInfo.nWidth;
    frame.height = out_frame.stFrameInfo.nHeight;
    frame.step = frame.width * 3;
    frame.data.resize(frame.width * frame.height * 3);

    convert_param_.pDstBuffer = frame.data.data();
    convert_param_.nDstBufferSize = static_cast<int>(frame.data.size());
    convert_param_.pSrcData = out_frame.pBufAddr;
    convert_param_.nSrcDataLen = out_frame.stFrameInfo.nFrameLen;
    convert_param_.enSrcPixelType = out_frame.stFrameInfo.enPixelType;

    MV_CC_ConvertPixelType(camera_handle_, &convert_param_);

    frame.timestamp = std::chrono::steady_clock::now();

    MV_CC_FreeImageBuffer(camera_handle_, &out_frame);
    return frame;
}