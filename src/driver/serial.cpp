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
#include "driver/serial.hpp"
#include "common/debug/tools.hpp"
#include "common/gobal.hpp"
#include "driver/crc8_crc16.hpp"
#include "driver/packet_typedef.hpp"
#include "driver/sharetype.hpp"
#include "type/type.hpp"
#include "wust_vl/common/logger.hpp"
#include <cmath>
#include <fcntl.h>
#include <iostream>
#include <pthread.h>
#include <stdio.h>
#include <sys/mman.h>
#include <unistd.h>
namespace serial {

Serial::Serial():
    device_name_(""),
    config_(SerialPortConfig()),
    is_usb_ok_(false),
    running_(false),
    driver_() {}
Serial::Serial(const std::string& device_name, const SerialPortConfig& config):
    device_name_(device_name),
    config_(config),
    is_usb_ok_(false),
    running_(false),
    driver_() {}

Serial::~Serial() {
    stopThread();
}
void Serial::init(std::string device_name, SerialPortConfig config) {
    device_name_ = device_name;
    config_ = config;
}
void Serial::startThread(bool if_use_serial, bool if_use_nav) {
    this->running_ = true;
    if (if_use_nav) {
        shm_thread_ = std::thread(&Serial::shmTheard, this);
    }
    if (if_use_serial) {
        protect_thread_ = std::thread(&Serial::serialPortProtect, this);
        receive_thread_ = std::thread(&Serial::receiveData, this);
        send_thread_ = std::thread(&Serial::sendData, this);
    }
}

void Serial::stopThread() {
    this->running_ = false;
    if (protect_thread_.joinable()) {
        protect_thread_.join();
    }
    if (receive_thread_.joinable()) {
        receive_thread_.join();
    }
    if (send_thread_.joinable()) {
        send_thread_.join();
    }
    if (shm_thread_.joinable()) {
        shm_thread_.join();
    }
    if (driver_.is_open()) {
        driver_.close();
    }
}

void Serial::serialPortProtect() {
    WUST_INFO(serial_logger_) << "Start serialPortProtect!";

    // 1. 初始化串口
    driver_.init_port(device_name_, config_);

    // 2. 尝试第一次打开
    try {
        if (!driver_.is_open()) {
            driver_.open();
            WUST_INFO(serial_logger_) << "Serial port opened: " << device_name_;
        }
        is_usb_ok_ = true;
    } catch (const std::exception& ex) {
        WUST_ERROR(serial_logger_) << "Open serial port failed: " << ex.what();
        is_usb_ok_ = false;
    }

    // 3. 循环监测，断开重连
    while (this->running_) {
        if (!is_usb_ok_) {
            try {
                if (driver_.is_open()) {
                    driver_.close();
                    WUST_WARN(serial_logger_) << "Serial port closed for reconnect";
                }
                driver_.open();
                if (driver_.is_open()) {
                    WUST_INFO(serial_logger_) << "Serial port re-opened successfully";
                    is_usb_ok_ = true;
                }
            } catch (const std::exception& ex) {
                is_usb_ok_ = false;
                WUST_ERROR(serial_logger_) << "Re-open serial port failed: " << ex.what() << "\n";
            }
        }

        std::this_thread::sleep_for(std::chrono::milliseconds(USB_PROTECT_SLEEP_MS));
    }

    WUST_INFO(serial_logger_) << "serialPortProtect stopped";
}
void Serial::receiveData() {
    WUST_INFO(serial_logger_) << "receiveData started";

    std::vector<uint8_t> buffer(sizeof(ReceiveAimINFO));
    int retry_count = 0;

    while (this->running_) {
        if (!is_usb_ok_) {
            WUST_WARN(serial_logger_) << "receive: USB not OK! Retry count: " << retry_count++;
            std::this_thread::sleep_for(std::chrono::milliseconds(USB_NOT_OK_SLEEP_MS));
            continue;
        }
        try {
            if (!driver_.receive(buffer)) {
                continue;
            }
            // if (buffer.size() != sizeof(ReceiveAimINFO)) {
            //     WUST_WARN(serial_logger_) << "receive: buffer too small, skipping";
            //     continue;
            // }
            // std::cout << "receive: " << buffer.size() << std::endl;
            // std::cout << "package"<<sizeof(ReceiveAimINFO)<<std::endl;

            auto aim = fromVector<ReceiveAimINFO>(buffer);
            //if (verifyChecksum(aim)) {
            aimCbk(aim);
            //}

        } catch (const std::exception& ex) {
            WUST_ERROR(serial_logger_) << "receiveData exception: " << ex.what();
            is_usb_ok_ = false;
        }
    }

    WUST_INFO(serial_logger_) << "receiveData stopped";
}

void Serial::aimCbk(ReceiveAimINFO& aim_data) {
    static int last_reset_count = -1;

    if (std::isnan(aim_data.roll) || std::isnan(aim_data.pitch) || std::isnan(aim_data.yaw)
        || !this->running_)
    {
        return;
    }

    double roll = -(aim_data.roll) * M_PI / 180.0;
    double pitch = (aim_data.pitch) * M_PI / 180.0;
    double yaw = (aim_data.yaw) * M_PI / 180.0;
    double v_x = aim_data.v_x;
    double v_y = aim_data.v_y;
    double v_z = aim_data.v_z;

    auto now = std::chrono::steady_clock::now();
    try {
        auto motion_buffer = gobal::stringanything.try_get_ptr<MotionBuffer>("motion_buffer");
        if (motion_buffer) {
            motion_buffer->get()->push(yaw, pitch, roll, v_x, v_y, v_z, now);
        } else {
            WUST_ERROR(serial_logger_) << "MotionBuffer null in stringanything";
        }
    } catch (std::exception) {
        WUST_ERROR(serial_logger_) << "MotionBuffer not found in stringanything";
    }

    int manual_reset_count = aim_data.manual_reset_count;
    // if (manual_reset_count != last_reset_count) {
    //     WUST_INFO(serial_logger_) << "Manual reset count changed: " << last_reset_count << " -> "
    //                              << manual_reset_count;
    //     gobal::if_manual_reset = true;
    //     last_reset_count = manual_reset_count;
    // } else {
    //     gobal::if_manual_reset = false;
    // }

    //gobal::detect_color_ = aim_data.detect_color;
    //gobal::velocity = aim_data.bullet_speed;
    auto common_info = gobal::stringanything.get_value<CommonInfo>("common_info");
    if (common_info.debug_mode) {
        writeSerialLogToJson(aim_data);
    }
}

void Serial::sendData() {
    WUST_INFO(serial_logger_) << "Start sendData!";

    // send_robot_cmd_data_.frame_header.sof = SOF_SEND;
    send_robot_cmd_data_.cmd_ID = ID_ROBOT_CMD;
    // send_robot_cmd_data_.frame_header.len = sizeof(SendRobotCmdData) - 6;

    //  crc8::append_CRC8_check_sum(
    //      reinterpret_cast<uint8_t *>(&send_robot_cmd_data_),
    //      sizeof(HeaderFrame));

    int retry_count = 0;

    while (this->running_) {
        if (!is_usb_ok_) {
            WUST_WARN(serial_logger_) << "send: usb is not ok! Retry count:" << retry_count++;
            std::this_thread::sleep_for(std::chrono::milliseconds(USB_NOT_OK_SLEEP_MS));
            continue;
        }

        try {
            std::vector<uint8_t> send_data = toVector(send_robot_cmd_data_);
            driver_.send(send_data);
        } catch (const std::exception& ex) {
            WUST_ERROR(serial_logger_) << "Error sending data: " << ex.what();
            is_usb_ok_ = false;
        }

        double us_interval =
            1e6 / static_cast<double>(gobal::stringanything.get_value<int>("control_rate"));
        std::this_thread::sleep_for(std::chrono::microseconds(static_cast<int64_t>(us_interval)));
    }
}
void Serial::transformGimbalCmd(GimbalCmd& gimbal_cmd, bool appear) {
    auto limit = [](double val, double max_change) {
        return std::clamp(val, -max_change, max_change);
    };
    if (appear) {
        double delta_yaw = gimbal_cmd.yaw - serial_last_yaw_;
        double delta_pitch = gimbal_cmd.pitch - serial_last_pitch_;

        delta_yaw = limit(delta_yaw, max_yaw_change_);
        delta_pitch = limit(delta_pitch, max_pitch_change_);

        send_robot_cmd_data_.yaw = serial_last_yaw_ + alpha_yaw_ * delta_yaw;
        send_robot_cmd_data_.pitch = serial_last_pitch_ + alpha_pitch_ * delta_pitch;

        serial_last_yaw_ = send_robot_cmd_data_.yaw;
        serial_last_pitch_ = send_robot_cmd_data_.pitch;
        send_robot_cmd_data_.v_yaw = gimbal_cmd.v_yaw;
        send_robot_cmd_data_.v_pitch = gimbal_cmd.v_pitch;
    } else {
        auto motion_buffer = gobal::stringanything.try_get_ptr<MotionBuffer>("motion_buffer");
        if (motion_buffer) {
            auto last_att = motion_buffer->get()->get_last();
            if (last_att) {
                send_robot_cmd_data_.pitch = last_att->pitch * 180 / M_PI;
                send_robot_cmd_data_.yaw = last_att->yaw * 180 / M_PI;
            }
        }

        send_robot_cmd_data_.v_yaw = 0;
        send_robot_cmd_data_.v_pitch = 0;
    }

    send_robot_cmd_data_.distance = gimbal_cmd.distance;
    send_robot_cmd_data_.pitch_diff = gimbal_cmd.pitch_diff;
    send_robot_cmd_data_.yaw_diff = gimbal_cmd.yaw_diff;
    send_robot_cmd_data_.fire = gimbal_cmd.fire_advice;
    send_robot_cmd_data_.detect_color = gobal::stringanything.get_value<int>("detect_color");
    send_robot_cmd_data_.appear = appear;
    auto now = std::chrono::steady_clock::now();
    auto duration = now.time_since_epoch();
    uint64_t millis = std::chrono::duration_cast<std::chrono::milliseconds>(duration).count();
    send_robot_cmd_data_.time_stamp = static_cast<uint32_t>(millis);
}
void Serial::shmTheard() {
    while (!gobal::is_inited_) {
        usleep(10000); // 每10ms检查一次，避免占用 CPU
    }

    const char* SHM_NAME = "/cmd_vel";
    const size_t SHM_SIZE = sizeof(TwistData);

    int shm_fd = shm_open(SHM_NAME, O_RDONLY, 0666);
    if (shm_fd == -1) {
        perror("shm_open");
        WUST_ERROR(serial_logger_) << "Error opening shared memory";
        return;
    }

    void* ptr = mmap(0, SHM_SIZE, PROT_READ, MAP_SHARED, shm_fd, 0);
    if (ptr == MAP_FAILED) {
        perror("mmap");
        WUST_ERROR(serial_logger_) << "Error mapping shared memory";
        return;
    }

    TwistData* data = static_cast<TwistData*>(ptr);

    while (gobal::is_inited_) {
        usleep(50000); // 50ms
    }

    WUST_INFO(serial_logger_) << "shmTheard end";
    return;
}
} // namespace serial