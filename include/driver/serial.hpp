// Copyright 2025 XiaoJian Wu
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
#pragma once
#include "common/gobal.hpp"
#include "driver/packet_typedef.hpp"
#include "wust_vl/common/drivers/serial_driver.hpp"
#include <memory>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

namespace serial {
static constexpr int USB_PROTECT_SLEEP_MS = 1000;
static constexpr int USB_NOT_OK_SLEEP_MS = 1000;
class Serial {
public:
    Serial();
    Serial(const std::string& device_name, const SerialPortConfig& config);
    ~Serial();
    void init(std::string device_name, SerialPortConfig config);
    void serialPortProtect();
    void startThread(bool if_use_serial, bool if_use_nav);
    void stopThread();
    void receiveData();
    void aimCbk(ReceiveAimINFO& aim_data);
    bool usbOk() const {
        return is_usb_ok_;
    }
    void sendData();
    void transformGimbalCmd(GimbalCmd& gimbal_cmd, bool appear);
    uint16_t calculateChecksum(const ReceiveAimINFO& info) {
        size_t length = sizeof(ReceiveAimINFO) - sizeof(info.checksum);
        const uint8_t* data = reinterpret_cast<const uint8_t*>(&info);

        uint32_t sum = 0;
        for (size_t i = 0; i < length; ++i) {
            sum += data[i];
        }
        return static_cast<uint16_t>(sum % 256);
    }
    bool verifyChecksum(const ReceiveAimINFO& info) {
        // printf("calchecksum: %x",calculateChecksum(info));
        // printf("checksum: %x",info.checksum);
        return info.checksum == calculateChecksum(info);
    }

    void shmTheard();
    double serial_last_yaw_;
    double serial_last_pitch_;
    std::string device_name_;
    SerialPortConfig config_;
    std::atomic<bool> is_usb_ok_;
    std::atomic<bool> running_;
    std::thread protect_thread_;
    std::thread receive_thread_;
    std::thread send_thread_;
    std::thread shm_thread_;
    SerialDriver driver_;
    std::string serial_logger_ = "serial";
    SendRobotCmdData send_robot_cmd_data_;
    double alpha_yaw_;
    double alpha_pitch_;
    double max_yaw_change_;
    double max_pitch_change_;
};
} // namespace serial