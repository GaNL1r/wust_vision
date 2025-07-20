// Copyright 2025 SMBU-PolarBear-Robotics-Team
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

#include <algorithm>
#include <cstdint>
#include <vector>

const uint8_t SOF_RECEIVE = 0x5A;
const uint8_t SOF_SEND = 0x5A;

// Receive
// const uint8_t ID_DEBUG = 0x01;
const uint8_t ID_IMU = 0x01;
const uint8_t ID_AIM_INFO = 0X02;

// Send
const uint8_t ID_ROBOT_CMD = 0x01;

struct HeaderFrame {
    uint8_t sof; // 数据帧起始字节，固定值为 0x5A
    uint8_t len; // 数据段长度
    uint8_t id; // 数据段id
    uint8_t crc; // 数据帧头的 CRC8 校验
} __attribute__((packed));

struct ReceiveAimINFO {
    uint8_t cmd_ID; //命令码
    uint32_t time_stamp; //时间戳

    float yaw; // rad
    float pitch; // rad
    float roll; // rad

    float yaw_vel; // rad/s
    float pitch_vel; // rad/s
    float roll_vel; // rad/s

    float v_x;
    float v_y;
    float v_z;

    float bullet_speed; // m/s
    float controller_delay; // s

    uint8_t manual_reset_count;
    uint8_t detect_color; // 0 red 1 blue
    uint16_t checksum;

} __attribute__((packed));
// IMU 数据包
struct ReceiveImuData {
    HeaderFrame frame_header;
    uint32_t time_stamp;

    struct {
        float yaw; // rad
        float pitch; // rad
        float roll; // rad

        float yaw_vel; // rad/s
        float pitch_vel; // rad/s
        float roll_vel; // rad/s

        // float x_accel;  // m/s^2
        // float y_accel;  // m/s^2
        // float z_accel;  // m/s^2
    } __attribute__((packed)) data;

    uint16_t crc;
} __attribute__((packed));

struct SendRobotCmdData {
    uint8_t cmd_ID; //命令码
    uint32_t time_stamp;

    float pitch;
    float yaw;
    float distance;

    uint8_t appear;
    uint8_t fire;
    float yaw_diff;
    float pitch_diff;
    float v_yaw;
    float v_pitch;

    uint8_t detect_color; // 0 red 1 blue
} __attribute__((packed));

template<typename T>
inline T fromVector(const std::vector<uint8_t>& data) {
    T packet;
    std::copy(data.begin(), data.end(), reinterpret_cast<uint8_t*>(&packet));
    return packet;
}

template<typename T>
inline std::vector<uint8_t> toVector(const T& data) {
    std::vector<uint8_t> packet(sizeof(T));
    std::copy(
        reinterpret_cast<const uint8_t*>(&data),
        reinterpret_cast<const uint8_t*>(&data) + sizeof(T),
        packet.begin()
    );
    return packet;
}