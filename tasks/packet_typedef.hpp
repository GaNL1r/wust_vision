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
#include <cstring>
#include <stdexcept>
#include <vector>

const uint8_t SOF_RECEIVE = 0x5A;
const uint8_t SOF_SEND = 0x5A;

// Receive
// const uint8_t ID_DEBUG = 0x01;
const uint8_t ID_IMU = 0x01;

// Send
const uint8_t ID_ROBOT_CMD = 0x01;
const uint8_t ID_NAV_CMD = 0x02;
struct HeaderFrame {
    uint8_t sof; // 数据帧起始字节，固定值为 0x5A
    uint8_t len; // 数据段长度
    uint8_t id; // 数据段id
    uint8_t crc; // 数据帧头的 CRC8 校验
} __attribute__((packed));
const uint8_t ID_AIM_INFO = 0X02;
const uint8_t ID_REFEREE_INFO = 0X03;
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

} __attribute__((packed));
struct ReceiveReferee //rmul2024
{
    uint8_t cmd_ID; //命令码
    uint32_t time_stamp;
    int game_time; //gamestate 到4 从0开始，其他状态发-1
    uint8_t my_team;
    int max_health;
    int cur_health;
    int cur_bullet;
    int r1_health;
    int r3_health;
    int r7_health;
    int b1_health;
    int b3_health;
    int b7_health;

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

    } __attribute__((packed)) data;

    uint16_t crc;
} __attribute__((packed));

struct SendRobotCmdData {
    uint8_t cmd_ID; //命令码
    uint32_t time_stamp;
    uint8_t appear;
    float pitch; //最佳控制yaw pitch
    float yaw;

    float target_yaw; //最佳击中yaw pitch
    float target_pitch;

    float enable_yaw_diff;
    float
        enable_pitch_diff; //计算当前yaw pitch 与本包次发送target_yaw target_pitch的差绝对值小于enable（角度）开火
    float v_yaw;
    float v_pitch;
    uint8_t detect_color; // 0 red 1 blue
} __attribute__((packed));

// struct SendRobotCmdData {
//     uint8_t cmd_ID; //命令码
//     uint32_t time_stamp;

//     float pitch;
//     float yaw;
//     float distance;

//     uint8_t appear;
//     uint8_t fire;

//     float yaw_diff;
//     float pitch_diff;

//     float v_yaw;
//     float v_pitch;

//     uint8_t detect_color; // 0 red 1 blue

//     float enable_yaw_diff;
//     float
//         enable_pitch_diff; //计算当前yaw pitch 与本包次发送yaw pitch的差绝对值小于enable（角度） 且 fire ==true ，开火
// } __attribute__((packed));
struct NavRobotCmdData {
    uint8_t cmd_ID; //命令码
    uint32_t time_stamp;
    uint8_t check;
    float vx;
    float vy;
    float wz;

} __attribute__((packed));
