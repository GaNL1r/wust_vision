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

#include <cstdint>
#include <fstream>
#include <nlohmann/json.hpp>
#include <sys/types.h>
namespace wust_vision {
constexpr uint8_t ID_ROBOT_CMD = 0x01;
constexpr uint8_t ID_NAV_CMD = 0x02;

constexpr uint8_t ID_AIM_INFO = 0X02;
constexpr uint8_t ID_REFEREE_INFO = 0X03;
constexpr const char* TARGET_TOPIC = "vision_target";
constexpr const char* NAV_STATE_TOPIC = "rose_state";
constexpr const char* MODE_TOPIC = "sentry_mode";
constexpr const char* ROBO_STATE_TOPIC = "robo_state";
constexpr const char* GOAL_TOPIC = "rose_goal";
struct ReceiveAimINFO {
    uint8_t cmd_ID; //命令码
    uint32_t time_stamp; //时间戳

    float yaw;
    float pitch;
    float roll;
    float ins_sum;
    float yaw_vel;
    float pitch_vel;
    float roll_vel;

    float v_x;
    float v_y;
    float v_z;

    float bullet_speed;
    float controller_delay;

    uint8_t manual_reset_count;
    uint8_t detect_color; // 0 red 1 blue

} __attribute__((packed));
inline void writeSerialLogToJson(const ReceiveAimINFO& aim) {
    nlohmann::json j;

    j["timestamp"] = aim.time_stamp;
    j["yaw"] = aim.yaw;
    j["pitch"] = aim.pitch;
    j["roll"] = aim.roll;

    j["yaw_vel"] = aim.yaw_vel;
    j["pitch_vel"] = aim.pitch_vel;
    j["roll_vel"] = aim.roll_vel;
    j["v_x"] = aim.v_x;
    j["v_y"] = aim.v_y;
    j["v_z"] = aim.v_z;
    j["manual_reset_count"] = aim.manual_reset_count;
    j["bullet_speed"] = aim.bullet_speed;
    j["controller_delay"] = aim.controller_delay;
    j["detect_color"] = (aim.detect_color == 0 ? "Red" : "Blue");

    // FPS 统计
    static int frame_count = 0;
    static double fps = 0.0;
    static auto last_time = std::chrono::steady_clock::now();

    ++frame_count;
    const auto now = std::chrono::steady_clock::now();
    const double elapsed = std::chrono::duration<double>(now - last_time).count();
    if (elapsed >= 1.0) {
        fps = frame_count / elapsed;
        frame_count = 0;
        last_time = now;
    }
    j["fps"] = fps;

    std::ofstream file("/dev/shm/serial_log.json");
    if (file.is_open()) {
        file << j.dump(2);
    }
}
struct SendRobotCmdData {
    uint8_t cmd_ID; //命令码
    uint32_t time_stamp;
    uint8_t appear;
    uint8_t shoot_rate = 3;
    float pitch; //最佳控制yaw pitch
    float yaw;

    float target_yaw; //最佳击中yaw pitch
    float target_pitch;

    float enable_yaw_diff;
    float
        enable_pitch_diff; //计算当前yaw pitch 与本包次发送target_yaw target_pitch的差绝对值小于enable（角度）开火
    float v_yaw;
    float v_pitch;
    float a_yaw;
    float a_pitch;

    uint8_t detect_color; // 0 red 1 blue
} __attribute__((packed));
constexpr uint8_t ID_NAV_CONTROL = 0;
constexpr uint8_t ID_OMNI_CONTROL = 1;
constexpr uint8_t ID_BT_CONTROL = 2;
struct NavRobotCmdData {
    uint8_t cmd_ID; //命令码
    uint32_t time_stamp;
    uint8_t packet_type; // 0 导航底盘控制  1 全向感知  2 决策信息
    // 0 导航底盘控制
    float vx;
    float vy;
    float wz;
    // 1 全向感知
    uint8_t omni_camera_detect_id;
    // 2 决策信息
} __attribute__((packed));

struct ReceiveReferee //rmul_2026
{
    uint8_t cmd_ID; //命令码
    uint32_t time_stamp;
    float big_yaw_in_world;
    int game_time; //gamestate 到4 从0开始，其他状态发-1
    int max_health;
    int cur_health;
    int cur_bullet;
    uint8_t center_state; //0 为未被占领，1 为被己方占领，2 为被对方占领，3 为被双方占领

} __attribute__((packed));
} // namespace wust_vision