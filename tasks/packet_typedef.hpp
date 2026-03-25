#pragma once

#include <chrono>
#include <cstdint>
#include <fstream>
#include <mutex>
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
    uint8_t cmd_ID;
    uint32_t time_stamp;

    float yaw;
    float pitch;
    float roll;

    float yaw_vel;
    float pitch_vel;
    float roll_vel;

    float v_x;
    float v_y;
    float v_z;

    float bullet_speed;
    uint8_t detect_color; // 0 red 1 blue
} __attribute__((packed));

struct ReceiveReferee {
    uint8_t cmd_ID;
    uint32_t time_stamp;

    float big_yaw_in_world;
    int game_time;
    int max_health;
    int cur_health;
    int cur_bullet;

    uint8_t center_state;
} __attribute__((packed));
struct SendRobotCmdData {
    uint8_t cmd_ID;
    uint32_t time_stamp;

    uint8_t appear;
    uint8_t shoot_rate = 3;

    float pitch;
    float yaw;

    float target_yaw;
    float target_pitch;

    float enable_yaw_diff;
    float enable_pitch_diff;

    float v_yaw;
    float v_pitch;
    float a_yaw;
    float a_pitch;

    uint8_t detect_color;
} __attribute__((packed));

constexpr uint8_t ID_NAV_CONTROL = 0;

struct NavRobotCmdData {
    uint8_t cmd_ID;
    uint32_t time_stamp;
    uint8_t packet_type;

    float vx;
    float vy;
    float wz;

} __attribute__((packed));

struct SerialLogBuffer {
    std::mutex mtx;
    nlohmann::json j;
    bool dirty = false;
};

inline SerialLogBuffer& getLogBuffer() {
    static SerialLogBuffer buf;
    return buf;
}

inline void updateFPS(nlohmann::json& j) {
    static int frame_count = 0;
    static double fps = 0.0;
    static auto last_time = std::chrono::steady_clock::now();

    ++frame_count;

    auto now = std::chrono::steady_clock::now();
    double elapsed = std::chrono::duration<double>(now - last_time).count();

    if (elapsed >= 1.0) {
        fps = frame_count / elapsed;
        frame_count = 0;
        last_time = now;
    }

    j["fps"] = fps;
}

inline void updateSerialLog(const ReceiveAimINFO& aim) {
    auto& buf = getLogBuffer();
    std::lock_guard<std::mutex> lock(buf.mtx);

    auto& j = buf.j["aim"];
    updateFPS(j);
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

    j["bullet_speed"] = aim.bullet_speed;
    j["detect_color"] = (aim.detect_color == 0 ? "Red" : "Blue");

    buf.dirty = true;
}

inline void updateSerialLog(const ReceiveReferee& ref) {
    auto& buf = getLogBuffer();
    std::lock_guard<std::mutex> lock(buf.mtx);

    auto& j = buf.j["referee"];
    updateFPS(j);
    j["timestamp"] = ref.time_stamp;
    j["big_yaw_in_world"] = ref.big_yaw_in_world;
    j["game_time"] = ref.game_time;
    j["max_health"] = ref.max_health;
    j["cur_health"] = ref.cur_health;
    j["cur_bullet"] = ref.cur_bullet;
    j["center_state"] = ref.center_state;

    buf.dirty = true;
}

inline void flushSerialLog() {
    static auto last_flush = std::chrono::steady_clock::now();

    auto& buf = getLogBuffer();

    auto now = std::chrono::steady_clock::now();
    double dt = std::chrono::duration<double>(now - last_flush).count();

    // 控制写入频率（例如 20Hz）
    if (dt < 0.05)
        return;

    std::lock_guard<std::mutex> lock(buf.mtx);

    if (!buf.dirty)
        return;

    // 更新 FPS

    std::ofstream file("/dev/shm/serial_log.json");
    if (file.is_open()) {
        file << buf.j.dump(2);
        buf.dirty = false;
    }

    last_flush = now;
}

} // namespace wust_vision