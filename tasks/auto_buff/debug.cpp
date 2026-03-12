#include "debug.hpp"
#include "tasks/auto_buff/auto_buff.hpp"

namespace wust_vision::auto_buff {
struct DebugLogs {
#define DEBUG_LOG_LIST(X) \
    X(double, 100, time) \
    X(double, 100, raw_yaw) \
    X(double, 100, raw_pitch) \
    X(double, 100, yaw) \
    X(double, 100, pitch) \
    X(double, 100, ypd_y) \
    X(double, 100, ypd_p) \
    X(double, 100, rune_obs) \
    X(double, 100, rune_pre) \
    X(double, 100, rune_obsv) \
    X(double, 100, rune_fitv) \
    X(double, 100, gimbal_yaw) \
    X(double, 100, gimbal_pitch) \
    X(double, 100, target_v_yaw) \
    X(double, 100, control_v_yaw) \
    X(double, 100, control_v_pitch) \
    X(double, 100, yaw_diff) \
    X(double, 100, fire) \
    X(double, 100, rune_dis) \
    X(double, 100, fly_time) \
    X(double, 100, control_a_yaw) \
    X(double, 100, control_a_pitch)
#define GEN_LOG(TYPE, SIZE, NAME) LogsStream<TYPE, SIZE> NAME##_log { #NAME };

#define X(TYPE, SIZE, NAME) GEN_LOG(TYPE, SIZE, NAME)
    DEBUG_LOG_LIST(X)
#undef X

    void clear() {
#define X(TYPE, SIZE, NAME) NAME##_log.clear();
        DEBUG_LOG_LIST(X)
#undef X
    }
};

void debuglog(const AutoBuffDebug& dbg_rune) {
    static bool first_log = true;
    static std::chrono::steady_clock::time_point start_time;
    static DebugLogs log;
    static GimbalCmd last_cmd_;
    static double rune_dis = 0.0;
    if (first_log) {
        start_time = std::chrono::steady_clock::now();
        first_log = false;
    }

    const auto now = std::chrono::steady_clock::now();
    const double t = std::chrono::duration<double>(now - start_time).count();

    const auto_buff::RuneTarget& rune_target = dbg_rune.target;
    writeTargetLogToJson(rune_target);

    double armor_yaw = 0.0, ypd_y = 0.0, ypd_p = 0.0, armor_distance = 0.0;
    if (dbg_rune.pnp_distance > 1.0) {
        rune_dis = dbg_rune.pnp_distance;
    }

    GimbalCmd i_use;
    if (dbg_rune.gimbal_cmd.appera) {
        i_use = dbg_rune.gimbal_cmd;
    } else {
        i_use = last_cmd_;
    }
    last_cmd_ = i_use;
    nlohmann::json j;
    log.time_log.handleOnce(t, j);
    log.raw_yaw_log.handleOnce(i_use.raw_yaw, j);
    log.raw_pitch_log.handleOnce(i_use.raw_pitch, j);
    log.yaw_log.handleOnce(i_use.yaw, j);
    log.pitch_log.handleOnce(i_use.pitch, j);
    log.rune_obs_log.handleOnce(dbg_rune.obs_angle, j);
    log.rune_pre_log.handleOnce(dbg_rune.pre_angle, j);
    log.rune_fitv_log.handleOnce(dbg_rune.fitter_v * 180.0 / M_PI, j);
    log.rune_obsv_log.handleOnce(dbg_rune.obs_v * 180.0 / M_PI, j);
    log.gimbal_pitch_log.handleOnce(dbg_rune.gimbal_py.first * 180.0 / M_PI, j);
    log.gimbal_yaw_log.handleOnce(dbg_rune.gimbal_py.second * 180.0 / M_PI, j);
    log.control_v_pitch_log.handleOnce(i_use.v_pitch, j);
    log.control_v_yaw_log.handleOnce(i_use.v_yaw, j);
    log.fire_log.handleOnce(i_use.fire_advice, j);
    log.rune_dis_log.handleOnce(rune_dis, j);
    log.fly_time_log.handleOnce(i_use.fly_time, j);
    log.control_a_yaw_log.handleOnce(i_use.a_yaw / 180.0 * M_PI, j);
    log.control_a_pitch_log.handleOnce(i_use.a_pitch / 180.0 * M_PI, j);

    std::ofstream file("/dev/shm/cmd_log.json");
    if (file.is_open()) {
        file << j.dump();
    }
}

} // namespace wust_vision::auto_buff