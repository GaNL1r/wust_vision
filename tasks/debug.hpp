#pragma once
#include "tasks/auto_aim/armor_tracker/target.hpp"
#include "tasks/auto_aim/auto_aim_fsm.hpp"
#include "tasks/auto_aim/type.hpp"
#include "tasks/auto_buff/rune_tracker/rune_target.hpp"
#include "tasks/auto_buff/type.hpp"
#include "tasks/packet_typedef.hpp"
#include <nlohmann/json.hpp>
namespace wust_vision {
struct DebugArmor {
    imgframe src_img;
    auto_aim::Armors armors;
    auto_aim::Target target;
    GimbalCmd gimbal_cmd;
    auto_aim::AutoAimFsm fsm;
    AimTarget aim_target;
    double latency_ms;
    Eigen::Matrix4d T_camera_to_odom;
    std::vector<auto_aim::ArmorObject> armor_objs;
    int detect_color = 0;
    cv::Rect expanded;
};
struct DebugRune {
    imgframe src_img;
    auto_buff::RuneTarget target;
    AimTarget aim_target;
    auto_buff::PowerRune power_rune;
    double predict_angle;
    GimbalCmd gimbal_cmd;
    Eigen::Matrix4d T_camera_to_odom;
    std::string debug_text;
    double latency_ms;
    double obs_angle;
    double pre_angle;
    double fitter_v;
    double obs_v;
    cv::Rect expanded;
    double pnp_distance;
};
template<typename T, int MAX_N, const char* NAME>
class LogsStream {
public:
    void handleOnce(const T& t, nlohmann::json& j) {
        log_data.push_back(t);
        trim();
        insertData(j);
    }
    void push_back(const T& t) {
        log_data.push_back(t);
    }
    void trim() {
        while (log_data.size() > MAX_N) {
            log_data.erase(log_data.begin());
        }
    }
    void insertData(nlohmann::json& j) {
        j[NAME] = log_data;
    }
    void clear() {
        log_data.clear();
    }

private:
    std::vector<T> log_data;
};

struct DebugLogs {
#define DEBUG_LOG_LIST(X) \
    X(double, 100, time) \
    X(double, 100, raw_yaw) \
    X(double, 100, raw_pitch) \
    X(double, 100, yaw) \
    X(double, 100, pitch) \
    X(double, 100, armor_dis) \
    X(double, 100, armor_x) \
    X(double, 100, armor_y) \
    X(double, 100, armor_z) \
    X(double, 100, armor_yaw) \
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
#define GEN_LOG(TYPE, SIZE, NAME) \
    inline static constexpr char k##NAME##Name[] = #NAME; \
    LogsStream<TYPE, SIZE, k##NAME##Name> NAME##_log;

#define X(TYPE, SIZE, NAME) GEN_LOG(TYPE, SIZE, NAME)
    DEBUG_LOG_LIST(X)
#undef X

    void clear() {
#define X(TYPE, SIZE, NAME) NAME##_log.clear();
        DEBUG_LOG_LIST(X)
#undef X
    }
};

void drawDebugOverlayShm(
    const DebugArmor& dbg,
    std::pair<cv::Mat, cv::Mat> camera_info,
    bool auto_fps
);
void drawDebugOverlayWrite(
    const DebugArmor& dbg,
    std::pair<cv::Mat, cv::Mat> camera_info,
    bool auto_fps
);
void drawDebugOverlayShow(
    const DebugArmor& dbg,
    std::pair<cv::Mat, cv::Mat> camera_info,
    bool auto_fps
);
cv::Mat drawDebugOverlayMat(const DebugArmor& dbg, std::pair<cv::Mat, cv::Mat> camera_info);
void drawDebugArmorContent(
    cv::Mat& debug_img,
    const DebugArmor& dbg,
    std::pair<cv::Mat, cv::Mat> camera_info
);
void drawDebugOverlayShm(
    const DebugRune& dbg,
    std::pair<cv::Mat, cv::Mat> camera_info,
    bool auto_fps
);
void drawDebugOverlayWrite(
    const DebugRune& dbg,
    std::pair<cv::Mat, cv::Mat> camera_info,
    bool auto_fps
);
void drawDebugOverlayShow(
    const DebugRune& dbg,
    std::pair<cv::Mat, cv::Mat> camera_info,
    bool auto_fps
);
cv::Mat drawDebugOverlayMat(const DebugRune& dbg, std::pair<cv::Mat, cv::Mat> camera_info);
void drawDebugRuneContent(
    cv::Mat& debug_img,
    const DebugRune& dbg,
    std::pair<cv::Mat, cv::Mat> camera_info
);
void debuglog(
    const DebugArmor& dbg_armor,
    const DebugRune& dbg_rune,
    const GimbalCmd& gimbal_cmd,
    const std::pair<double, double>& gimbal_py
);
void writeSerialLogToJson(const ReceiveAimINFO& aim);
} // namespace wust_vision