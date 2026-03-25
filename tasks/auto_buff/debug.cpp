#include "debug.hpp"
#include "tasks/auto_buff/auto_buff.hpp"

namespace wust_vision::auto_buff {
void drawDebugRuneContent(
    cv::Mat& debug_img,
    const AutoBuffDebug& dbg,
    std::pair<cv::Mat, cv::Mat> camera_info
) {
    const auto& gimbal_cmd = dbg.gimbal_cmd;
    double predict_angle = dbg.predict_angle;
    auto aim_target = dbg.aim_target;
    auto auto_buff = dbg.power_rune;
    const cv::Rect img_rect(0, 0, debug_img.cols, debug_img.rows);
    const cv::Rect roi = dbg.expanded & img_rect;
    cv::rectangle(debug_img, roi, cv::Scalar(255, 255, 255), 2);

    const std::string latency_str = fmt::format("Latency: {:.2f}ms", dbg.latency_ms);
    cv::putText(
        debug_img,
        latency_str,
        cv::Point(10, 30),
        cv::FONT_HERSHEY_SIMPLEX,
        0.8,
        cv::Scalar(255, 255, 255),
        2
    );
    aim_target.tf(dbg.T_camera_to_odom.inverse());
    {
        const auto pts = aim_target.toPts(camera_info.first, camera_info.second);
        if (!pts.empty()) {
            cv::Scalar color = cv::Scalar(255, 255, 255);
            for (int i = 0; i < 4; i++)
                cv::line(debug_img, pts[i], pts[(i + 1) % 4], color, 2);

            // 后表面
            for (int i = 4; i < 8; i++)
                cv::line(debug_img, pts[i], pts[4 + (i + 1) % 4], color, 2);

            // 侧边
            for (int i = 0; i < 4; i++)
                cv::line(debug_img, pts[i], pts[i + 4], color, 2);
            cv::Point2f center(0.f, 0.f);
            for (auto pt: pts) {
                center += pt;
            }
            center *= 1.0 / pts.size();

            if (gimbal_cmd.fire_advice) {
                int cross_len = 60;
                cv::line(
                    debug_img,
                    center + cv::Point2f(-cross_len, -cross_len),
                    center + cv::Point2f(+cross_len, +cross_len),
                    cv::Scalar(0, 0, 255),
                    5
                );
                cv::line(
                    debug_img,
                    center + cv::Point2f(-cross_len, +cross_len),
                    center + cv::Point2f(+cross_len, -cross_len),
                    cv::Scalar(0, 0, 255),
                    5
                );
            }

            const double scale = 10.0;
            const double v_yaw = gimbal_cmd.v_yaw;
            const double v_pitch = gimbal_cmd.v_pitch;
            const double dx = -scale * v_yaw;
            const double dy = scale * v_pitch;

            const cv::Point2f start_pt = center;
            const cv::Point2f end_pt = start_pt + cv::Point2f(dx, dy);
            const cv::Scalar color_x = cv::Scalar(50, 50, 255);
            cv::arrowedLine(debug_img, start_pt, end_pt, color_x, 4, cv::LINE_AA, 0, 0.2);
        }
    }
    if (gimbal_cmd.fire_advice) {
        const std::string fire_str = "Fire!";
        cv::putText(
            debug_img,
            fire_str,
            { debug_img.cols / 2 - 100, 200 },
            cv::FONT_HERSHEY_SIMPLEX,
            2.85,
            cv::Scalar(0, 0, 255),
            2
        );
    }

    const std::string gimbal_str = fmt::format(
        "Pitch: {:.2f}, Yaw: {:.2f}, Enable_pitch_diff: {:.2f}, Enable_yaw_diff: {:.2f}, V_yaw: {:.2f}, V_pitch: {:.2f}",
        gimbal_cmd.pitch,
        gimbal_cmd.yaw,
        gimbal_cmd.enable_pitch_diff,
        gimbal_cmd.enable_yaw_diff,
        gimbal_cmd.v_yaw,
        gimbal_cmd.v_pitch
    );
    cv::putText(
        debug_img,
        gimbal_str,
        { 10, debug_img.rows - 30 },
        cv::FONT_HERSHEY_SIMPLEX,
        0.8,
        cv::Scalar(255, 255, 0),
        2
    );
    auto_buff.tf(dbg.T_camera_to_odom.inverse());
    auto_buff.draw(debug_img, camera_info.first, camera_info.second);
    cv::circle(
        debug_img,
        cv::Point2i(debug_img.cols / 2, debug_img.rows / 2),
        5,
        cv::Scalar(255, 255, 255),
        2
    );
}
void writeTargetLogToJson(const auto_buff::RuneTarget& rune_target) {
    nlohmann::json j;

    const auto now = std::chrono::steady_clock::now();

    nlohmann::json jr;
    jr["tracking"] = true;
    jr["id"] = static_cast<int>(rune_target.last_id);

    const auto age_ms_r =
        std::chrono::duration_cast<std::chrono::milliseconds>(now - rune_target.timestamp_).count();
    jr["timestamp_age_ms"] = age_ms_r;

    jr["position"] = { { "x", rune_target.centerPos().x() },
                       { "y", rune_target.centerPos().y() },
                       { "z", rune_target.centerPos().z() } };

    jr["roll"] = rune_target.roll() * 180.0 / M_PI;
    jr["yaw"] = rune_target.yaw() * 180.0 / M_PI;
    jr["v_roll"] = rune_target.v_roll() * 180.0 / M_PI;

    j["rune_target"] = jr;

    // -------- 写文件 --------
    std::ofstream file("/dev/shm/target_log.json");
    if (file.is_open()) {
        file << j.dump(2);
    }
}
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
    if (dbg_rune.gimbal_cmd.appear) {
        i_use = dbg_rune.gimbal_cmd;
    } else {
        i_use = last_cmd_;
    }
    last_cmd_ = i_use;
    nlohmann::json j;
    log.time_log.handleOnce(t, j);
    log.raw_yaw_log.handleOnce(i_use.target_yaw, j);
    log.raw_pitch_log.handleOnce(i_use.target_pitch, j);
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