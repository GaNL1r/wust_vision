#pragma once
#include "tasks/auto_buff/auto_buff.hpp"
#include "tasks/auto_buff/rune_tracker/rune_target.hpp"
#include "tasks/utils/debug_utils.hpp"
#include <wust_vl/video/icamera.hpp>
namespace wust_vision::auto_buff {
struct AutoBuffDebug {
    wust_vl::video::ImageFrame img_frame;
    auto_buff::RuneTarget target;
    AimTarget aim_target;
    auto_buff::PowerRune power_rune;
    double predict_angle;
    GimbalCmd gimbal_cmd;
    Eigen::Matrix4d T_camera_to_odom;
    double latency_ms;
    double obs_angle;
    double pre_angle;
    double fitter_v;
    double obs_v;
    cv::Rect expanded;
    double pnp_distance;
    std::pair<double, double> gimbal_py;
};
inline void drawDebugRuneContent(
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
    if (!aim_target.is_old) {
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
inline void writeTargetLogToJson(const auto_buff::RuneTarget& rune_target) {
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
inline void drawDebugOverlayShm(
    const AutoBuffDebug& dbg,
    std::pair<cv::Mat, cv::Mat> camera_info,
    bool auto_fps
) {
    static ShmWriter shm { "/debug_frame" };
    drawDebugOverlayImpl(dbg, camera_info, auto_fps, drawDebugRuneContent, shm);
}
void debuglog(const AutoBuffDebug& dbg);
} // namespace wust_vision::auto_buff