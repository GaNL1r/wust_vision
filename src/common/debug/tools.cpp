#include "common/debug/tools.hpp"
#include "common/debug/toolsgobal.hpp"
#include "common/gobal.hpp"
#include "common/tf.hpp"
#include "detect/mono_measure_tool.hpp"
#include "fmt/format.h"
#include "tracker/tracker.hpp"
#include "type/type.hpp"
#include <chrono>
#include <cstddef>
#include <fstream>
#include <nlohmann/json.hpp>
#include <opencv2/core/mat.hpp>
#include <opencv2/core/types.hpp>
#include <opencv2/opencv.hpp>
#include <optional>
#include <string>
#include <vector>
void drawDebugArmorContent(cv::Mat& debug_img, const DebugArmor& dbg) {
    static float yaw_diff = 0;

    auto now = std::chrono::steady_clock::now();

    // =================== 装甲板绘制 ===================
    if (dbg.armors) {
        const auto& armors = *dbg.armors;
        static const int next_indices[] = { 2, 0, 3, 1 };

        for (const auto& armor: armors.armors) {
            std::vector<cv::Point2f> pts;
            if (!gobal::measure_tool->reprojectArmorCorners_raw(
                    armor,
                    pts,
                    gobal::camera_intrinsic,
                    gobal::camera_distortion
                ))
                continue;

            for (size_t i = 0; i < 4; ++i) {
                cv::Scalar color = armor.is_ok ? cv::Scalar(0, 255, 0) : cv::Scalar(0, 255, 255);
                cv::line(debug_img, pts[i], pts[next_indices[i]], color, 2);
            }

            std::string yaw_info = fmt::format("Yaw: {:.2f}", armor.yaw * 180.0 / M_PI);
            cv::putText(
                debug_img,
                yaw_info,
                pts[0] + cv::Point2f(0, -50),
                cv::FONT_HERSHEY_SIMPLEX,
                0.7,
                cv::Scalar(0, 200, 200),
                2
            );
            auto armorName = [](armor::ArmorNumber num) {
                switch (num) {
                    case armor::ArmorNumber::SENTRY:
                        return "SENTRY";
                    case armor::ArmorNumber::BASE:
                        return "BASE";
                    case armor::ArmorNumber::OUTPOST:
                        return "OUTPOST";
                    case armor::ArmorNumber::NO1:
                        return "NO1";
                    case armor::ArmorNumber::NO2:
                        return "NO2";
                    case armor::ArmorNumber::NO3:
                        return "NO3";
                    case armor::ArmorNumber::NO4:
                        return "NO4";
                    case armor::ArmorNumber::NO5:
                        return "NO5";
                    default:
                        return "UNKNOWN";
                }
            };
            std::string armor_str = armorName(armor.number);
            cv::putText(
                debug_img,
                armor_str,
                pts[1] + cv::Point2f(0, 50),
                cv::FONT_HERSHEY_SIMPLEX,
                0.7,
                cv::Scalar(0, 200, 200),
                2
            );
        }

        if (armors.armors.size() == 2) {
            double diff = armors.armors[0].yaw - armors.armors[1].yaw;
            while (diff > M_PI)
                diff -= 2 * M_PI;
            while (diff < -M_PI)
                diff += 2 * M_PI;
            yaw_diff = std::abs(diff);
        }

        std::string yaw_diff_str = fmt::format("Yaw_diff: {:.2f}", yaw_diff * 180.0 / M_PI);
        cv::putText(
            debug_img,
            yaw_diff_str,
            cv::Point(100, 150),
            cv::FONT_HERSHEY_SIMPLEX,
            2.0,
            cv::Scalar(40, 255, 40),
            2
        );

        std::string latency_str = fmt::format("Latency: {:.2f}ms", toolsgobal::latency_ms);
        cv::putText(
            debug_img,
            latency_str,
            cv::Point(10, 30),
            cv::FONT_HERSHEY_SIMPLEX,
            0.8,
            cv::Scalar(255, 255, 255),
            2
        );
    }

    // =================== 目标绘制 ===================
    std::vector<cv::Point2f> all_corners;
    if (dbg.target_info && dbg.target) {
        const auto& target_info = *dbg.target_info;
        const auto& target = *dbg.target;

        for (size_t i = 0; i < target_info.pts.size(); ++i) {
            const auto& pts = target_info.pts[i];
            const auto& pos = target_info.pos[i];
            const auto& ori = target_info.ori[i];
            const auto& is_ok = target_info.is_ok[i];

            for (size_t j = 0; j < 4; ++j) {
                cv::Scalar color = is_ok ? cv::Scalar(255, 0, 0) : cv::Scalar(0, 0, 255);
                cv::line(debug_img, pts[j], pts[(j + 1) % 4], color, 2);
            }

            all_corners.insert(all_corners.end(), pts.begin(), pts.end());

            double yaw = getYawFromQuaternion(ori);
            double distance = std::sqrt(pos.x * pos.x + pos.y * pos.y + pos.z * pos.z);

            std::vector<std::string> info_lines = {
                fmt::format("Dis: {:.1f}cm", distance * 100),
                fmt::format("X: {:.2f}", pos.x),
                fmt::format("Y: {:.2f}", pos.y),
                fmt::format("Z: {:.2f}", pos.z),
                fmt::format("Yaw: {:.2f}", yaw * 180.0 / M_PI)
            };

            cv::Point2f text_org = pts[0] + cv::Point2f(0, 200);
            for (int k = 0; k < info_lines.size(); ++k) {
                cv::putText(
                    debug_img,
                    info_lines[k],
                    text_org + cv::Point2f(0, -10 - 20 * k),
                    cv::FONT_HERSHEY_SIMPLEX,
                    0.6,
                    cv::Scalar(50, 255, 255),
                    1
                );
            }
        }

        if (target_info.select_id != -1 && target_info.select_id < target_info.pts.size()
            && !target_info.pts[target_info.select_id].empty())
        {
            cv::Point2f center(0.f, 0.f);
            for (int i = 0; i < 4; ++i)
                center += target_info.pts[target_info.select_id][i];
            center *= 0.25f;

            cv::circle(debug_img, center + cv::Point2f(0, -200), 20, cv::Scalar(0, 255, 255), 5);

            if (dbg.gimbal_cmd && dbg.gimbal_cmd->fire_advice) {
                int cross_len = 60;
                cv::line(
                    debug_img,
                    center + cv::Point2f(0, -200) + cv::Point2f(-cross_len, -cross_len),
                    center + cv::Point2f(0, -200) + cv::Point2f(+cross_len, +cross_len),
                    cv::Scalar(0, 0, 255),
                    5
                );
                cv::line(
                    debug_img,
                    center + cv::Point2f(0, -200) + cv::Point2f(-cross_len, +cross_len),
                    center + cv::Point2f(0, -200) + cv::Point2f(+cross_len, -cross_len),
                    cv::Scalar(0, 0, 255),
                    5
                );
            }
        }

        if (!all_corners.empty()) {
            cv::Point2f avg(0.f, 0.f);
            for (const auto& pt: all_corners)
                avg += pt;
            avg *= 1.0f / all_corners.size();
            cv::circle(debug_img, avg, 5, cv::Scalar(0, 255, 0), -1);
        }

        if (dbg.src_img) {
            auto latency_img_target = std::chrono::duration_cast<std::chrono::microseconds>(
                                          dbg.src_img->timestamp - target.timestamp
                                      )
                                          .count()
                / 1000.0;
            cv::putText(
                debug_img,
                fmt::format("Img-Frame Delay: {:.2f}ms", latency_img_target),
                cv::Point(10, 60),
                cv::FONT_HERSHEY_SIMPLEX,
                0.8,
                cv::Scalar(255, 255, 255),
                2
            );
        }
    }

    // =================== Tracker 状态 ===================
    if (dbg.tracker_state) {
        std::string state_str;
        switch (dbg.tracker_state.value()) {
            case Tracker::LOST:
                state_str = "LOST";
                break;
            case Tracker::DETECTING:
                state_str = "DETECTING";
                break;
            case Tracker::TRACKING:
                state_str = "TRACKING";
                break;
            case Tracker::TEMP_LOST:
                state_str = "TEMP_LOST";
                break;
            default:
                state_str = "UNKNOWN";
                break;
        }
        int baseline = 0;
        cv::Size size = cv::getTextSize(state_str, cv::FONT_HERSHEY_SIMPLEX, 2.5, 2, &baseline);
        int x = std::max(0, debug_img.cols - size.width - 10);
        int y = std::min(debug_img.rows - 1, size.height + 10);
        cv::putText(
            debug_img,
            state_str,
            { x, y },
            cv::FONT_HERSHEY_SIMPLEX,
            2.5,
            cv::Scalar(0, 0, 255),
            2
        );
    }

    // =================== 当前攻击目标 ID ===================
    if (dbg.target) {
        std::string id_str = fmt::format("Attack: {}", armorNumberToString(dbg.target->id));
        int baseline = 0;
        cv::Size size = cv::getTextSize(id_str, cv::FONT_HERSHEY_SIMPLEX, 1.6, 2, &baseline);
        int x = std::max(0, debug_img.cols - size.width - 10);
        int y = std::min(debug_img.rows - 1, 100);
        cv::putText(
            debug_img,
            id_str,
            { x, y },
            cv::FONT_HERSHEY_SIMPLEX,
            1.6,
            cv::Scalar(255, 0, 255),
            2
        );
    }

    // =================== Fire 标志 ===================
    std::string fire_str = (dbg.gimbal_cmd && dbg.gimbal_cmd->fire_advice) ? "Fire!" : "";
    if (!fire_str.empty()) {
        int baseline = 0;
        cv::Size fire_size = cv::getTextSize(fire_str, cv::FONT_HERSHEY_SIMPLEX, 1.2, 2, &baseline);
        int fire_x = 1440 / 2 - fire_size.width - 10;
        int fire_y = 200;
        cv::putText(
            debug_img,
            fire_str,
            { fire_x, fire_y },
            cv::FONT_HERSHEY_SIMPLEX,
            2.85,
            cv::Scalar(0, 0, 255),
            2
        );
    }

    // =================== 云台指令 ===================
    if (dbg.gimbal_cmd) {
        const auto& cmd = *dbg.gimbal_cmd;
        std::string gimbal_str = fmt::format(
            "Pitch: {:.2f}, Yaw: {:.2f}, Pitch_diff: {:.2f}, Yaw_diff: {:.2f}",
            cmd.pitch,
            cmd.yaw,
            cmd.pitch_diff,
            cmd.yaw_diff
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
    }

    // =================== 屏幕中心点 ===================
    cv::circle(
        debug_img,
        cv::Point2i(debug_img.cols / 2, debug_img.rows / 2),
        5,
        cv::Scalar(255, 255, 255),
        2
    );
}
void drawDebugRuneContent(cv::Mat& debug_img, const DebugRune& dbg) {
    if (!dbg.objs.has_value() || !dbg.gimbal_cmd.has_value())
        return;

    const auto& objs = dbg.objs.value();
    const auto& gimbal_cmd = dbg.gimbal_cmd.value();
    double predict_angle = dbg.predict_angle.value_or(0.0);
    const auto& manual_r_box = dbg.manual_r_box.value_or(std::vector<cv::Point2f> {});

    for (const auto& obj: objs) {
        if (obj.type == rune::RuneType::INACTIVATED) {
            const auto pts = obj.pts.toVector2f();
            if (pts.size() < 3)
                break;

            int sharpest_idx = 0;
            const cv::Point2f& tip = pts[sharpest_idx];
            cv::Point2f aim_point =
                std::accumulate(pts.begin(), pts.end(), cv::Point2f(0, 0)) - tip;
            aim_point *= (1.f / (pts.size() - 1));

            cv::Point2f vec_to_aim = normalize(aim_point - tip);
            float base_angle = std::atan2(vec_to_aim.y, vec_to_aim.x);
            float angle_rad = base_angle - predict_angle;
            std::vector<cv::Point2f> pts_exclude_tip;
            for (size_t i = 0; i < pts.size(); ++i) {
                if (i != sharpest_idx)
                    pts_exclude_tip.push_back(pts[i]);
            }
            float area = std::fabs(cv::contourArea(pts_exclude_tip));

            float radius = std::sqrt(area / static_cast<float>(CV_PI));

            float length = cv::norm(aim_point - tip);
            cv::Point2f end_point_line =
                tip + cv::Point2f(std::cos(angle_rad), std::sin(angle_rad)) * (length - radius);
            cv::Point2f end_point_circle =
                tip + cv::Point2f(std::cos(angle_rad), std::sin(angle_rad)) * length;

            cv::line(debug_img, tip, end_point_line, cv::Scalar(255, 255, 255), 2);

            cv::circle(
                debug_img,
                end_point_circle,
                static_cast<int>(radius),
                cv::Scalar(255, 255, 255),
                5
            );
            cv::circle(debug_img, end_point_circle, 5, cv::Scalar(255, 255, 255), -1);

            break;
        }
    }

    for (int i = 0; i < manual_r_box.size(); i++) {
        cv::line(debug_img, manual_r_box[i], manual_r_box[(i + 1) % 4], cv::Scalar(48, 48, 255), 1);
    }

    for (const auto& obj: objs) {
        auto pts = obj.pts.toVector2f();
        cv::Point2f aim_point =
            std::accumulate(pts.begin() + 1, pts.end(), cv::Point2f(0, 0)) / 4.0f;

        cv::Scalar line_color = obj.type == rune::RuneType::INACTIVATED ? cv::Scalar(50, 255, 50)
                                                                        : cv::Scalar(255, 50, 255);

        cv::putText(
            debug_img,
            fmt::format("{:.2f}", obj.prob),
            cv::Point2i(pts[1]),
            cv::FONT_HERSHEY_SIMPLEX,
            0.8,
            line_color,
            2
        );
        cv::polylines(debug_img, obj.pts.toVector2i(), true, line_color, 2);
        cv::circle(debug_img, aim_point, 5, line_color, -1);

        std::string rune_type = obj.type == rune::RuneType::INACTIVATED ? "_HIT" : "_OK";
        std::string rune_color = armor::enemyColorToString(obj.color);
        cv::putText(
            debug_img,
            rune_color + rune_type,
            cv::Point2i(pts[2]),
            cv::FONT_HERSHEY_SIMPLEX,
            0.8,
            line_color,
            2
        );
        for (int i = 0; i < pts.size(); i++) {
            std::string str = std::to_string(i);
            cv::putText(
                debug_img,
                str,
                pts[i],
                cv::FONT_HERSHEY_SIMPLEX,
                0.8,
                cv::Scalar(0, 50, 255),
                2
            );
        }
    }

    std::string latency_str = fmt::format("Latency: {:.2f}ms", toolsgobal::latency_ms);
    cv::putText(
        debug_img,
        latency_str,
        cv::Point2i(10, 30),
        cv::FONT_HERSHEY_SIMPLEX,
        0.8,
        cv::Scalar(255, 255, 255),
        2
    );

    cv::circle(
        debug_img,
        cv::Point2i(debug_img.cols / 2, debug_img.rows / 2),
        5,
        cv::Scalar(255, 255, 255),
        2
    );

    int baseline = 0;
    std::string fire_str = gimbal_cmd.fire_advice ? "Fire!" : "";
    cv::Size fire_size = cv::getTextSize(fire_str, cv::FONT_HERSHEY_SIMPLEX, 1.2, 2, &baseline);
    int fire_x = 1440 / 2 - fire_size.width - 10;
    int fire_y = 200;

    cv::putText(
        debug_img,
        fire_str,
        { fire_x, fire_y },
        cv::FONT_HERSHEY_SIMPLEX,
        2.85,
        cv::Scalar(0, 0, 255),
        2
    );

    std::string gimbal_str = fmt::format(
        "Pitch: {:.2f}, Yaw: {:.2f}, Pitch_diff: {:.2f}, Yaw_diff: {:.2f}",
        gimbal_cmd.pitch,
        gimbal_cmd.yaw,
        gimbal_cmd.pitch_diff,
        gimbal_cmd.yaw_diff
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
}

void drawResult(const imgframe& src_img, const armor::Armors& armors) {
    static auto last_show_time = std::chrono::steady_clock::now();
    static bool window_initialized = false;
    static int brightness_slider = 200;
    static float yaw_diff;

    if (src_img.img.empty())
        return;

    if (!window_initialized) {
        cv::namedWindow("debug_armor", cv::WINDOW_NORMAL);
        cv::resizeWindow("debug_armor", toolsgobal::debug_w, toolsgobal::debug_h);
        cv::createTrackbar("Brightness", "debug_armor", &brightness_slider, 400);
        window_initialized = true;
    }

    auto now = std::chrono::steady_clock::now();
    const double min_interval_ms = 1000.0 / toolsgobal::debug_fps;

    double elapsed_ms = std::chrono::duration<double, std::milli>(now - last_show_time).count();
    if (elapsed_ms < min_interval_ms) {
        return;
    }
    last_show_time = now;

    // 调整亮度
    cv::Mat debug_img;

    double brightness_factor = brightness_slider / 100.0;
    src_img.img.convertTo(debug_img, -1, brightness_factor, 0);

    cv::cvtColor(debug_img, debug_img, cv::COLOR_BGR2RGB);

    static const int next_indices[] = { 2, 0, 3, 1 };
    for (auto& armor: armors.armors) {
        // std::cout<<"cc\n";
        std::vector<cv::Point2f> pts;

        if (!gobal::measure_tool->reprojectArmorCorners_raw(
                armor,
                pts,
                gobal::camera_intrinsic,
                gobal::camera_distortion
            ))
            continue;
        for (size_t i = 0; i < 4; ++i) {
            cv::line(debug_img, pts[i], pts[(i + 1) % 4], cv::Scalar(255, 100, 0), 2);
        }
        std::string yaw_info = fmt::format("Yaw: {:.3f}", armor.yaw / M_PI * 180);
        cv::putText(
            debug_img,
            yaw_info,
            cv::Point(pts[0].x, pts[0].y - 50),
            cv::FONT_HERSHEY_SIMPLEX,
            0.7,
            cv::Scalar(0, 200, 200),
            2
        );
        std::string armor_info;
        switch (armor.number) {
            case armor::ArmorNumber::SENTRY:
                armor_info = "SENTRY";
                break;
            case armor::ArmorNumber::BASE:
                armor_info = "BASE";
                break;
            case armor::ArmorNumber::OUTPOST:
                armor_info = "OUTPOST";
                break;
            case armor::ArmorNumber::NO1:
                armor_info = "NO1";
                break;
            case armor::ArmorNumber::NO2:
                armor_info = "NO2";
                break;
            case armor::ArmorNumber::NO3:
                armor_info = "NO3";
                break;
            case armor::ArmorNumber::NO4:
                armor_info = "NO4";
                break;
            case armor::ArmorNumber::NO5:
                armor_info = "NO5";
                break;
            case armor::ArmorNumber::UNKNOWN:
                armor_info = "UNKNOWN";
                break;
        }
        cv::putText(
            debug_img,
            armor_info,
            cv::Point(pts[1].x, pts[1].y + 50),
            cv::FONT_HERSHEY_SIMPLEX,
            0.7,
            cv::Scalar(0, 200, 200),
            2
        );
    }

    if (armors.armors.size() == 2) {
        double yaw1 = armors.armors[0].yaw;
        double yaw2 = armors.armors[1].yaw;

        // 计算周期性最小差值，结果范围在 [-π, π]
        double diff = yaw1 - yaw2;
        while (diff > M_PI)
            diff -= 2 * M_PI;
        while (diff < -M_PI)
            diff += 2 * M_PI;

        yaw_diff = std::abs(diff); // 始终是非负、最小差值

    } else {
        // yaw_diff = 0;
    }
    std::string yaw_diff_info = "Yaw_diff: " + std::to_string(yaw_diff / M_PI * 180);
    cv::putText(
        debug_img,
        yaw_diff_info,
        cv::Point(100, 100),
        cv::FONT_HERSHEY_SIMPLEX,
        2.7,
        cv::Scalar(40, 255, 40),
        2
    );
    cv::circle(
        debug_img,
        cv::Point2i(debug_img.cols / 2, debug_img.rows / 2),
        5,
        cv::Scalar(255, 255, 255),
        2
    );

    auto latency_nano =
        std::chrono::duration_cast<std::chrono::nanoseconds>(now - armors.timestamp).count();
    double latency_ms = static_cast<double>(latency_nano) / 1e6;

    std::string latency = fmt::format("Latency: {:.3f}ms", latency_ms);
    cv::putText(
        debug_img,
        latency,
        cv::Point(10, 30),
        cv::FONT_HERSHEY_SIMPLEX,
        0.8,
        cv::Scalar(255, 255, 255),
        2
    );

    cv::imshow("debug_armor", debug_img);
    cv::waitKey(1);
}

std::string formatTargetInfo(const armor::Target& target) {
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(2);

    auto now = std::chrono::steady_clock::now();
    auto age =
        std::chrono::duration_cast<std::chrono::milliseconds>(now - target.timestamp).count();

    oss << "=== Target Info ===\n";
    oss << "Frame ID      : " << target.frame_id << "\n";
    oss << "Type          : " << target.type << "\n";
    oss << "Tracking      : " << (target.tracking ? "Yes" : "No") << "\n";
    oss << "ID            : " << static_cast<int>(target.id) << "\n";
    oss << "Armors Num    : " << target.armors_num << "\n";
    oss << "Timestamp Age : " << age << " ms ago\n";

    oss << "\n-- Position --\n";
    oss << "x: " << target.position_.x << ", y: " << target.position_.y
        << ", z: " << target.position_.z << "\n";

    oss << "\n-- Velocity --\n";
    oss << "vx: " << target.velocity_.x << ", vy: " << target.velocity_.y
        << ", vz: " << target.velocity_.z << "\n";

    oss << "\n-- Yaw Info --\n";
    oss << "Yaw      : " << target.yaw << "\n";
    oss << "v_yaw    : " << target.v_yaw << "\n";
    oss << "Yaw Diff : " << target.yaw_diff << "\n";

    oss << "\n-- Radii  --\n";
    oss << "Radius 1       : " << target.radius_1 << "\n";
    oss << "Radius 2       : " << target.radius_2 << "\n";
    oss << "d_za           : " << target.d_za << "\n";
    oss << "d_zc           : " << target.d_zc << "\n";
    oss << "Position Diff  : " << target.position_diff << "\n";
    oss << "z_diff         : " << std::abs(target.d_za) + std::abs(target.d_zc) << "\n";

    return oss.str();
}
void dumpTargetToFile(const armor::Target& target, const std::string& path) {
    std::ofstream file(path);
    if (file.is_open()) {
        file << formatTargetInfo(target);
        file.close();
    }
}

std::string formatImuInfo(const ReceiveImuData& imu) {
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(3); // 设置输出精度

    // 静态变量用于统计帧率
    static int frame_count = 0;
    static double fps = 0.0;
    static auto last_time = std::chrono::steady_clock::now();

    // 每帧计数
    ++frame_count;

    // 时间间隔
    auto now = std::chrono::steady_clock::now();
    double elapsed = std::chrono::duration<double>(now - last_time).count();

    if (elapsed >= 1.0) {
        fps = frame_count / elapsed;
        frame_count = 0;
        last_time = now;
    }

    oss << "=== IMU Info ===\n";
    oss << "Timestamp     : " << imu.time_stamp << "\n";

    oss << "\n-- Orientation (rad) --\n";
    oss << "Yaw   : " << imu.data.yaw << "\n";
    oss << "Pitch : " << imu.data.pitch << "\n";
    oss << "Roll  : " << imu.data.roll << "\n";

    oss << "\n-- Angular Velocity (rad/s) --\n";
    oss << "Yaw_vel   : " << imu.data.yaw_vel << "\n";
    oss << "Pitch_vel : " << imu.data.pitch_vel << "\n";
    oss << "Roll_vel  : " << imu.data.roll_vel << "\n";

    oss << "\n-- Orientation (deg) --\n";
    oss << "Yaw   : " << imu.data.yaw * 180 / M_PI << "\n";
    oss << "Pitch : " << imu.data.pitch * 180 / M_PI << "\n";
    oss << "Roll  : " << imu.data.roll * 180 / M_PI << "\n";

    oss << "\nCRC           : 0x" << std::hex << imu.crc << std::dec << "\n";

    // 显示最近一次统计得到的 FPS
    oss << "Frame Rate (FPS): " << std::setprecision(1) << fps << "\n";

    return oss.str();
}

void dumpImuToFile(const ReceiveImuData& imu, const std::string& path) {
    std::ofstream file(path);
    if (file.is_open()) {
        file << formatImuInfo(imu);
        file.close();
    }
}
std::string formatAimInfo(const ReceiveAimINFO& aim) {
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(3);

    // 静态变量用于帧率统计
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

    oss << "=== Aim Info ===\n";
    oss << "Timestamp     : " << aim.time_stamp << "\n";

    oss << "\n-- Orientation (rad) --\n";
    oss << "Yaw   : " << aim.yaw << "\n";
    oss << "Pitch : " << aim.pitch << "\n";
    oss << "Roll  : " << aim.roll << "\n";

    oss << "\n-- Angular Velocity (rad/s) --\n";
    oss << "Yaw_vel   : " << aim.yaw_vel << "\n";
    oss << "Pitch_vel : " << aim.pitch_vel << "\n";
    oss << "Roll_vel  : " << aim.roll_vel << "\n";

    oss << "\n-- Orientation (deg) --\n";
    oss << "Yaw   : " << aim.yaw * 180 / M_PI << "\n";
    oss << "Pitch : " << aim.pitch * 180 / M_PI << "\n";
    oss << "Roll  : " << aim.roll * 180 / M_PI << "\n";

    oss << "\n-- System Info --\n";
    oss << "Bullet Speed     : " << aim.bullet_speed << " m/s\n";
    oss << "Controller Delay : " << aim.controller_delay << " s\n";
    oss << "Detect Color     : " << (aim.detect_color == 0 ? "Red" : "Blue") << "\n";

    oss << "Frame Rate (FPS) : " << std::setprecision(1) << fps << "\n";

    return oss.str();
}
void dumpAimToFile(const ReceiveAimINFO& aim, const std::string& path) {
    std::ofstream file(path);
    if (file.is_open()) {
        file << formatAimInfo(aim);
        file.close();
    }
}
void writeAimLogToJson(const ReceiveAimINFO& aim) {
    nlohmann::json j;

    j["timestamp"] = aim.time_stamp;
    j["yaw"] = aim.yaw;
    j["pitch"] = aim.pitch;
    j["roll"] = aim.roll;

    j["yaw_vel"] = aim.yaw_vel;
    j["pitch_vel"] = aim.pitch_vel;
    j["roll_vel"] = aim.roll_vel;

    j["manual_reset_count"] = aim.manual_reset_count;
    j["bullet_speed"] = aim.bullet_speed;
    j["controller_delay"] = aim.controller_delay;
    j["detect_color"] = (aim.detect_color == 0 ? "Red" : "Blue");

    // FPS 统计
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

    std::ofstream file("/dev/shm/aim_log.json");
    if (file.is_open()) {
        file << j.dump(2);
    }
}
void writeTargetLogToJson(const armor::Target& target) {
    nlohmann::json j;

    j["frame_id"] = target.frame_id;
    j["type"] = target.type;
    j["tracking"] = target.tracking;
    j["id"] = static_cast<int>(target.id);
    j["armors_num"] = target.armors_num;

    auto now = std::chrono::steady_clock::now();
    auto age_ms =
        std::chrono::duration_cast<std::chrono::milliseconds>(now - target.timestamp).count();
    j["timestamp_age_ms"] = age_ms;

    j["position"] = { { "x", target.position_.x },
                      { "y", target.position_.y },
                      { "z", target.position_.z } };

    j["velocity"] = { { "x", target.velocity_.x },
                      { "y", target.velocity_.y },
                      { "z", target.velocity_.z } };

    j["yaw"] = target.yaw;
    j["v_yaw"] = target.v_yaw;
    j["yaw_diff"] = target.yaw_diff;

    j["radius_1"] = target.radius_1;
    j["radius_2"] = target.radius_2;
    j["d_za"] = target.d_za;
    j["d_zc"] = target.d_zc;
    j["position_diff"] = target.position_diff;
    j["z_diff"] = std::abs(target.d_za) + std::abs(target.d_zc);

    std::ofstream file("/dev/shm/target_log.json");
    if (file.is_open()) {
        file << j.dump(2);
    }
}

std::string GetUniqueVideoFilename(const std::string& folder, const std::string& prefix) {
    auto now = std::chrono::system_clock::now();
    std::time_t now_c = std::chrono::system_clock::to_time_t(now);
    std::tm* tm_ptr = std::localtime(&now_c);

    std::ostringstream oss;
    oss << folder << "/" << prefix << "_" << std::put_time(tm_ptr, "%Y%m%d_%H%M%S") << ".avi";
    return oss.str();
}
cv::Point2f normalize(const cv::Point2f& v) {
    float norm = std::sqrt(v.x * v.x + v.y * v.y);
    if (norm > 1e-6f)
        return v / norm;
    else
        return cv::Point2f(0, 0);
}
void writeCmdLogToJson() {
    nlohmann::json j;
    {
        std::lock_guard<std::mutex> lock(toolsgobal::robot_cmd_mutex_);
        auto log = toolsgobal::debug_logs_;

        j["time"] = log.time_log;
        j["yaw"] = log.cmd_yaw_log;
        j["pitch"] = log.cmd_pitch_log;
        j["armor_dis"] = log.armor_dis_log;
        j["armor_x"] = log.armor_x_log;
        j["armor_y"] = log.armor_y_log;
        j["armor_z"] = log.armor_z_log;
        j["armor_yaw"] = log.armor_yaw_log;
        j["ypd_y"] = log.ypd_y_log;
        j["ypd_p"] = log.ypd_p_log;
        j["rune_obs"] = log.rune_obs_log;
        j["rune_pre"] = log.rune_pre_log;
    }
    std::ofstream file("/dev/shm/cmd_log.json");
    if (file.is_open()) {
        file << j.dump();
    }
}

void robotCmdLoggerThread() {
    while (!gobal::is_inited_) {
        usleep(10000);
    }
    while (gobal::is_inited_) {
        writeCmdLogToJson();
        std::this_thread::sleep_for(std::chrono::milliseconds(50)); // 20Hz
    }
}
void drawDebugOverlayWrite(const DebugArmor& dbg, bool auto_fps) {
    static auto last_show_time = std::chrono::steady_clock::now();

    if (dbg.src_img->img.empty())
        return;
    cv::Mat src_img = dbg.src_img->img;
    auto now = std::chrono::steady_clock::now();
    const double min_interval_ms = 1000.0 / toolsgobal::debug_fps;
    if (std::chrono::duration<double, std::milli>(now - last_show_time).count() < min_interval_ms
        && auto_fps)
        return;
    last_show_time = now;

    // 图像构造
    cv::Mat debug_img;
    src_img.convertTo(debug_img, -1, 1, 0);
    cv::cvtColor(debug_img, debug_img, cv::COLOR_BGR2RGB);
    if (debug_img.empty())
        return;

    // 封装后的绘图函数
    drawDebugArmorContent(debug_img, dbg);

    // 编码写入共享内存路径
    std::vector<uchar> buf;
    cv::imencode(".jpg", debug_img, buf);
    std::ofstream ofs("/dev/shm/debug_frame.jpg.tmp", std::ios::binary);
    ofs.write(reinterpret_cast<const char*>(buf.data()), buf.size());
    ofs.close();
    std::rename("/dev/shm/debug_frame.jpg.tmp", "/dev/shm/debug_frame.jpg");
}
void drawDebugOverlayShow(const DebugArmor& dbg, bool auto_fps) {
    static auto last_show_time = std::chrono::steady_clock::now();

    if (dbg.src_img->img.empty())
        return;
    cv::Mat src_img = dbg.src_img->img;
    auto now = std::chrono::steady_clock::now();
    const double min_interval_ms = 1000.0 / toolsgobal::debug_fps;
    if (std::chrono::duration<double, std::milli>(now - last_show_time).count() < min_interval_ms
        && auto_fps)
        return;
    last_show_time = now;

    // 图像构造
    cv::Mat debug_img;
    src_img.convertTo(debug_img, -1, 1, 0);
    cv::cvtColor(debug_img, debug_img, cv::COLOR_BGR2RGB);
    if (debug_img.empty())
        return;

    // 封装后的绘图函数
    drawDebugArmorContent(debug_img, dbg);

    cv::imshow("debug_armor", debug_img);
    cv::waitKey(1);
}
cv::Mat drawDebugOverlayMat(const DebugArmor& dbg) {
    static auto last_show_time = std::chrono::steady_clock::now();

    if (dbg.src_img->img.empty())
        return cv::Mat();
    cv::Mat src_img = dbg.src_img->img;
    auto now = std::chrono::steady_clock::now();
    const double min_interval_ms = 1000.0 / toolsgobal::debug_fps;
    if (std::chrono::duration<double, std::milli>(now - last_show_time).count() < min_interval_ms)
        return cv::Mat();
    last_show_time = now;

    // 图像构造
    cv::Mat debug_img;
    src_img.convertTo(debug_img, -1, 1, 0);
    cv::cvtColor(debug_img, debug_img, cv::COLOR_BGR2RGB);
    if (debug_img.empty())
        return cv::Mat();

    // 封装后的绘图函数
    drawDebugArmorContent(debug_img, dbg);

    return debug_img;
}
void drawDebugOverlayWrite(const DebugRune& dbg, bool auto_fps) {
    static auto last_show_time = std::chrono::steady_clock::now();

    if (dbg.src_img->img.empty())
        return;
    cv::Mat src_img = dbg.src_img->img;
    auto now = std::chrono::steady_clock::now();
    const double min_interval_ms = 1000.0 / toolsgobal::debug_fps;
    if (std::chrono::duration<double, std::milli>(now - last_show_time).count() < min_interval_ms
        && auto_fps)
        return;
    last_show_time = now;

    // 图像构造
    cv::Mat debug_img;
    src_img.convertTo(debug_img, -1, 1, 0);
    cv::cvtColor(debug_img, debug_img, cv::COLOR_BGR2RGB);
    if (debug_img.empty())
        return;

    // 封装后的绘图函数
    drawDebugRuneContent(debug_img, dbg);

    // 编码写入共享内存路径
    std::vector<uchar> buf;
    cv::imencode(".jpg", debug_img, buf);
    std::ofstream ofs("/dev/shm/debug_frame.jpg.tmp", std::ios::binary);
    ofs.write(reinterpret_cast<const char*>(buf.data()), buf.size());
    ofs.close();
    std::rename("/dev/shm/debug_frame.jpg.tmp", "/dev/shm/debug_frame.jpg");
}
void drawDebugOverlayShow(const DebugRune& dbg, bool auto_fps) {
    static auto last_show_time = std::chrono::steady_clock::now();

    if (dbg.src_img->img.empty())
        return;
    cv::Mat src_img = dbg.src_img->img;
    auto now = std::chrono::steady_clock::now();
    const double min_interval_ms = 1000.0 / toolsgobal::debug_fps;
    if (std::chrono::duration<double, std::milli>(now - last_show_time).count() < min_interval_ms
        && auto_fps)
        return;
    last_show_time = now;

    // 图像构造
    cv::Mat debug_img;
    src_img.convertTo(debug_img, -1, 1, 0);
    cv::cvtColor(debug_img, debug_img, cv::COLOR_BGR2RGB);
    if (debug_img.empty())
        return;

    // 封装后的绘图函数
    drawDebugRuneContent(debug_img, dbg);

    cv::imshow("debug_rune", debug_img);
    cv::waitKey(1);
}
cv::Mat drawDebugOverlayMat(const DebugRune& dbg) {
    static auto last_show_time = std::chrono::steady_clock::now();

    if (dbg.src_img->img.empty())
        return cv::Mat();
    cv::Mat src_img = dbg.src_img->img;
    auto now = std::chrono::steady_clock::now();
    const double min_interval_ms = 1000.0 / toolsgobal::debug_fps;
    if (std::chrono::duration<double, std::milli>(now - last_show_time).count() < min_interval_ms)
        return cv::Mat();
    last_show_time = now;

    // 图像构造
    cv::Mat debug_img;
    src_img.convertTo(debug_img, -1, 1, 0);
    cv::cvtColor(debug_img, debug_img, cv::COLOR_BGR2RGB);
    if (debug_img.empty())
        return cv::Mat();

    // 封装后的绘图函数
    drawDebugRuneContent(debug_img, dbg);

    return debug_img;
}
