#pragma once
#include "tasks/auto_aim/armor_tracker/target.hpp"
#include "tasks/auto_aim/auto_aim_fsm.hpp"
#include "tasks/auto_aim/type.hpp"
#include "tasks/utils/debug_utils.hpp"
namespace wust_vision::auto_aim {

struct AutoAimDebug {
    wust_vl::video::ImageFrame img_frame;
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
    std::pair<double, double> gimbal_py;
};
inline void drawDebugArmorContent(
    cv::Mat& debug_img,
    const AutoAimDebug& dbg,
    std::pair<cv::Mat, cv::Mat> camera_info
) {
    if (debug_img.empty()) {
        std::cout << "debug_img is empty" << std::endl;
        return;
    }
    const auto now = std::chrono::steady_clock::now();
    const auto& armors = dbg.armors;
    const auto& gimbal_cmd = dbg.gimbal_cmd;
    const auto& target = dbg.target;
    auto aim_target = dbg.aim_target;
    const auto& armor_objs = dbg.armor_objs;
    const cv::Rect img_rect(0, 0, debug_img.cols, debug_img.rows);
    const cv::Rect roi = dbg.expanded & img_rect;
    cv::rectangle(debug_img, roi, cv::Scalar(255, 255, 255), 2);

    static const int next_indices[] = { 2, 0, 3, 1 };

    for (size_t i = 0; i < armor_objs.size(); i++) {
        const auto pts = armor_objs[i].toPts();

        for (size_t j = 0; j < 4; ++j) {
            const cv::Scalar color =
                armor_objs[i].is_ok ? cv::Scalar(50, 255, 50) : cv::Scalar(50, 255, 255);
            cv::line(debug_img, pts[j], pts[next_indices[j]], color, 2);
        }

        auto armorName = [](auto_aim::ArmorNumber num) {
            switch (num) {
                case auto_aim::ArmorNumber::SENTRY:
                    return "SENTRY";
                case auto_aim::ArmorNumber::BASE:
                    return "BASE";
                case auto_aim::ArmorNumber::OUTPOST:
                    return "OUTPOST";
                case auto_aim::ArmorNumber::NO1:
                    return "NO1";
                case auto_aim::ArmorNumber::NO2:
                    return "NO2";
                case auto_aim::ArmorNumber::NO3:
                    return "NO3";
                case auto_aim::ArmorNumber::NO4:
                    return "NO4";
                case auto_aim::ArmorNumber::NO5:
                    return "NO5";
                default:
                    return "UNKNOWN";
            }
        };

        const std::string armor_str = armorName(armor_objs[i].number);
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

    static std::deque<std::pair<Eigen::Vector3d, double>> traj3d;

    double _now =
        std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();

    traj3d.emplace_back(aim_target.pos, _now);

    while (!traj3d.empty() && _now - traj3d.front().second > 1.0)
        traj3d.pop_front();
    aim_target.tf(dbg.T_camera_to_odom.inverse());
    if (!aim_target.is_old) {
        const auto pts = aim_target.toPts(camera_info.first, camera_info.second);

        if (!pts.empty()) {
            if (traj3d.size() > 1) {
                std::vector<std::pair<cv::Point, double>> img_pts;

                for (auto& p: traj3d) {
                    auto p3d_odom = p.first;

                    Eigen::Vector4d p_odom(p3d_odom.x(), p3d_odom.y(), p3d_odom.z(), 1);

                    Eigen::Vector4d p_camera = dbg.T_camera_to_odom.inverse() * p_odom;

                    std::vector<cv::Point3f> obj;
                    obj.emplace_back(p_camera.x(), p_camera.y(), p_camera.z());

                    std::vector<cv::Point2f> proj;

                    cv::projectPoints(
                        obj,
                        cv::Vec3d(0, 0, 0),
                        cv::Vec3d(0, 0, 0),
                        camera_info.first,
                        camera_info.second,
                        proj
                    );

                    if (!proj.empty()) {
                        const auto& pt = proj[0];

                        if (std::isfinite(pt.x) && std::isfinite(pt.y)) {
                            img_pts.emplace_back(cv::Point(int(pt.x), int(pt.y)), p.second);
                        }
                    }
                }

                if (img_pts.size() >= 2) {
                    double now = std::chrono::duration<double>(
                                     std::chrono::steady_clock::now().time_since_epoch()
                    )
                                     .count();

                    const double max_age = 1.0;

                    for (size_t i = 1; i < img_pts.size(); ++i) {
                        double age = now - img_pts[i].second;

                        double t = std::clamp(age / max_age, 0.0, 1.0);

                        int r = int(255 * (1.0 - t));
                        int b = int(255 * t);

                        cv::Scalar color(b, 0, r);

                        cv::line(
                            debug_img,
                            img_pts[i - 1].first,
                            img_pts[i].first,
                            color,
                            2,
                            cv::LINE_AA
                        );
                    }
                }
            }

            cv::Point2f center(0.f, 0.f);

            for (auto pt: pts)
                center += pt;

            center *= 1.0f / pts.size();

            cv::Scalar color(255, 255, 255);

            for (int i = 0; i < 4; i++)
                cv::line(debug_img, pts[i], pts[(i + 1) % 4], color, 2);

            for (int i = 4; i < 8; i++)
                cv::line(debug_img, pts[i], pts[4 + (i + 1) % 4], color, 2);

            for (int i = 0; i < 4; i++)
                cv::line(debug_img, pts[i], pts[i + 4], color, 2);

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

            const cv::Scalar color_x =
                dbg.detect_color ? cv::Scalar(255, 50, 50) : cv::Scalar(50, 50, 255);

            cv::arrowedLine(debug_img, start_pt, end_pt, color_x, 4, cv::LINE_AA, 0, 0.2);
        }
    }
    std::vector<cv::Point2f> all_corners;

    auto visualizeTargetProjection = [&](auto_aim::Target armor_target) -> auto_aim::Armors {
        auto_aim::Armors armor_data;
        armor_data.timestamp = armor_target.timestamp_;

        if (armor_target.is_tracking) {
            Eigen::Vector3d pos = armor_target.target_state_.pos();
            if (pos.norm() > 0.5) {
                armor_data.armors.clear();
                const size_t a_n = armor_target.armor_num_;
                armor_data.armors.reserve(a_n);
                const auto now = wust_vl::common::utils::time_utils::now();
                armor_target.predictSimple(now);
                const std::vector<Eigen::Vector4d> armors_posandyaw =
                    armor_target.getArmorPosAndYaw();
                for (size_t i = 0; i < a_n; ++i) {
                    const Eigen::Vector3d pos = { armors_posandyaw[i][0],
                                                  armors_posandyaw[i][1],
                                                  armors_posandyaw[i][2] };
                    Eigen::Vector3d euler;
                    euler.z() = M_PI / 2.0;
                    euler.y() = (armor_target.tracked_id_ == auto_aim::ArmorNumber::OUTPOST)
                        ? -0.2618
                        : 0.2618;
                    euler.x() = armors_posandyaw[i][3];
                    const Eigen::Quaterniond ori =
                        utils::eulerToQuat(euler, utils::EulerOrder::ZYX);
                    armor_data.armors.emplace_back(auto_aim::Armor {
                        .type = armor_target.type_,
                        .pos = pos,
                        .ori = ori,
                        .is_ok = true,
                        .id = (int)(i),
                    });
                }
            }
        }
        return armor_data;
    };
    auto armor_data = visualizeTargetProjection(dbg.target);
    transformArmorData(armor_data, dbg.T_camera_to_odom.inverse());
    for (size_t i = 0; i < armor_data.armors.size(); ++i) {
        const auto& pts = armor_data.armors[i].toPtsDebug(camera_info.first, camera_info.second);
        const auto& pos = armor_data.armors[i].pos;
        const auto& ori = armor_data.armors[i].ori;
        const auto& id = armor_data.armors[i].id;
        cv::Scalar color;
        if (dbg.detect_color) {
            color = cv::Scalar(255, 0, 0);
        } else {
            color = cv::Scalar(0, 0, 255);
        }
        // 绘制前表面
        for (size_t j = 0; j < 4; ++j) {
            cv::line(debug_img, pts[j], pts[(j + 1) % 4], color, 2);
        }

        // 绘制后表面
        for (size_t j = 4; j < 8; ++j) {
            cv::line(debug_img, pts[j], pts[4 + (j + 1) % 4], color, 2);
        }

        // 绘制侧边
        for (size_t j = 0; j < 4; ++j) {
            cv::line(debug_img, pts[j], pts[j + 4], color, 2);
        }

        all_corners.insert(all_corners.end(), pts.begin(), pts.end());

        const Eigen::Vector3d euler = ori.toRotationMatrix().eulerAngles(2, 1, 0);
        const double yaw = euler[0];
        const double distance =
            std::sqrt(pos.x() * pos.x() + pos.y() * pos.y() + pos.z() * pos.z());

        const std::vector<std::string> info_lines = {
            fmt::format("Dis: {:.1f}cm", distance * 100),
            fmt::format("X: {:.2f}", pos.x()),
            fmt::format("Y: {:.2f}", pos.y()),
            fmt::format("Z: {:.2f}", pos.z()),
            fmt::format("Yaw: {:.2f}", yaw * 180.0 / M_PI),
            fmt::format("ID: {:d}", id)
        };

        const cv::Point2f text_org = pts[0] + cv::Point2f(0, 200);
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

    if (!all_corners.empty()) {
        cv::Point2f avg(0.f, 0.f);
        for (const auto& pt: all_corners)
            avg += pt;
        avg *= 1.0f / all_corners.size();
        cv::circle(debug_img, avg, 10, cv::Scalar(50, 255, 50), -1);

        const double scale = 50.0;
        const double dy = scale * target.target_state_.vyaw();
        const cv::Point2f start_pt = avg;
        const cv::Point2f end_pt = start_pt + cv::Point2f(0, dy);
        cv::arrowedLine(
            debug_img,
            start_pt,
            end_pt,
            cv::Scalar(50, 255, 50),
            3,
            cv::LINE_AA,
            0,
            0.1
        );
        cv::putText(
            debug_img,
            fmt::format("V_yaw: {:.2f}", target.target_state_.vyaw()),
            avg + cv::Point2f(0, -20),
            cv::FONT_HERSHEY_SIMPLEX,
            1.0,
            cv::Scalar(50, 255, 50),
            2
        );
    }

    std::string state_str;
    state_str = auto_aim_fsm_to_string(dbg.fsm);

    int baseline = 0;
    cv::Size text_size = cv::getTextSize(state_str, cv::FONT_HERSHEY_SIMPLEX, 2.5, 2, &baseline);

    // 保证在图像内
    const int x =
        std::clamp(debug_img.cols - text_size.width - 10, 0, debug_img.cols - text_size.width);
    const int y = std::clamp(text_size.height + 10, text_size.height, debug_img.rows - 1);

    cv::putText(
        debug_img,
        state_str,
        { x, y },
        cv::FONT_HERSHEY_SIMPLEX,
        2.5,
        cv::Scalar(0, 0, 255),
        2
    );

    const std::string id_str =
        fmt::format("Attack: {}", armorNumberToString(dbg.target.tracked_id_));
    const cv::Size id_size = cv::getTextSize(id_str, cv::FONT_HERSHEY_SIMPLEX, 1.6, 2, &baseline);

    // 保证在图像内
    const int id_x = std::clamp(debug_img.cols - 300, 0, debug_img.cols - id_size.width - 10);
    const int id_y = std::clamp(150, id_size.height, debug_img.rows - 1);

    cv::putText(
        debug_img,
        id_str,
        { id_x, id_y },
        cv::FONT_HERSHEY_SIMPLEX,
        1.6,
        cv::Scalar(255, 0, 255),
        2
    );

    if (gimbal_cmd.fire_advice) {
        std::string fire_str = "Fire!";
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

    double scale = 100.0;
    double armor_len = 0.135;

    std::vector<Eigen::Vector2d> pts;
    pts.reserve(armors.armors.size() + armor_data.armors.size());

    auto collect_xy = [&](auto& list, bool use_target) {
        for (auto& a: list)
            pts.emplace_back(
                use_target ? a.target_pos.x() : a.pos.x(),
                use_target ? a.target_pos.y() : a.pos.y()
            );
    };

    collect_xy(armors.armors, true);
    collect_xy(armor_data.armors, false);

    double max_abs_x = 1e-6, max_abs_y = 1e-6;
    for (auto& p: pts) {
        max_abs_x = std::max(max_abs_x, std::abs(p.x()));
        max_abs_y = std::max(max_abs_y, std::abs(p.y()));
    }

    const double margin = 200.0;
    const double cx = debug_img.cols * 0.5;
    const double cy = debug_img.rows * 0.5;

    scale = std::min({ (cx - margin) / max_abs_x,
                       (debug_img.cols - cx - margin) / max_abs_x,
                       (cy - margin) / max_abs_y,
                       (debug_img.rows - cy - margin) / max_abs_y,
                       550.0 });

    const cv::Point2d origin(cx, cy);

    auto to_img = [&](const Eigen::Vector3d& p) {
        return cv::Point2d(origin.x + p.x() * scale, origin.y - p.y() * scale);
    };

    auto draw2dArmor = [&](const Eigen::Vector3d& pos, double yaw, const cv::Scalar& color) {
        cv::Point2d C = to_img(pos);
        cv::circle(debug_img, C, 3, color, -1, cv::LINE_AA);

        double nx = -sin(yaw), ny = cos(yaw);
        double half_len_px = armor_len * 0.5 * scale;

        cv::Point2d P1(C.x + nx * half_len_px, C.y - ny * half_len_px);
        cv::Point2d P2(C.x - nx * half_len_px, C.y + ny * half_len_px);
        cv::line(debug_img, P1, P2, color, 2, cv::LINE_AA);
    };

    Eigen::Vector3d center(0, 0, 0);
    if (!armor_data.armors.empty()) {
        for (auto& a: armor_data.armors)
            center += a.pos;
        center /= armor_data.armors.size();
    }

    const cv::Point2d Cc = to_img(center);
    if (!armor_data.armors.empty())
        cv::circle(debug_img, Cc, 5, cv::Scalar(255, 0, 0), -1, cv::LINE_AA);

    for (auto& a: armors.armors)
        draw2dArmor(a.target_pos, a.yaw, cv::Scalar(0, 255, 255));

    std::vector<cv::Point2d> data_pts;

    for (auto& a: armor_data.armors) {
        double yaw = a.ori.toRotationMatrix().eulerAngles(2, 1, 0)[0];
        draw2dArmor(a.pos, yaw, cv::Scalar(255, 255, 255));
        data_pts.push_back(to_img(a.pos));
    }

    for (auto& pt: data_pts)
        cv::line(debug_img, Cc, pt, cv::Scalar(180, 180, 255), 1, cv::LINE_AA);

    for (auto& a: armors.armors)
        cv::line(debug_img, Cc, to_img(a.target_pos), cv::Scalar(0, 150, 255), 1, cv::LINE_AA);

    cv::circle(
        debug_img,
        cv::Point2i(debug_img.cols / 2, debug_img.rows / 2),
        5,
        cv::Scalar(255, 255, 255),
        2
    );
}
inline void writeTargetLogToJson(const auto_aim::Target& armor_target) {
    nlohmann::json j;

    // -------- armor_target 部分 --------
    nlohmann::json jt;
    jt["type"] = armor_target.type_;
    jt["tracking"] = armor_target.is_tracking;
    jt["id"] = static_cast<int>(armor_target.tracked_id_);
    jt["armors_num"] = armor_target.armor_num_;

    const auto now = std::chrono::steady_clock::now();
    const auto age_ms_t =
        std::chrono::duration_cast<std::chrono::milliseconds>(now - armor_target.timestamp_)
            .count();
    jt["timestamp_age_ms"] = age_ms_t;

    jt["position"] = { { "x", armor_target.target_state_.cx() },
                       { "y", armor_target.target_state_.cy() },
                       { "z", armor_target.target_state_.cz() } };

    jt["velocity"] = { { "x", armor_target.target_state_.vcx() },
                       { "y", armor_target.target_state_.vcy() },
                       { "z", armor_target.target_state_.vcz() } };

    jt["r"] = armor_target.target_state_.r();
    jt["l"] = armor_target.target_state_.l();
    jt["h"] = armor_target.target_state_.h();
    jt["yaw"] = armor_target.target_state_.yaw();
    jt["v_yaw"] = armor_target.target_state_.vyaw();
    j["armor_target"] = jt;
    // -------- 写文件 --------
    std::ofstream file("/dev/shm/target_log.json");
    if (file.is_open()) {
        file << j.dump(2);
    }
}

inline void drawDebugOverlayShm(
    const AutoAimDebug& dbg,
    std::pair<cv::Mat, cv::Mat> camera_info,
    bool auto_fps
) {
    static ShmWriter shm { "/debug_frame" };
    drawDebugOverlayImpl(dbg, camera_info, auto_fps, drawDebugArmorContent, shm);
}
void debuglog(const AutoAimDebug& dbg_armor);
} // namespace wust_vision::auto_aim