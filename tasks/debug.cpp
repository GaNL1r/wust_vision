#include "tasks/debug.hpp"
#include "tasks/utils.hpp"
#include <arpa/inet.h>
#include <fcntl.h>
#include <nlohmann/json.hpp>
#include <sys/mman.h>
#include <sys/socket.h>
#include <unistd.h>
#include <wust_vl/common/utils/timer.hpp>
void drawDebugArmorContent(
    cv::Mat& debug_img,
    const DebugArmor& dbg,
    std::pair<cv::Mat, cv::Mat> camera_info
) {
    if (debug_img.empty()) {
        std::cout << "debug_img is empty" << std::endl;
        return;
    }
    auto now = std::chrono::steady_clock::now();
    auto armors = dbg.armors;
    auto gimbal_cmd = dbg.gimbal_cmd;
    auto target = dbg.target;
    auto aim_target = dbg.aim_target;
    auto armor_objs = dbg.armor_objs;
    cv::Rect img_rect(0, 0, debug_img.cols, debug_img.rows);
    cv::Rect roi = dbg.expanded & img_rect;
    cv::rectangle(debug_img, roi, cv::Scalar(255, 255, 255), 2);

    static const int next_indices[] = { 2, 0, 3, 1 };

    // =================== 绘制装甲板 ===================
    for (size_t i = 0; i < armor_objs.size(); i++) {
        auto pts = armor_objs[i].toPts();

        for (size_t j = 0; j < 4; ++j) {
            cv::Scalar color =
                armor_objs[i].is_ok ? cv::Scalar(50, 255, 50) : cv::Scalar(50, 255, 255);
            cv::line(debug_img, pts[j], pts[next_indices[j]], color, 2);
        }

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

        std::string armor_str = armorName(armor_objs[i].number);
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

    std::string latency_str = fmt::format("Latency: {:.2f}ms", dbg.latency_ms);
    cv::putText(
        debug_img,
        latency_str,
        cv::Point(10, 30),
        cv::FONT_HERSHEY_SIMPLEX,
        0.8,
        cv::Scalar(255, 255, 255),
        2
    );

    // =================== 目标绘制 ===================
    std::vector<cv::Point2f> all_corners;

    auto visualizeTargetProjection = [&](Target armor_target) -> armor::Armors {
        armor::Armors armor_data;
        armor_data.frame_id = "gimbal_odom";
        armor_data.timestamp = armor_target.timestamp_;
        double debug_dt = 0.01;

        if (armor_target.is_tracking) {
            Eigen::Vector3d pos = armor_target.position();
            Eigen::Vector3d vel = armor_target.velocity();
            utils::addPosFromVelDt(pos, vel, debug_dt);
            if (pos.norm() > 0.5) {
                armor_data.armors.clear();
                size_t a_n = armor_target.armor_num_;
                armor_data.armors.reserve(a_n);
                auto now = time_utils::now();
                double dt0 = time_utils::durationSec(armor_target.timestamp_, now);
                std::chrono::steady_clock::time_point future =
                    now + std::chrono::microseconds(int(dt0 * 1e6));
                armor_target.predict(future);
                std::vector<Eigen::Vector4d> armors_posandyaw = armor_target.getArmorPosAndYaw();
                for (size_t i = 0; i < a_n; ++i) {
                    Eigen::Vector3d pos = { armors_posandyaw[i][0],
                                            armors_posandyaw[i][1],
                                            armors_posandyaw[i][2] };
                    Eigen::Vector3d euler;
                    euler.z() = M_PI / 2.0;
                    euler.y() = (armor_target.tracked_id_ == armor::ArmorNumber::OUTPOST) ? -0.2618
                                                                                          : 0.2618;
                    euler.x() = armors_posandyaw[i][3];
                    Eigen::Quaterniond ori = utils::eulerToQuat(euler, utils::EulerOrder::ZYX);
                    armor_data.armors.emplace_back(armor::Armor {
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
    armor::transformArmorData(armor_data, dbg.T_camera_to_odom.inverse());
    for (size_t i = 0; i < armor_data.armors.size(); ++i) {
        const auto& pts = armor_data.armors[i].toPtsDebug(camera_info.first, camera_info.second);
        const auto& pos = armor_data.armors[i].pos;
        const auto& ori = armor_data.armors[i].ori;
        const auto& is_ok = armor_data.armors[i].is_ok;
        const auto& id = armor_data.armors[i].id;
        cv::Scalar color;
        if (dbg.detect_color) {
            color = is_ok ? cv::Scalar(255, 0, 0) : cv::Scalar(0, 0, 255);
        } else {
            color = !is_ok ? cv::Scalar(255, 0, 0) : cv::Scalar(0, 0, 255);
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

        if (is_ok) {
            all_corners.insert(all_corners.end(), pts.begin(), pts.end());
        }
        Eigen::Vector3d euler = ori.toRotationMatrix().eulerAngles(2, 1, 0);
        double yaw = euler[0];
        double distance = std::sqrt(pos.x() * pos.x() + pos.y() * pos.y() + pos.z() * pos.z());

        std::vector<std::string> info_lines = { fmt::format("Dis: {:.1f}cm", distance * 100),
                                                fmt::format("X: {:.2f}", pos.x()),
                                                fmt::format("Y: {:.2f}", pos.y()),
                                                fmt::format("Z: {:.2f}", pos.z()),
                                                fmt::format("Yaw: {:.2f}", yaw * 180.0 / M_PI),
                                                fmt::format("ID: {:d}", id) };

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
    aim_target.tf(dbg.T_camera_to_odom.inverse());
    if (!aim_target.is_old) {
        auto pts = aim_target.toPts(camera_info.first, camera_info.second);
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

            double scale = 10.0;
            double v_yaw = gimbal_cmd.v_yaw;
            double v_pitch = gimbal_cmd.v_pitch;
            double dx = -scale * v_yaw;
            double dy = scale * v_pitch;

            cv::Point2f start_pt = center;
            cv::Point2f end_pt = start_pt + cv::Point2f(dx, dy);
            cv::Scalar color_x =
                dbg.detect_color ? cv::Scalar(255, 50, 50) : cv::Scalar(50, 50, 255);
            cv::arrowedLine(debug_img, start_pt, end_pt, color_x, 4, cv::LINE_AA, 0, 0.2);
        }
    }

    if (!all_corners.empty()) {
        cv::Point2f avg(0.f, 0.f);
        for (const auto& pt: all_corners)
            avg += pt;
        avg *= 1.0f / all_corners.size();
        cv::circle(debug_img, avg, 10, cv::Scalar(50, 255, 50), -1);

        double scale = 50.0;
        double dy = scale * target.v_yaw();
        cv::Point2f start_pt = avg;
        cv::Point2f end_pt = start_pt + cv::Point2f(0, dy);
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
            fmt::format("V_yaw: {:.2f}", target.v_yaw()),
            avg + cv::Point2f(0, -20),
            cv::FONT_HERSHEY_SIMPLEX,
            1.0,
            cv::Scalar(50, 255, 50),
            2
        );
    }

    // =================== 状态绘制 ===================
    std::string state_str;
    state_str = auto_aim_fsm_to_string(dbg.fsm);

    int baseline = 0;
    cv::Size text_size = cv::getTextSize(state_str, cv::FONT_HERSHEY_SIMPLEX, 2.5, 2, &baseline);

    // 保证在图像内
    int x = std::clamp(debug_img.cols - text_size.width - 10, 0, debug_img.cols - text_size.width);
    int y = std::clamp(text_size.height + 10, text_size.height, debug_img.rows - 1);

    cv::putText(
        debug_img,
        state_str,
        { x, y },
        cv::FONT_HERSHEY_SIMPLEX,
        2.5,
        cv::Scalar(0, 0, 255),
        2
    );

    // =================== Attack ID ===================
    std::string id_str = fmt::format("Attack: {}", armorNumberToString(dbg.target.tracked_id_));
    cv::Size id_size = cv::getTextSize(id_str, cv::FONT_HERSHEY_SIMPLEX, 1.6, 2, &baseline);

    // 保证在图像内
    int id_x = std::clamp(debug_img.cols - 300, 0, debug_img.cols - id_size.width - 10);
    int id_y = std::clamp(150, id_size.height, debug_img.rows - 1);

    cv::putText(
        debug_img,
        id_str,
        { id_x, id_y },
        cv::FONT_HERSHEY_SIMPLEX,
        1.6,
        cv::Scalar(255, 0, 255),
        2
    );

    // =================== Fire 标志 ===================
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

    // =================== 云台指令 ===================
    std::string gimbal_str = fmt::format(
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

    double margin = 200.0;
    double cx = debug_img.cols * 0.5;
    double cy = debug_img.rows * 0.5;

    scale = std::min({ (cx - margin) / max_abs_x,
                       (debug_img.cols - cx - margin) / max_abs_x,
                       (cy - margin) / max_abs_y,
                       (debug_img.rows - cy - margin) / max_abs_y,
                       550.0 });

    cv::Point2d origin(cx, cy);

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

    cv::Point2d Cc = to_img(center);
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
static cv::Point2f normalize(const cv::Point2f& v) {
    float norm = std::sqrt(v.x * v.x + v.y * v.y);
    if (norm > 1e-6f)
        return v / norm;
    else
        return cv::Point2f(0, 0);
}
void drawDebugRuneContent(
    cv::Mat& debug_img,
    const DebugRune& dbg,
    std::pair<cv::Mat, cv::Mat> camera_info
) {
    const auto& gimbal_cmd = dbg.gimbal_cmd;
    double predict_angle = dbg.predict_angle;
    const auto& debug_text = dbg.debug_text;
    auto aim_target = dbg.aim_target;
    auto rune = dbg.power_rune;
    cv::Rect img_rect(0, 0, debug_img.cols, debug_img.rows);
    cv::Rect roi = dbg.expanded & img_rect;
    cv::rectangle(debug_img, roi, cv::Scalar(255, 255, 255), 2);

    std::string latency_str = fmt::format("Latency: {:.2f}ms", dbg.latency_ms);
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
        auto pts = aim_target.toPts(camera_info.first, camera_info.second);
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

            double scale = 10.0;
            double v_yaw = gimbal_cmd.v_yaw;
            double v_pitch = gimbal_cmd.v_pitch;
            double dx = -scale * v_yaw;
            double dy = scale * v_pitch;

            cv::Point2f start_pt = center;
            cv::Point2f end_pt = start_pt + cv::Point2f(dx, dy);
            cv::Scalar color_x = cv::Scalar(50, 50, 255);
            cv::arrowedLine(debug_img, start_pt, end_pt, color_x, 4, cv::LINE_AA, 0, 0.2);
        }
    }
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

    // =================== 云台指令 ===================
    std::string gimbal_str = fmt::format(
        "Pitch: {:.2f}, Yaw: {:.2f}, Pitch_diff: {:.2f}, Yaw_diff: {:.2f}, V_yaw: {:.2f}, V_pitch: {:.2f}",
        gimbal_cmd.pitch,
        gimbal_cmd.yaw,
        gimbal_cmd.pitch_diff,
        gimbal_cmd.yaw_diff,
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
    rune.tf(dbg.T_camera_to_odom.inverse());
    rune.draw(debug_img, camera_info.first, camera_info.second);
    cv::circle(
        debug_img,
        cv::Point2i(debug_img.cols / 2, debug_img.rows / 2),
        5,
        cv::Scalar(255, 255, 255),
        2
    );
}
void drawDebugOverlayWrite(
    const DebugArmor& dbg,
    std::pair<cv::Mat, cv::Mat> camera_info,
    bool auto_fps
) {
    static auto last_show_time = std::chrono::steady_clock::now();

    if (dbg.src_img.img.empty())
        return;
    cv::Mat src_img = dbg.src_img.img;
    auto now = std::chrono::steady_clock::now();
    const double min_interval_ms = 1000.0 / 30.0;
    if (std::chrono::duration<double, std::milli>(now - last_show_time).count() < min_interval_ms
        && auto_fps)
        return;
    last_show_time = now;

    // 图像构造
    cv::Mat debug_img;
    cv::cvtColor(dbg.src_img.img, debug_img, cv::COLOR_BGR2RGB);

    if (debug_img.empty())
        return;

    // 封装后的绘图函数
    drawDebugArmorContent(debug_img, dbg, camera_info);
    // 编码写入共享内存路径
    std::vector<uchar> buf;
    cv::imencode(".jpg", debug_img, buf);
    std::ofstream ofs("/dev/shm/debug_frame.jpg.tmp", std::ios::binary);
    ofs.write(reinterpret_cast<const char*>(buf.data()), buf.size());
    ofs.close();
    std::rename("/dev/shm/debug_frame.jpg.tmp", "/dev/shm/debug_frame.jpg");
}
void drawDebugOverlayShm(
    const DebugArmor& dbg,
    std::pair<cv::Mat, cv::Mat> camera_info,
    bool auto_fps
) {
    static auto last_show_time = std::chrono::steady_clock::now();
    static int shm_fd = -1;
    static void* shm_ptr = nullptr;
    static bool shm_inited = false;

    constexpr size_t shm_max_size = 2 * 1024 * 1024;
    constexpr double min_interval_ms = 1000.0 / 30.0; // 30 FPS

    if (dbg.src_img.img.empty())
        return;

    auto now = std::chrono::steady_clock::now();
    if (auto_fps
        && std::chrono::duration<double, std::milli>(now - last_show_time).count()
            < min_interval_ms)
        return;
    last_show_time = now;

    if (!shm_inited) {
        shm_fd = shm_open("/debug_frame", O_CREAT | O_RDWR, 0777);
        if (shm_fd == -1) {
            std::cerr << "[SHM] shm_open failed\n";
            return;
        }
        if (ftruncate(shm_fd, shm_max_size) == -1) {
            std::cerr << "[SHM] ftruncate failed\n";
            close(shm_fd);
            shm_fd = -1;
            return;
        }
        shm_ptr = mmap(nullptr, shm_max_size, PROT_WRITE | PROT_READ, MAP_SHARED, shm_fd, 0);
        if (shm_ptr == MAP_FAILED) {
            std::cerr << "[SHM] mmap failed\n";
            close(shm_fd);
            shm_fd = -1;
            shm_ptr = nullptr;
            return;
        }

        shm_inited = true;
    }

    cv::Mat debug_img;
    cv::cvtColor(dbg.src_img.img, debug_img, cv::COLOR_BGR2RGB);

    drawDebugArmorContent(debug_img, dbg, camera_info);
    static std::vector<int> jpeg_params = { cv::IMWRITE_JPEG_QUALITY, 75 };

    std::vector<uchar> buf;
    cv::imencode(".jpg", debug_img, buf, jpeg_params);

    if (buf.size() + 4 > shm_max_size)
        return;

    uint32_t size = buf.size();
    std::memcpy(shm_ptr, &size, 4);
    std::memcpy(static_cast<char*>(shm_ptr) + 4, buf.data(), size);
}
void drawDebugOverlayShow(
    const DebugArmor& dbg,
    std::pair<cv::Mat, cv::Mat> camera_info,
    bool auto_fps
) {
    static auto last_show_time = std::chrono::steady_clock::now();

    if (dbg.src_img.img.empty())
        return;
    cv::Mat src_img = dbg.src_img.img;
    auto now = std::chrono::steady_clock::now();
    const double min_interval_ms = 1000.0 / 30.0;
    if (std::chrono::duration<double, std::milli>(now - last_show_time).count() < min_interval_ms
        && auto_fps)
        return;
    last_show_time = now;

    // 图像构造
    cv::Mat debug_img;
    cv::cvtColor(dbg.src_img.img, debug_img, cv::COLOR_BGR2RGB);
    if (debug_img.empty())
        return;

    // 封装后的绘图函数
    drawDebugArmorContent(debug_img, dbg, camera_info);

    cv::imshow("debug_armor", debug_img);
    cv::waitKey(1);
}

void drawDebugOverlayWrite(
    const DebugRune& dbg,
    std::pair<cv::Mat, cv::Mat> camera_info,
    bool auto_fps
) {
    static auto last_show_time = std::chrono::steady_clock::now();

    if (dbg.src_img.img.empty())
        return;
    cv::Mat src_img = dbg.src_img.img;
    auto now = std::chrono::steady_clock::now();
    const double min_interval_ms = 1000.0 / 30.0;
    if (std::chrono::duration<double, std::milli>(now - last_show_time).count() < min_interval_ms
        && auto_fps)
        return;
    last_show_time = now;

    // 图像构造
    cv::Mat debug_img;
    cv::cvtColor(dbg.src_img.img, debug_img, cv::COLOR_BGR2RGB);

    if (debug_img.empty())
        return;

    // 封装后的绘图函数
    drawDebugRuneContent(debug_img, dbg, camera_info);
    cv::cvtColor(debug_img, debug_img, cv::COLOR_RGB2BGR);
    // 编码写入共享内存路径
    std::vector<uchar> buf;
    cv::imencode(".jpg", debug_img, buf);
    std::ofstream ofs("/dev/shm/debug_frame.jpg.tmp", std::ios::binary);
    ofs.write(reinterpret_cast<const char*>(buf.data()), buf.size());
    ofs.close();
    std::rename("/dev/shm/debug_frame.jpg.tmp", "/dev/shm/debug_frame.jpg");
}
void drawDebugOverlayShm(
    const DebugRune& dbg,
    std::pair<cv::Mat, cv::Mat> camera_info,
    bool auto_fps
) {
    static auto last_show_time = std::chrono::steady_clock::now();
    static int shm_fd = -1;
    static void* shm_ptr = nullptr;
    static bool shm_inited = false;

    constexpr size_t shm_max_size = 2 * 1024 * 1024;
    constexpr double min_interval_ms = 1000.0 / 30.0; // 30 FPS

    if (dbg.src_img.img.empty())
        return;

    auto now = std::chrono::steady_clock::now();
    if (auto_fps
        && std::chrono::duration<double, std::milli>(now - last_show_time).count()
            < min_interval_ms)
        return;
    last_show_time = now;

    if (!shm_inited) {
        shm_fd = shm_open("/debug_frame", O_CREAT | O_RDWR, 0666);
        if (shm_fd == -1) {
            std::cerr << "[SHM] shm_open failed\n";
            return;
        }
        if (ftruncate(shm_fd, shm_max_size) == -1) {
            std::cerr << "[SHM] ftruncate failed\n";
            close(shm_fd);
            shm_fd = -1;
            return;
        }
        shm_ptr = mmap(nullptr, shm_max_size, PROT_WRITE | PROT_READ, MAP_SHARED, shm_fd, 0);
        if (shm_ptr == MAP_FAILED) {
            std::cerr << "[SHM] mmap failed\n";
            close(shm_fd);
            shm_fd = -1;
            shm_ptr = nullptr;
            return;
        }

        shm_inited = true;
    }

    cv::Mat debug_img;
    cv::cvtColor(dbg.src_img.img, debug_img, cv::COLOR_BGR2RGB);

    drawDebugRuneContent(debug_img, dbg, camera_info);

    static std::vector<int> jpeg_params = { cv::IMWRITE_JPEG_QUALITY, 75 };

    std::vector<uchar> buf;
    cv::imencode(".jpg", debug_img, buf, jpeg_params);

    if (buf.size() + 4 > shm_max_size)
        return;

    uint32_t size = buf.size();
    std::memcpy(shm_ptr, &size, 4);
    std::memcpy(static_cast<char*>(shm_ptr) + 4, buf.data(), size);
}

void drawDebugOverlayShow(
    const DebugRune& dbg,
    std::pair<cv::Mat, cv::Mat> camera_info,
    bool auto_fps
) {
    static auto last_show_time = std::chrono::steady_clock::now();

    if (dbg.src_img.img.empty())
        return;
    cv::Mat src_img = dbg.src_img.img;
    auto now = std::chrono::steady_clock::now();
    const double min_interval_ms = 1000.0 / 30.0;
    if (std::chrono::duration<double, std::milli>(now - last_show_time).count() < min_interval_ms
        && auto_fps)
        return;
    last_show_time = now;

    // 图像构造
    cv::Mat debug_img;
    cv::cvtColor(dbg.src_img.img, debug_img, cv::COLOR_BGR2RGB);

    if (debug_img.empty())
        return;

    // 封装后的绘图函数
    drawDebugRuneContent(debug_img, dbg, camera_info);

    cv::imshow("debug_rune", debug_img);
    cv::waitKey(1);
}

void writeTargetLogToJson(const Target& armor_target, const rune::RuneTarget& rune_target) {
    nlohmann::json j;

    // -------- armor_target 部分 --------
    nlohmann::json jt;
    jt["type"] = armor_target.type_;
    jt["tracking"] = armor_target.is_tracking;
    jt["id"] = static_cast<int>(armor_target.tracked_id_);
    jt["armors_num"] = armor_target.armor_num_;

    auto now = std::chrono::steady_clock::now();
    auto age_ms_t =
        std::chrono::duration_cast<std::chrono::milliseconds>(now - armor_target.timestamp_)
            .count();
    jt["timestamp_age_ms"] = age_ms_t;

    jt["position"] = { { "x", armor_target.position().x() },
                       { "y", armor_target.position().y() },
                       { "z", armor_target.position().z() } };

    jt["velocity"] = { { "x", armor_target.velocity().x() },
                       { "y", armor_target.velocity().y() },
                       { "z", armor_target.velocity().z() } };

    jt["r"] = armor_target.r();
    jt["l"] = armor_target.l();
    jt["h"] = armor_target.h();
    jt["yaw"] = armor_target.yaw();
    jt["v_yaw"] = armor_target.v_yaw();

    // -------- RuneTarget 部分 --------
    nlohmann::json jr;
    jr["tracking"] = rune_target.is_tracking;
    jr["id"] = static_cast<int>(rune_target.last_id);

    auto age_ms_r =
        std::chrono::duration_cast<std::chrono::milliseconds>(now - rune_target.timestamp_).count();
    jr["timestamp_age_ms"] = age_ms_r;

    jr["position"] = { { "x", rune_target.centerPos().x() },
                       { "y", rune_target.centerPos().y() },
                       { "z", rune_target.centerPos().z() } };

    jr["roll"] = rune_target.roll() * 180.0 / M_PI;
    jr["yaw"] = rune_target.yaw() * 180.0 / M_PI;
    jr["v_roll"] = rune_target.v_roll() * 180.0 / M_PI;

    // -------- 合并 --------
    j["armor_target"] = jt;
    j["rune_target"] = jr;

    // -------- 写文件 --------
    std::ofstream file("/dev/shm/target_log.json");
    if (file.is_open()) {
        file << j.dump(2);
    }
}

void writeSerialLogToJson(const ReceiveAimINFO& aim) {
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
    auto now = std::chrono::steady_clock::now();
    double elapsed = std::chrono::duration<double>(now - last_time).count();
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
void debuglog(
    const DebugArmor& dbg_armor,
    const DebugRune& dbg_rune,
    const GimbalCmd& gimbal_cmd,
    const std::pair<double, double>& gimbal_py
) {
    static bool first_log = true;
    static std::chrono::steady_clock::time_point start_time;

    // last_* 静态化
    static armor::Armor last_armor_;
    static double last_armor_yaw_ = 0.0;
    static double last_ypd_y_ = 0.0;
    static double last_ypd_p_ = 0.0;
    static double last_distance_ = 0.0;
    static DebugLogs log;
    static GimbalCmd last_cmd_;
    static double rune_dis = 0.0;
    if (first_log) {
        start_time = std::chrono::steady_clock::now();
        first_log = false;
    }

    auto now = std::chrono::steady_clock::now();
    armor::Armors armors = dbg_armor.armors;
    double t = std::chrono::duration<double>(now - start_time).count();
    Target target = dbg_armor.target;
    rune::RuneTarget rune_target = dbg_rune.target;
    writeTargetLogToJson(target, rune_target);

    double armor_yaw = 0.0, ypd_y = 0.0, ypd_p = 0.0, armor_distance = 0.0;
    if (dbg_rune.pnp_distance > 1.0) {
        rune_dis = dbg_rune.pnp_distance;
    }

    if (!armors.armors.empty()) {
        std::vector<armor::Armor> ok_armors;
        for (const auto& armor: armors.armors) {
            if (armor.number != armor::ArmorNumber::OUTPOST)
                ok_armors.push_back(armor);
        }

        if (!ok_armors.empty()) {
            const armor::Armor& min_armor = *std::min_element(
                ok_armors.begin(),
                ok_armors.end(),
                [](const armor::Armor& a, const armor::Armor& b) {
                    return a.distance_to_image_center < b.distance_to_image_center;
                }
            );

            last_armor_ = min_armor;

            armor_distance = std::hypot(
                min_armor.target_pos.x(),
                min_armor.target_pos.y(),
                min_armor.target_pos.z()
            );
            auto orientationToYaw = [](const Eigen::Quaterniond& q) noexcept -> double {
                Eigen::Vector3d euler = utils::quatToEuler(q, utils::EulerOrder::ZYX, false);
                double yaw = euler[0];
                yaw = last_armor_yaw_ + angles::shortest_angular_distance(last_armor_yaw_, yaw);
                last_armor_yaw_ = yaw;
                return yaw;
            };

            armor_yaw = orientationToYaw(min_armor.target_ori);

            ypd_y = std::atan2(min_armor.target_pos.y(), min_armor.target_pos.x());
            ypd_y = last_ypd_y_ + angles::shortest_angular_distance(last_ypd_y_, ypd_y);
            last_ypd_y_ = ypd_y;

            ypd_p = std::atan2(
                min_armor.target_pos.z(),
                std::hypot(min_armor.target_pos.x(), min_armor.target_pos.y())
            );
            last_ypd_p_ = ypd_p;

            last_distance_ = armor_distance;
        }
    }
    GimbalCmd i_use;
    if (gimbal_cmd.appera) {
        i_use = gimbal_cmd;
    } else {
        i_use = last_cmd_;
    }
    log.time_log.push_back(t);
    log.raw_yaw_log.push_back(i_use.raw_yaw);
    log.raw_pitch_log.push_back(i_use.raw_pitch);
    log.cmd_yaw_log.push_back(i_use.yaw);
    log.cmd_pitch_log.push_back(i_use.pitch);
    log.rune_obs_log.push_back(dbg_rune.obs_angle);
    log.rune_pre_log.push_back(dbg_rune.pre_angle);
    log.rune_fitv_log.push_back(dbg_rune.fitter_v * 180.0 / M_PI);
    log.rune_obsv_log.push_back(dbg_rune.obs_v * 180.0 / M_PI);
    log.armor_yaw_log.push_back(armor_yaw * 180.0 / M_PI);
    log.armor_x_log.push_back(last_armor_.target_pos.x());
    log.armor_y_log.push_back(last_armor_.target_pos.y());
    log.armor_z_log.push_back(last_armor_.target_pos.z());
    log.ypd_y_log.push_back(last_ypd_y_ * 180.0 / M_PI);
    log.ypd_p_log.push_back(last_ypd_p_ * 180.0 / M_PI);
    log.armor_dis_log.push_back(last_distance_);
    log.gimbal_pitch_log.push_back(gimbal_py.first * 180.0 / M_PI);
    log.gimbal_yaw_log.push_back(gimbal_py.second * 180.0 / M_PI);
    log.target_v_yaw_log.push_back(target.v_yaw());
    log.control_v_pitch_log.push_back(i_use.v_pitch);
    log.control_v_yaw_log.push_back(i_use.v_yaw);
    log.yaw_diff_log.push_back(i_use.yaw_diff);
    log.fire_log.push_back(i_use.fire_advice);
    log.rune_dis_log.push_back(rune_dis);
    if (gimbal_cmd.appera) {
        last_cmd_ = gimbal_cmd;
    }

    // 控制长度不超过 100
    auto trim = [](std::vector<double>& v) {
        if (v.size() > 100)
            v.erase(v.begin());
    };

    trim(log.time_log);
    trim(log.raw_yaw_log);
    trim(log.raw_pitch_log);
    trim(log.cmd_yaw_log);
    trim(log.cmd_pitch_log);
    trim(log.rune_obs_log);
    trim(log.rune_pre_log);
    trim(log.rune_obsv_log);
    trim(log.rune_fitv_log);
    trim(log.armor_yaw_log);
    trim(log.armor_x_log);
    trim(log.armor_y_log);
    trim(log.armor_z_log);
    trim(log.ypd_y_log);
    trim(log.ypd_p_log);
    trim(log.armor_dis_log);
    trim(log.gimbal_pitch_log);
    trim(log.gimbal_yaw_log);
    trim(log.target_v_yaw_log);
    trim(log.control_v_pitch_log);
    trim(log.control_v_yaw_log);
    trim(log.yaw_diff_log);
    trim(log.fire_log);
    trim(log.rune_dis_log);
    nlohmann::json j;
    {
        j["time"] = log.time_log;
        j["raw_yaw"] = log.raw_yaw_log;
        j["raw_pitch"] = log.raw_pitch_log;
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
        j["rune_obsv"] = log.rune_obsv_log;
        j["rune_fitv"] = log.rune_fitv_log;
        j["gimbal_yaw"] = log.gimbal_yaw_log;
        j["gimbal_pitch"] = log.gimbal_pitch_log;
        j["target_v_yaw"] = log.target_v_yaw_log;
        j["control_v_pitch"] = log.control_v_pitch_log;
        j["control_v_yaw"] = log.control_v_yaw_log;
        j["yaw_diff"] = log.yaw_diff_log;
        j["fire"] = log.fire_log;
        j["rune_dis"] = log.rune_dis_log;
    }
    std::ofstream file("/dev/shm/cmd_log.json");
    if (file.is_open()) {
        file << j.dump();
    }
}
