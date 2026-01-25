#include "debug.hpp"
#include <fcntl.h>
#include <fmt/format.h>
#include <fstream>
#include <nlohmann/json.hpp>
#include <sys/mman.h>
#include <sys/stat.h>
namespace wust_vision {
namespace auto_guidance {
    void drawAutoGuidanceDebugContent(cv::Mat& debug_img, const AutoGuidanceDebug& dbg) {
        auto target = dbg.target;
        auto lights = dbg.lights;

        lights.drawFront(debug_img);

        if (target.is_tracking_) {
            auto now = std::chrono::steady_clock::now();
            target.predict(now);
            target.draw(debug_img);
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

        cv::circle(
            debug_img,
            cv::Point2i(debug_img.cols / 2, debug_img.rows / 2),
            5,
            cv::Scalar(255, 255, 255),
            2
        );
        double cx_norm = target.center().x / target.image_size_.width * 2.0 - 1.0;
        double diff_center_norm = (target.is_tracking_) ? cx_norm : 0;
        {
            std::string diff_str = fmt::format("diff_cx_norm: {:.3f}", diff_center_norm);

            int margin = 10;
            int font_face = cv::FONT_HERSHEY_SIMPLEX;
            double font_scale = 0.7;
            int thickness = 2;

            int baseline = 0;
            cv::Size text_size =
                cv::getTextSize(diff_str, font_face, font_scale, thickness, &baseline);

            // 右上角文本左下角坐标
            int x = debug_img.cols - margin - text_size.width;
            int y = margin + text_size.height;

            // 背景框，可选
            cv::rectangle(
                debug_img,
                cv::Rect(
                    x - 5,
                    y - text_size.height - 5,
                    text_size.width + 10,
                    text_size.height + 10
                ),
                cv::Scalar(0, 0, 0),
                cv::FILLED,
                cv::LINE_AA
            );

            cv::putText(
                debug_img,
                diff_str,
                cv::Point(x, y),
                font_face,
                font_scale,
                cv::Scalar(0, 255, 255),
                thickness,
                cv::LINE_AA
            );
        }

        if (target.is_tracking_) {
            const auto& s = target.target_state_;

            std::string line1 =
                fmt::format("pos: {:.1f} {:.1f} {:.1f} {:.1f}", s(0), s(2), s(4), s(6));
            std::string line2 =
                fmt::format("vel: {:.2f} {:.2f} {:.2f} {:.2f}", s(1), s(3), s(5), s(7));

            int x = 10;
            int y = debug_img.rows - 30; // 左下角位置
            int dy = 28;

            cv::putText(
                debug_img,
                line1,
                cv::Point(x, y),
                cv::FONT_HERSHEY_SIMPLEX,
                0.75,
                cv::Scalar(0, 255, 0),
                2
            );

            cv::putText(
                debug_img,
                line2,
                cv::Point(x, y + dy),
                cv::FONT_HERSHEY_SIMPLEX,
                0.75,
                cv::Scalar(0, 255, 0),
                2
            );

            double vx = s(1);
            double vy = s(3);

            cv::Point2f p0 = target.center();

            double scale = 0.5;

            cv::Point2f p1(p0.x + vx * scale, p0.y + vy * scale);

            cv::arrowedLine(
                debug_img,
                p0,
                p1,
                cv::Scalar(0, 255, 0),
                2,
                cv::LINE_AA,
                0,
                0.25 // 箭头比例
            );
        }
    }

    void drawDebugOverlayWrite(const AutoGuidanceDebug& dbg, bool auto_fps) {
        static auto last_show_time = std::chrono::steady_clock::now();

        if (dbg.src_img.empty())
            return;
        cv::Mat src_img = dbg.src_img;
        auto now = std::chrono::steady_clock::now();
        const double min_interval_ms = 1000.0 / 30.0;
        if (std::chrono::duration<double, std::milli>(now - last_show_time).count()
                < min_interval_ms
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
        drawAutoGuidanceDebugContent(debug_img, dbg);
        cv::cvtColor(debug_img, debug_img, cv::COLOR_RGB2BGR);
        // 编码写入共享内存路径
        std::vector<uchar> buf;
        cv::imencode(".jpg", debug_img, buf);
        std::ofstream ofs("/dev/shm/debug_frame.jpg.tmp", std::ios::binary);
        ofs.write(reinterpret_cast<const char*>(buf.data()), buf.size());
        ofs.close();
        std::rename("/dev/shm/debug_frame.jpg.tmp", "/dev/shm/debug_frame.jpg");
    }
    void drawDebugOverlayShm(const AutoGuidanceDebug& dbg, bool auto_fps) {
        static auto last_show_time = std::chrono::steady_clock::now();
        const char* shm_name = "/debug_frame";
        const size_t shm_max_size = 2 * 1024 * 1024; // 2MB 最大图像编码缓存

        if (dbg.src_img.empty())
            return;
        cv::Mat src_img = dbg.src_img;

        auto now = std::chrono::steady_clock::now();
        const double min_interval_ms = 1000.0 / 30.0;
        if (std::chrono::duration<double, std::milli>(now - last_show_time).count()
                < min_interval_ms
            && auto_fps)
            return;
        last_show_time = now;

        // 复制并转RGB
        cv::Mat debug_img;
        src_img.convertTo(debug_img, -1, 1, 0);
        cv::cvtColor(debug_img, debug_img, cv::COLOR_BGR2RGB);
        if (debug_img.empty())
            return;

        // 绘制内容
        drawAutoGuidanceDebugContent(debug_img, dbg);
        // 编码为 JPG
        std::vector<uchar> buf;
        cv::imencode(".jpg", debug_img, buf);
        size_t img_size = buf.size();

        if (img_size > shm_max_size) {
            std::cerr << "[drawDebugOverlayWrite] 图像过大: " << img_size << " bytes\n";
            return;
        }

        // 创建/打开共享内存
        int fd = shm_open(shm_name, O_CREAT | O_RDWR, 0666);
        if (fd == -1) {
            perror("shm_open failed");
            return;
        }

        // 设置共享内存大小
        if (ftruncate(fd, shm_max_size) == -1) {
            perror("ftruncate failed");
            close(fd);
            return;
        }

        // 映射共享内存
        void* ptr = mmap(nullptr, shm_max_size, PROT_WRITE, MAP_SHARED, fd, 0);
        if (ptr == MAP_FAILED) {
            perror("mmap failed");
            close(fd);
            return;
        }

        // 写入图像数据
        uint32_t size = static_cast<uint32_t>(img_size);
        std::memcpy(ptr, &size, 4); // 前4字节写入长度
        std::memcpy(static_cast<char*>(ptr) + 4, buf.data(), img_size);

        // 关闭映射和文件描述符
        munmap(ptr, shm_max_size);
        close(fd);
    }
    void drawDebugOverlayShow(const AutoGuidanceDebug& dbg, bool auto_fps) {
        static auto last_show_time = std::chrono::steady_clock::now();

        if (dbg.src_img.empty())
            return;
        cv::Mat src_img = dbg.src_img;
        auto now = std::chrono::steady_clock::now();
        const double min_interval_ms = 1000.0 / 30.0;
        if (std::chrono::duration<double, std::milli>(now - last_show_time).count()
                < min_interval_ms
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
        drawAutoGuidanceDebugContent(debug_img, dbg);

        cv::imshow("debug_armor", debug_img);
        cv::waitKey(1);
    }
    void debuglog(const GuidanceTarget& target) {
        static bool first_log = true;
        static std::chrono::steady_clock::time_point start_time;
        static DebugLogs log;
        if (first_log) {
            start_time = std::chrono::steady_clock::now();
            first_log = false;
        }
        auto now = std::chrono::steady_clock::now();
        double t = std::chrono::duration<double>(now - start_time).count();
        log.time_log.push_back(t);
        double cx_norm = target.center().x / target.image_size_.width * 2.0 - 1.0;
        log.cx_log.push_back(cx_norm);
        auto trim = [](std::vector<double>& v) {
            if (v.size() > 1000)
                v.erase(v.begin());
        };

        trim(log.time_log);
        trim(log.cx_log);
        nlohmann::json j;
        {
            j["time"] = log.time_log;
            j["yaw"] = log.cx_log;
        }
        std::ofstream file("/dev/shm/cmd_log.json");
        if (file.is_open()) {
            file << j.dump();
        }
    }
} // namespace auto_guidance
} // namespace wust_vision