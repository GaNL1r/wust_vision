#pragma once
#include <fcntl.h>
#include <fmt/core.h>
#include <fstream>
#include <nlohmann/json.hpp>
#include <opencv2/opencv.hpp>
#include <sys/mman.h>
#include <wust_vl/video/icamera.hpp>
namespace wust_vision {

template<typename T, int MAX_N>
class LogsStream {
public:
    LogsStream(const std::string& n) {
        name = n;
    }
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
        j[name] = log_data;
    }
    void clear() {
        log_data.clear();
    }

private:
    std::string name;
    std::vector<T> log_data;
};
template<typename DebugT, typename DrawFn, typename OutputFn>
void drawDebugOverlayImpl(
    const DebugT& dbg,
    std::pair<cv::Mat, cv::Mat> camera_info,
    bool auto_fps,
    DrawFn&& draw_fn,
    OutputFn&& output_fn
) {
    static auto last_show_time = std::chrono::steady_clock::now();

    if (dbg.img_frame.src_img.empty())
        return;

    constexpr double min_interval_ms = 1000.0 / 30.0;
    const auto now = std::chrono::steady_clock::now();

    if (auto_fps
        && std::chrono::duration<double, std::milli>(now - last_show_time).count()
            < min_interval_ms)
        return;

    last_show_time = now;

    cv::Mat debug_img;
    if (dbg.img_frame.pixel_format == wust_vl::video::PixelFormat::GRAY) {
        cv::cvtColor(dbg.img_frame.src_img, debug_img, cv::COLOR_GRAY2RGB);
    } else if (dbg.img_frame.pixel_format == wust_vl::video::PixelFormat::BGR) {
        cv::cvtColor(dbg.img_frame.src_img, debug_img, cv::COLOR_BGR2RGB);
    } else {
        debug_img = dbg.img_frame.src_img;
    }

    if (debug_img.empty())
        return;
    draw_fn(debug_img, dbg, camera_info);

    output_fn(debug_img);
}
inline auto writeToFile = [](const cv::Mat& img) {
    cv::Mat bgr;
    cv::cvtColor(img, bgr, cv::COLOR_RGB2BGR);

    std::vector<uchar> buf;
    cv::imencode(".jpg", bgr, buf);

    std::ofstream ofs("/dev/shm/debug_frame.jpg.tmp", std::ios::binary);
    ofs.write(reinterpret_cast<const char*>(buf.data()), buf.size());
    ofs.close();

    std::rename("/dev/shm/debug_frame.jpg.tmp", "/dev/shm/debug_frame.jpg");
};
class ShmWriter {
public:
    static constexpr size_t shm_max_size = 2 * 1024 * 1024;

    explicit ShmWriter(const char* name, mode_t mode = 0666) {
        fd_ = shm_open(name, O_CREAT | O_RDWR, mode);
        if (fd_ == -1) {
            std::cerr << "[SHM] shm_open failed\n";
            return;
        }

        if (ftruncate(fd_, shm_max_size) == -1) {
            std::cerr << "[SHM] ftruncate failed\n";
            close(fd_);
            fd_ = -1;
            return;
        }

        ptr_ = mmap(nullptr, shm_max_size, PROT_READ | PROT_WRITE, MAP_SHARED, fd_, 0);

        if (ptr_ == MAP_FAILED) {
            std::cerr << "[SHM] mmap failed\n";
            close(fd_);
            fd_ = -1;
            ptr_ = nullptr;
        }
    }

    ~ShmWriter() {
        if (ptr_)
            munmap(ptr_, shm_max_size);
        if (fd_ != -1)
            close(fd_);
    }

    void operator()(const cv::Mat& img) const {
        if (!ptr_)
            return;

        static const std::vector<int> jpeg_params = { cv::IMWRITE_JPEG_QUALITY, 75 };

        std::vector<uchar> buf;
        cv::imencode(".jpg", img, buf, jpeg_params);

        if (buf.size() + 4 > shm_max_size)
            return;

        uint32_t size = static_cast<uint32_t>(buf.size());
        std::memcpy(ptr_, &size, 4);
        std::memcpy(static_cast<char*>(ptr_) + 4, buf.data(), size);
    }

private:
    int fd_ { -1 };
    void* ptr_ { nullptr };
};
inline auto showWindow(const char* win_name) {
    return [win_name](const cv::Mat& img) {
        cv::imshow(win_name, img);
        cv::waitKey(1);
    };
}
} // namespace wust_vision