#pragma once
#include "Eigen/Dense"
#include <opencv2/opencv.hpp>
#include <optional>
#include <shared_mutex>
#define ROOT_CONFIG "/home/hy/wust_vision/config/config_common.yaml"
#define OPENVINO_CONFIG "/home/hy/wust_vision/config/detect_openvino.yaml"
#define TENSORRT_CONFIG "/home/hy/wust_vision/config/detect_trt.yaml"
#define NCNN_CONFIG "/home/hy/wust_vision/config/detect_ncnn.yaml"
#define ONNXRUNTIME_CONFIG "/home/hy/wust_vision/config/detect_ort.yaml"
#define OPENCV_CONFIG "/home/hy/wust_vision/config/detect_opencv.yaml"
#define OMNI_CONFIG "/home/hy/wust_vision/config/omni_config.yaml"
#define FUN_CONFIG "/home/hy/wust_vision/config/fun.yaml"

struct CommonFrame {
    cv::Mat src_img;
    std::chrono::steady_clock::time_point timestamp;
    Eigen::Matrix4d T_camera_to_odom;
    Eigen::Vector3d v;
    int id;
    int detect_color;
};
enum class EnemyColor {
    RED = 0,
    BLUE = 1,
    WHITE = 2,
};
inline std::string enemyColorToString(EnemyColor color) {
    switch (color) {
        case EnemyColor::RED:
            return "RED";
            break;
        case EnemyColor::BLUE:
            return "BLUE";
            break;
        case EnemyColor::WHITE:
            return "WHITE";
            break;
        default:
            return "UNKNOWN";
    }
}
struct GridAndStride {
    int grid0;
    int grid1;
    int stride;
};
struct imgframe {
    cv::Mat img;
    std::chrono::steady_clock::time_point timestamp;
};
struct GimbalCmd {
    std::chrono::steady_clock::time_point timestamp;
    float pitch = 0;
    float yaw = 0;
    float yaw_diff = 0;
    float pitch_diff = 0;
    float v_yaw = 0;
    float v_pitch = 0;
    float distance = -1;
    bool fire_advice = false;
    int select_id = -1;
};
enum class AttackMode { ARMOR = 0, SMALL_RUNE, BIG_RUNE, UNKNOWN };
inline AttackMode toAttackMode(int value) {
    switch (value) {
        case 0:
            return AttackMode::ARMOR;
        case 1:
            return AttackMode::SMALL_RUNE;
        case 2:
            return AttackMode::BIG_RUNE;
        default:
            return AttackMode::UNKNOWN;
    }
}
class MotionBuffer {
public:
    struct MotionStamped {
        double yaw, pitch, roll;
        double vx, vy, vz;
        std::chrono::steady_clock::time_point stamp;
    };

    static constexpr size_t BUFFER_SIZE = 512;

    void push(
        double yaw,
        double pitch,
        double roll,
        double vx,
        double vy,
        double vz,
        std::chrono::steady_clock::time_point stamp
    ) {
        std::unique_lock lock(mutex_);

        if (has_last_) {
            yaw = unwrap_angle(last_.yaw, yaw);
            pitch = unwrap_angle(last_.pitch, pitch);
            roll = unwrap_angle(last_.roll, roll);
        }

        buffer_[head_] = { yaw, pitch, roll, vx, vy, vz, stamp };
        time_buffer_[head_] = stamp;

        head_ = (head_ + 1) % BUFFER_SIZE;
        if (size_ < BUFFER_SIZE)
            ++size_;

        last_ = buffer_[(head_ + BUFFER_SIZE - 1) % BUFFER_SIZE];
        has_last_ = true;
    }

    std::optional<MotionStamped> get_interpolated(std::chrono::steady_clock::time_point t_query) {
        std::shared_lock lock(mutex_);
        if (size_ < 2)
            return std::nullopt;

        size_t begin = (head_ + BUFFER_SIZE - size_) % BUFFER_SIZE;

        std::vector<std::chrono::steady_clock::time_point> times;
        times.reserve(size_);
        for (size_t i = 0; i < size_; ++i) {
            times.push_back(time_buffer_[(begin + i) % BUFFER_SIZE]);
        }

        auto it_hi = std::lower_bound(times.begin(), times.end(), t_query);
        if (it_hi == times.begin() || it_hi == times.end())
            return std::nullopt;

        size_t idx_hi = std::distance(times.begin(), it_hi);
        size_t idx_lo = idx_hi - 1;

        const auto& lo = buffer_[(begin + idx_lo) % BUFFER_SIZE];
        const auto& hi = buffer_[(begin + idx_hi) % BUFFER_SIZE];
        double span = std::chrono::duration<double>(hi.stamp - lo.stamp).count();
        if (span <= 0.0)
            return lo;

        double t = std::chrono::duration<double>(t_query - lo.stamp).count() / span;

        MotionStamped res;
        res.yaw = lo.yaw + t * (hi.yaw - lo.yaw);
        res.pitch = lo.pitch + t * (hi.pitch - lo.pitch);
        res.roll = lo.roll + t * (hi.roll - lo.roll);
        res.vx = lo.vx + t * (hi.vx - lo.vx);
        res.vy = lo.vy + t * (hi.vy - lo.vy);
        res.vz = lo.vz + t * (hi.vz - lo.vz);
        res.stamp = t_query;
        return res;
    }

    // 🔹 新增方法：获取最后一帧
    std::optional<MotionStamped> get_last() const {
        std::shared_lock lock(mutex_);
        if (!has_last_)
            return std::nullopt;
        return last_;
    }

private:
    std::array<MotionStamped, BUFFER_SIZE> buffer_;
    std::array<std::chrono::steady_clock::time_point, BUFFER_SIZE> time_buffer_;
    size_t head_ = 0, size_ = 0;

    bool has_last_ = false;
    MotionStamped last_;

    mutable std::shared_mutex mutex_;

    double unwrap_angle(double prev, double curr) {
        double d = curr - prev;
        if (d > M_PI)
            curr -= 2 * M_PI;
        else if (d < -M_PI)
            curr += 2 * M_PI;
        return curr;
    }
};
