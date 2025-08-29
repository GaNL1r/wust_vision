#pragma once
#include "3rdparty/angles.h"
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
    float raw_yaw;
    float pitch = 0;
    float yaw = 0;
    float yaw_diff = 0;
    float pitch_diff = 0;
    float v_yaw = 0;
    float v_pitch = 0;
    float distance = -1;
    bool fire_advice = false;
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
static std::vector<cv::Point3f> AIM_TARGET_BLOCK = {
    { -0.025f, -0.025f, -0.025f }, // 0: 左下前
    { 0.025f, -0.025f, -0.025f }, // 1: 右下前
    { 0.025f, 0.025f, -0.025f }, // 2: 右上前
    { -0.025f, 0.025f, -0.025f }, // 3: 左上前
    { -0.025f, -0.025f, 0.025f }, // 4: 左下后
    { 0.025f, -0.025f, 0.025f }, // 5: 右下后
    { 0.025f, 0.025f, 0.025f }, // 6: 右上后
    { -0.025f, 0.025f, 0.025f } // 7: 左上后
};

struct AimTarget {
    Eigen::Vector3d pos = Eigen::Vector3d(0, 0, 0);
    Eigen::Vector3d vel = Eigen::Vector3d(0, 0, 0);

    bool have_host = false;
    Eigen::Vector3d host_pos = Eigen::Vector3d(0, 0, 0);
    Eigen::Vector3d host_vel = Eigen::Vector3d(0, 0, 0);

    double host_v_yaw = 0;

    double shoot_pitch = 0;
    int idx = -1;
    bool is_big_armor = false;
    bool is_old = false;

    Eigen::Quaterniond ori;
    void predictSelf(double dt_sec) {
        if (!have_host)
            return;

        Eigen::Vector3d rel_pos = pos - host_pos;

        double theta = host_v_yaw * dt_sec;

        Eigen::Matrix3d R;
        R = Eigen::AngleAxisd(theta, Eigen::Vector3d::UnitZ());

        pos = host_pos + R * rel_pos + host_vel * dt_sec;
    }
    double calYaw() const {
        return angles::normalize_angle(std::atan2(pos.y(), pos.x()));
    }

    double calRawPitch() const {
        return std::atan2(pos.z(), std::sqrt(pos.x() * pos.x() + pos.y() * pos.y()));
    }

    double distance() const {
        return pos.norm();
    }

    double calVYaw() const {
        double x = pos.x();
        double y = pos.y();
        double vx = vel.x();
        double vy = vel.y();

        double denom = x * x + y * y;
        if (denom < 1e-6)
            return 0.0;

        return (x * vy - y * vx) / denom;
    }
    double calHostVYaw() const {
        double x = host_pos.x();
        double y = host_pos.y();
        double vx = host_vel.x();
        double vy = host_vel.y();

        double denom = x * x + y * y;
        if (denom < 1e-6)
            return 0.0;

        return (x * vy - y * vx) / denom;
    }

    double calVPitch() const {
        double x = pos.x();
        double y = pos.y();
        double z = pos.z();
        double vx = vel.x();
        double vy = vel.y();
        double vz = vel.z();

        double rho2 = x * x + y * y;
        double rho = std::sqrt(rho2);
        double r2 = rho2 + z * z;

        if (r2 < 1e-12 || rho < 1e-6)
            return 0.0;

        return (x * z * vx + y * z * vy - rho2 * vz) / (r2 * rho);
    }
    void tf(Eigen::Matrix4d T_camera_to_odom) {
        Eigen::Vector4d pos_camera(pos.x(), pos.y(), pos.z(), 1.0);
        Eigen::Vector4d pos_odom = T_camera_to_odom * pos_camera;

        pos.x() = pos_odom.x();
        pos.y() = pos_odom.y();
        pos.z() = pos_odom.z();
        Eigen::Matrix3d R_camera_to_odom = T_camera_to_odom.block<3, 3>(0, 0);
        Eigen::Quaterniond q_camera(ori.w(), ori.x(), ori.y(), ori.z());
        Eigen::Matrix3d R_ori_camera = q_camera.normalized().toRotationMatrix();

        Eigen::Matrix3d R_ori_odom = R_camera_to_odom * R_ori_camera;
        Eigen::Quaterniond q_odom(R_ori_odom);

        ori.w() = q_odom.w();
        ori.x() = q_odom.x();
        ori.y() = q_odom.y();
        ori.z() = q_odom.z();
    }
    std::vector<cv::Point2f>
    toPts(const cv::Mat& camera_intrinsic, const cv::Mat& camera_distortion) {
        std::vector<cv::Point2f> pts;
        if (pos.norm() < 0.5) {
            return pts;
        }

        cv::Mat tvec = (cv::Mat_<double>(3, 1) << pos.x(), pos.y(), pos.z());
        Eigen::Matrix3d tf_rot = ori.toRotationMatrix();
        cv::Mat rot_mat =
            (cv::Mat_<double>(3, 3) << tf_rot(0, 0),
             tf_rot(0, 1),
             tf_rot(0, 2),
             tf_rot(1, 0),
             tf_rot(1, 1),
             tf_rot(1, 2),
             tf_rot(2, 0),
             tf_rot(2, 1),
             tf_rot(2, 2));

        // 旋转矩阵 -> 旋转向量
        cv::Mat rvec;
        cv::Rodrigues(rot_mat, rvec);

        cv::projectPoints(AIM_TARGET_BLOCK, rvec, tvec, camera_intrinsic, camera_distortion, pts);

        return pts;
    }
};
