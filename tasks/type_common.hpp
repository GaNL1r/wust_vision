#pragma once
#include "3rdparty/angles.h"
#include "Eigen/Dense"
#include "wust_vl/common/utils/motion_buffer.hpp"
#include <opencv2/opencv.hpp>
#include <optional>
#include <shared_mutex>
#include <yaml-cpp/yaml.h>

#define OPENVINO_CONFIG "config/detect_openvino.yaml"
#define TENSORRT_CONFIG "config/detect_trt.yaml"
#define NCNN_CONFIG "config/detect_ncnn.yaml"
#define ONNXRUNTIME_CONFIG "config/detect_ort.yaml"
#define OPENCV_CONFIG "config/detect_opencv.yaml"
#define COMMON_CONFIG "config/common.yaml"
#define CAMERA_CONFIG "config/camera.yaml"
#define AUTO_AIM_CONFIG "config/auto_aim.yaml"
#define AUTO_BUFF_CONFIG "config/auto_buff.yaml"
#define AUTO_GUIDANCE_CONFIG "config/auto_guidance.yaml"
struct CommonFrame {
    cv::Mat src_img;
    std::chrono::steady_clock::time_point timestamp;
    int id;
    int detect_color;
    cv::Rect expanded;
    cv::Point2f offset = cv::Point2f(0, 0);
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
struct Motion {
    double yaw, pitch, roll; // 欧拉角 (rad)
    double vyaw, vpitch, vroll; // 角速度
    double vx, vy, vz; // 线速度
};

// 角度 unwrap 辅助函数
inline double unwrap_angle(double prev, double curr) {
    double d = curr - prev;
    while (d > M_PI) {
        curr -= 2.0 * M_PI;
        d -= 2.0 * M_PI;
    }
    while (d < -M_PI) {
        curr += 2.0 * M_PI;
        d += 2.0 * M_PI;
    }
    return curr;
}

// 角度插值（wrap-safe）
inline double interp_angle(double a, double b, double t) {
    double diff = b - a;
    while (diff > M_PI)
        diff -= 2.0 * M_PI;
    while (diff < -M_PI)
        diff += 2.0 * M_PI;
    return a + diff * t;
}

// 🔹 MotionTraits 特化
template<>
struct MotionTraits<Motion> {
    static void unwrap(const Motion& prev, Motion& curr) {
        curr.yaw = unwrap_angle(prev.yaw, curr.yaw);
        curr.pitch = unwrap_angle(prev.pitch, curr.pitch);
        curr.roll = unwrap_angle(prev.roll, curr.roll);
        // 速度部分不需要 unwrap
    }

    static Motion interpolate(const Motion& a, const Motion& b, double t) {
        Motion out;
        // 欧拉角 wrap-safe 插值
        out.yaw = interp_angle(a.yaw, b.yaw, t);
        out.pitch = interp_angle(a.pitch, b.pitch, t);
        out.roll = interp_angle(a.roll, b.roll, t);

        // 角速度和线速度线性插值
        out.vyaw = a.vyaw + (b.vyaw - a.vyaw) * t;
        out.vpitch = a.vpitch + (b.vpitch - a.vpitch) * t;
        out.vroll = a.vroll + (b.vroll - a.vroll) * t;
        out.vx = a.vx + (b.vx - a.vx) * t;
        out.vy = a.vy + (b.vy - a.vy) * t;
        out.vz = a.vz + (b.vz - a.vz) * t;
        return out;
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
    bool valid;
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

    double dt = 0.0;

    Eigen::Quaterniond ori;
    std::vector<Eigen::Vector4d> armor_posandyaw;
    void predictSelf(double dt_sec) {
        if (!have_host)
            return;

        Eigen::Vector3d rel_pos = pos - host_pos;

        double theta = host_v_yaw * dt_sec;

        Eigen::Matrix3d R;
        R = Eigen::AngleAxisd(theta, Eigen::Vector3d::UnitZ());

        pos = host_pos + R * rel_pos + host_vel * dt_sec;
    }
    double calYaw(double self_v_yaw = 0) const {
        return angles::normalize_angle(std::atan2(pos.y(), pos.x()) - self_v_yaw * dt);
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
struct GimbalCmd {
    std::chrono::steady_clock::time_point timestamp;
    float raw_yaw = 0;
    float raw_pitch = 0;
    float pitch = 0;
    float yaw = 0;
    float yaw_diff = 0;
    float pitch_diff = 0;
    float target_yaw = 0;
    float target_pitch = 0;
    float v_yaw = 0;
    float v_pitch = 0;
    float distance = -1;
    bool fire_advice = false;
    float enable_yaw_diff = 0;
    float enable_pitch_diff = 0;
    bool appera = false;
    std::vector<Eigen::Vector4d> armor_posandyaw;
    AimTarget aim_target;
};

struct AutoExposureCfg {
    bool enable = false;
    double target_brightness = 20.0;
    double tolerance = 5.0;
    double step_gain = 10.0;
    double decay_step = 1.0;
    double exposure_min = 100.0;
    double exposure_max = 2500.0;
    double control_interval_ms = 300;

    // 成员函数：从 YAML 加载配置
    bool loadFromYaml(const YAML::Node& root) {
        try {
            if (root["enable"])
                enable = root["enable"].as<bool>();
            if (root["target_brightness"])
                target_brightness = root["target_brightness"].as<double>();
            if (root["tolerance"])
                tolerance = root["tolerance"].as<double>();
            if (root["step_gain"])
                step_gain = root["step_gain"].as<double>();
            if (root["decay_step"])
                decay_step = root["decay_step"].as<double>();
            if (root["exposure_min"])
                exposure_min = root["exposure_min"].as<double>();
            if (root["exposure_max"])
                exposure_max = root["exposure_max"].as<double>();
            if (root["control_interval_ms"])
                control_interval_ms = root["control_interval_ms"].as<double>();

            return true;
        } catch (const YAML::Exception& e) {
            std::cerr << "加载 YAML 配置失败: " << e.what() << std::endl;
            return false;
        }
    }
};