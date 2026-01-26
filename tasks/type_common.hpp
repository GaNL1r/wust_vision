#pragma once
#include "pch.hpp"
#include "wust_vl/common/utils/motion_buffer.hpp"
#include "wust_vl/common/utils/parameter.hpp"
#include "wust_vl/common/utils/trajectory_compensator.hpp"

namespace wust_vision {
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
std::string enemyColorToString(EnemyColor color) noexcept;
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
AttackMode toAttackMode(int value) noexcept;
struct Motion {
    double yaw, pitch, roll; // 欧拉角 (rad)
    double vyaw, vpitch, vroll; // 角速度
    double vx, vy, vz; // 线速度
    static double unwrap_angle(double prev, double curr) noexcept {
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
    static double interp_angle(double a, double b, double t) noexcept {
        double diff = b - a;
        while (diff > M_PI)
            diff -= 2.0 * M_PI;
        while (diff < -M_PI)
            diff += 2.0 * M_PI;
        return a + diff * t;
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
    void predictSelf(double dt_sec) noexcept;
    double calYaw(double self_v_yaw = 0) const noexcept {
        return angles::normalize_angle(std::atan2(pos.y(), pos.x()) - self_v_yaw * dt);
    }

    double calRawPitch() const noexcept {
        return std::atan2(pos.z(), std::sqrt(pos.x() * pos.x() + pos.y() * pos.y()));
    }

    double distance() const noexcept {
        return pos.norm();
    }

    double calVYaw() const noexcept {
        double x = pos.x();
        double y = pos.y();
        double vx = vel.x();
        double vy = vel.y();

        double denom = x * x + y * y;
        if (denom < 1e-6)
            return 0.0;

        return (x * vy - y * vx) / denom;
    }
    double calHostVYaw() const noexcept {
        double x = host_pos.x();
        double y = host_pos.y();
        double vx = host_vel.x();
        double vy = host_vel.y();

        double denom = x * x + y * y;
        if (denom < 1e-6)
            return 0.0;

        return (x * vy - y * vx) / denom;
    }

    double calVPitch() const noexcept {
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
    void tf(Eigen::Matrix4d T_camera_to_odom) noexcept;
    std::vector<cv::Point2f>
    toPts(const cv::Mat& camera_intrinsic, const cv::Mat& camera_distortion) noexcept;
};
struct GimbalCmd {
    std::chrono::steady_clock::time_point timestamp;
    double raw_yaw = 0;
    double raw_pitch = 0;
    double pitch = 0;
    double yaw = 0;
    double target_yaw = 0;
    double target_pitch = 0;
    double v_yaw = 0;
    double v_pitch = 0;
    double a_yaw = 0;
    double a_pitch = 0;
    double distance = -1;
    bool fire_advice = false;
    double enable_yaw_diff = 0;
    double enable_pitch_diff = 0;
    double fly_time = 0;
    bool appera = false;
    AimTarget aim_target;
    double score = 0;
    inline bool isValid() const noexcept {
        auto bad = [](double x) { return std::isnan(x) || std::isinf(x); };

        if (bad(raw_yaw) || bad(raw_pitch) || bad(pitch) || bad(yaw) || bad(target_yaw)
            || bad(target_pitch) || bad(target_pitch) || bad(v_yaw) || bad(v_pitch) || bad(distance)
            || bad(enable_yaw_diff) || bad(enable_pitch_diff))
            return false;

        return true;
    }
};

struct AutoExposureCfg: wust_vl::common::utils::ParamGroup {
    static constexpr const char* Logger = "Config: auto_exposure";
    static constexpr const char* kKey = "auto_exposure";
    const char* key() const override {
        return kKey;
    }

    using Ptr = std::shared_ptr<AutoExposureCfg>;
    AutoExposureCfg() {}
    static Ptr create() {
        return std::make_shared<AutoExposureCfg>();
    }
    GEN_PARAM(bool, enable);
    GEN_PARAM(double, target_brightness);
    GEN_PARAM(double, step_gain);
    GEN_PARAM(double, decay_step);
    GEN_PARAM(double, tolerance);
    GEN_PARAM(double, exposure_min);
    GEN_PARAM(double, exposure_max);
    GEN_PARAM(double, control_interval_ms);
    void loadSelf(const YAML::Node& node) override {
        enable_param.load(node);
        target_brightness_param.load(node);
        step_gain_param.load(node);
        decay_step_param.load(node);
        tolerance_param.load(node);
        exposure_min_param.load(node);
        exposure_max_param.load(node);
        control_interval_ms_param.load(node);
    }
};
struct TFConfig: wust_vl::common::utils::ParamGroup {
public:
    static constexpr const char* kKey = "tf";
    static constexpr const char* Logger = "Config: common::tf";
    const char* key() const override {
        return kKey;
    }
    using Ptr = std::shared_ptr<TFConfig>;
    TFConfig() {}
    static Ptr create() {
        return std::make_shared<TFConfig>();
    }

    bool first_load = false;

    Eigen::Matrix3d R_camera2gimbal;
    Eigen::Vector3d t_camera2gimbal;
    void loadSelf(const YAML::Node& node) override {
        if (!first_load) {
            auto t_vec = node["t_camera2gimbal"].as<std::vector<double>>();
            if (t_vec.size() != 3) {
                throw std::runtime_error("YAML tf.t_camera2gimbal must have 3 elements");
            }
            t_camera2gimbal = Eigen::Vector3d(t_vec[0], t_vec[1], t_vec[2]);

            auto R_vec = node["R_camera2gimbal"].as<std::vector<double>>();
            if (R_vec.size() != 9) {
                throw std::runtime_error("YAML tf.R_camera2gimbal must have 9 elements");
            }
            R_camera2gimbal =
                Eigen::Map<const Eigen::Matrix<double, 3, 3, Eigen::RowMajor>>(R_vec.data());
            first_load = true;
        } else {
        }
    }
};
struct TrajectoryCompensatorConfig: public wust_vl::common::utils::ParamGroup {
    static constexpr const char* Logger = "Config: auto_aim::trajectory_compensator";
    static constexpr const char* kKey = "trajectory_compensator";
    const char* key() const override {
        return kKey;
    }
    using Ptr = std::shared_ptr<TrajectoryCompensatorConfig>;
    TrajectoryCompensatorConfig() {}
    static Ptr create() {
        return std::make_shared<TrajectoryCompensatorConfig>();
    }
    std::shared_ptr<wust_vl::common::utils::TrajectoryCompensator> trajectory_compensator;
    bool first_load = false;
    void loadSelf(const YAML::Node& node) override

    {
        if (!first_load) {
            std::string comp_type = node["compenstator_type"].as<std::string>("ideal");
            trajectory_compensator =
                wust_vl::common::utils::CompensatorFactory::createCompensator(comp_type);
            trajectory_compensator->load(node);
            first_load = true;
        } else {
        }
    }
};
} // namespace wust_vision
template<>
struct wust_vl::common::utils::MotionTraits<wust_vision::Motion> {
    static void unwrap(const wust_vision::Motion& prev, wust_vision::Motion& curr) noexcept {
        curr.yaw = wust_vision::Motion::unwrap_angle(prev.yaw, curr.yaw);
        curr.pitch = wust_vision::Motion::unwrap_angle(prev.pitch, curr.pitch);
        curr.roll = wust_vision::Motion::unwrap_angle(prev.roll, curr.roll);
        // 速度部分不需要 unwrap
    }

    static wust_vision::Motion
    interpolate(const wust_vision::Motion& a, const wust_vision::Motion& b, double t) noexcept {
        wust_vision::Motion out;
        // 欧拉角 wrap-safe 插值
        out.yaw = wust_vision::Motion::interp_angle(a.yaw, b.yaw, t);
        out.pitch = wust_vision::Motion::interp_angle(a.pitch, b.pitch, t);
        out.roll = wust_vision::Motion::interp_angle(a.roll, b.roll, t);

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