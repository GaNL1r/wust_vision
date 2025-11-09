#pragma once
#include "tasks/auto_aim/armor_tracker/motion_models/factorypd.hpp"
#include "tasks/auto_aim/armor_tracker/motion_models/motion_modelypdv2.hpp"
#include "tasks/auto_aim/type.hpp"
#include "tasks/utils.hpp"
#include <wust_vl/common/utils/timer.hpp>
struct TargetConfig {
    int esekf_iter_num = 2;
    double qxyz_common = 100;
    double qyaw_common = 400;
    double qxyz_output = 10;
    double qyaw_output = 0.1;
    double q_r = 0;
    double q_l = 0;
    double q_h = 0;
    double q_outpost_dz = 0.0;
    double yp_r = 4e-3;
    double dis_r_front = 0.05;
    double dis_r_side = 0.07;
    double dis2_r_ratio = 0.01;
    double yaw_r_base_front = 0.09;
    double yaw_r_base_side = 0.09;
    double yaw_r_log_ratio = 0.005;
};
class Target {
public:
    Target();
    Target& operator=(const Target&) = default;
    Target(
        const armor::Armor& armor,
        const TargetConfig& target_config,
        double radius,
        int armor_num,
        Eigen::DiagonalMatrix<double, ypdv2armor_motion_model::X_N> p0
    );
    armor::ArmorNumber tracked_id_;
    std::string type_;
    Eigen::Matrix<double, ypdv2armor_motion_model::Z_N, 1> measurement_ =
        Eigen::Matrix<double, ypdv2armor_motion_model::Z_N, 1>::Zero();
    Eigen::Matrix<double, ypdv2armor_motion_model::X_N, 1> target_state_ =
        Eigen::Matrix<double, ypdv2armor_motion_model::X_N, 1>::Zero();

    double radius_pre_;
    double last_yaw_ = 0;
    double last_ypd_y = 0;
    int armor_num_ = 4;
    int switch_count_ = 0;
    int update_count_ = 0;
    bool is_switch_, is_converged_;
    bool jumped = false;
    int last_id;
    bool is_inited = false;
    bool is_tracking = false;
    bool is_temp_lost_ = false;
    std::chrono::steady_clock::time_point last_t_;
    std::chrono::steady_clock::time_point timestamp_;
    double dt_;
    ypdv2armor_motion_model::RobotStateESEKF esekf_ypd_;
    TargetConfig target_config_;

    void predict(
        std::chrono::steady_clock::time_point t,
        Eigen::Vector3d self_v = Eigen::Vector3d::Zero(),
        bool use_lin_pre = false
    );
    void
    predict(double dt, Eigen::Vector3d self_v = Eigen::Vector3d::Zero(), bool use_lin_pre = false);
    bool update(const armor::Armor& armor);
    Eigen::Matrix<double, ypdv2armor_motion_model::Z_N, ypdv2armor_motion_model::Z_N>
    computeMeasurementCovariance(const Eigen::Matrix<double, ypdv2armor_motion_model::Z_N, 1>& z
    ) const;
    Eigen::Matrix<double, ypdv2armor_motion_model::X_N, ypdv2armor_motion_model::X_N>
    computeProcessNoise(double dt) const;
    double orientationToYaw(const Eigen::Quaterniond& q) noexcept {
        double roll, pitch, yaw;
        Eigen::Vector3d euler = utils::quatToEuler(q, utils::EulerOrder::ZYX, false);
        yaw = euler[0];
        yaw = this->last_yaw_ + angles::shortest_angular_distance(this->last_yaw_, yaw);
        this->last_yaw_ = yaw;
        return yaw;
    }
    std::vector<double> getArmorYaws() const {
        std::vector<double> yaw_list;

        for (int i = 0; i < armor_num_; i++) {
            auto angle = angles::normalize_angle(target_state_[6] + i * 2 * CV_PI / armor_num_);
            yaw_list.push_back(angle);
        }
        return yaw_list;
    }
    Eigen::Vector3d position() const {
        return { target_state_[0], target_state_[2], target_state_[4] };
    }
    Eigen::Vector3d velocity() const {
        return { target_state_[1], target_state_[3], target_state_[5] };
    }
    std::vector<Eigen::Vector3d> getArmorPositions() const {
        std::vector<Eigen::Vector3d> armor_positions;

        for (int i = 0; i < armor_num_; i++) {
            auto angle = angles::normalize_angle(target_state_[6] + i * 2 * CV_PI / armor_num_);
            Eigen::Vector3d xyz = h_armor_xyz(target_state_, i);
            armor_positions.push_back(xyz);
        }
        return armor_positions;
    }
    std::vector<Eigen::Vector3d> getArmorVelocities() const {
        std::vector<Eigen::Vector3d> armor_velocities;

        for (int i = 0; i < armor_num_; i++) {
            auto angle = angles::normalize_angle(target_state_[6] + i * 2 * CV_PI / armor_num_);
            Eigen::Vector3d xyz = h_armor_vxyz(target_state_, i);
            armor_velocities.push_back(xyz);
        }
        return armor_velocities;
    }
    double yaw() const {
        return target_state_(6);
    }
    double v_yaw() const {
        return target_state_(7);
    }
    double r() const {
        return target_state_(8);
    }
    double l() const {
        return target_state_(9);
    }
    double h() const {
        return target_state_(10);
    }

    inline bool checkTargetAppear() {
        bool appear = is_tracking && time_utils::durationSec(timestamp_, time_utils::now()) < 5.0;
        return appear;
    }
    bool diverged() const {
        return diverged(target_state_);
    }
    bool diverged(Eigen::VectorXd target_state) const {
        auto r_ok = target_state[8] > 0.05 && target_state[8] < 0.5;
        auto l_ok =
            target_state[8] + target_state[9] > 0.05 && target_state[8] + target_state[9] < 0.5;
        if (tracked_id_ == armor::ArmorNumber::OUTPOST) {
            l_ok = true;
        }
        auto v_yaw_ok = std::abs(target_state[7]) < 30.0;
        Eigen::Vector3d vel = velocity();
        auto v_xyz_ok = std::abs(vel.norm()) < 10.0;
        auto pos_ok = position().norm() < 10.0 && position().norm() > 0.5;
        if (r_ok && l_ok && v_xyz_ok && v_yaw_ok && pos_ok)
            return false;

        return true;
    }
    std::vector<Eigen::Vector4d> getArmorPosAndYaw() const {
        std::vector<Eigen::Vector4d> _armor_xyza_list;

        for (int i = 0; i < armor_num_; i++) {
            auto angle = angles::normalize_angle(target_state_[6] + i * 2 * CV_PI / armor_num_);
            Eigen::Vector3d xyz = h_armor_xyz(target_state_, i);
            _armor_xyza_list.push_back({ xyz[0], xyz[1], xyz[2], angle });
        }
        return _armor_xyza_list;
    }
    Eigen::Vector3d h_armor_xyz(const Eigen::VectorXd& x, int id) const {
        auto angle = angles::normalize_angle(x[6] + id * 2 * CV_PI / armor_num_);
        auto use_l_h = (armor_num_ == 4) && (id == 1 || id == 3);
        auto outpost = armor_num_ == 3;

        auto r = (use_l_h) ? x[8] + x[9] : x[8];
        auto armor_x = x[0] - r * std::cos(angle);
        auto armor_y = x[2] - r * std::sin(angle);
        auto armor_z = (outpost) ? getoutpost_armor_z(id, x) : (use_l_h) ? x[4] + x[10] : x[4];

        return { armor_x, armor_y, armor_z };
    }
    double getoutpost_armor_z(int id, const Eigen::VectorXd x) const {
        return (id == 0) ? x[4] : (id == 1) ? x[4] + x[9] : (id == 2) ? x[4] + x[10] : x[4];
    }

    Eigen::Vector3d h_armor_vxyz(const Eigen::VectorXd& x, int id) const {
        Eigen::Vector3d v_center(x[1], x[3], x[5]);

        auto angle = angles::normalize_angle(x[6] + id * 2 * CV_PI / armor_num_);
        auto use_l_h = (armor_num_ == 4) && (id == 1 || id == 3);

        auto r = (use_l_h) ? x[8] + x[9] : x[8];

        Eigen::Vector3d p(-r * std::cos(angle), -r * std::sin(angle), (use_l_h ? x[10] : 0.0));

        Eigen::Vector3d omega(0.0, 0.0, x[7]);

        Eigen::Vector3d v_rot = omega.cross(p);
        return v_center + v_rot;
    }
};