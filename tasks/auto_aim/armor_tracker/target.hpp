#pragma once
#include "tasks/auto_aim/armor_tracker/motion_models/motion_modelypdv2.hpp"
#include "tasks/auto_aim/type.hpp"
#include "tasks/utils.hpp"
#include <wust_vl/common/utils/timer.hpp>
namespace MModel = ypdv2armor_motion_model;

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
    double match_gate = 10;
    double lost_dt = 0.5;
    double grow_k = 0.5;
    void loadConfig(const YAML::Node& node) {
        esekf_iter_num = node["esekf_iter_num"].as<int>();
        qyaw_common = node["qyaw_common"].as<double>();
        qxyz_common = node["qxyz_common"].as<double>();
        qxyz_output = node["qxyz_output"].as<double>();
        qyaw_output = node["qyaw_output"].as<double>();
        q_l = node["q_l"].as<double>();
        q_h = node["q_h"].as<double>();
        q_r = node["q_r"].as<double>();
        q_outpost_dz = node["q_outpost_dz"].as<double>();
        yp_r = node["yp_r"].as<double>();
        dis_r_front = node["dis_r_front"].as<double>();
        dis_r_side = node["dis_r_side"].as<double>();
        dis2_r_ratio = node["dis2_r_ratio"].as<double>();
        yaw_r_base_front = node["yaw_r_base_front"].as<double>();
        yaw_r_base_side = node["yaw_r_base_side"].as<double>();
        yaw_r_log_ratio = node["yaw_r_log_ratio"].as<double>();
        match_gate = node["match_gate"].as<double>();
        lost_dt = node["lost_time_thres"].as<double>();
        grow_k = node["grow_k"].as<double>();
    }
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
        Eigen::DiagonalMatrix<double, MModel::X_N> p0
    );
    MModel::Measure::MeasureCtx ctx_;
    armor::ArmorNumber tracked_id_;
    std::string type_;
    MModel::VecZ measurement_ = Eigen::Matrix<double, MModel::Z_N, 1>::Zero();
    MModel::VecX target_state_ = Eigen::Matrix<double, MModel::X_N, 1>::Zero();

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
    MModel::RobotStateESEKF esekf_ypd_;
    TargetConfig target_config_;
    cv::Rect expanded(
        Eigen::Matrix4d T_camera_to_odom,
        const cv::Mat& camera_intrinsic,
        const cv::Mat& camera_distortion,
        const cv::Size& image_size
    );
    void predict(
        std::chrono::steady_clock::time_point t,
        Eigen::Vector3d self_v = Eigen::Vector3d::Zero()
    );
    void predict(double dt, Eigen::Vector3d self_v = Eigen::Vector3d::Zero());
    bool update(const std::pair<int, armor::Armor>& armor);
    Eigen::Matrix<double, MModel::Z_N, MModel::Z_N>
    computeMeasurementCovariance(const Eigen::Matrix<double, MModel::Z_N, 1>& z) const;
    Eigen::Matrix<double, MModel::X_N, MModel::X_N> computeProcessNoise(double dt) const;

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
        return { target_state_[(int)MModel::State::CX],
                 target_state_[(int)MModel::State::CY],
                 target_state_[(int)MModel::State::CZ] };
    }
    Eigen::Vector3d velocity() const {
        return { target_state_[(int)MModel::State::VCX],
                 target_state_[(int)MModel::State::VCY],
                 target_state_[(int)MModel::State::VCZ] };
    }
    std::vector<Eigen::Vector3d> getArmorPositions() const {
        std::vector<Eigen::Vector3d> armor_positions;

        for (int i = 0; i < armor_num_; i++) {
            auto angle = angles::normalize_angle(
                target_state_[(int)MModel::State::YAW] + i * 2 * CV_PI / armor_num_
            );
            Eigen::Vector3d xyz = h_armor_xyz(target_state_, i);
            armor_positions.push_back(xyz);
        }
        return armor_positions;
    }
    std::vector<Eigen::Vector3d> getArmorVelocities() const {
        std::vector<Eigen::Vector3d> armor_velocities;

        for (int i = 0; i < armor_num_; i++) {
            auto angle = angles::normalize_angle(
                target_state_[(int)MModel::State::YAW] + i * 2 * CV_PI / armor_num_
            );
            Eigen::Vector3d xyz = h_armor_vxyz(target_state_, i);
            armor_velocities.push_back(xyz);
        }
        return armor_velocities;
    }
    std::vector<std::pair<int, armor::Armor>> match(const std::vector<armor::Armor>& armors);
    double yaw() const {
        return target_state_((int)MModel::State::YAW);
    }
    double v_yaw() const {
        return target_state_((int)MModel::State::VYAW);
    }
    double r() const {
        return target_state_((int)MModel::State::R);
    }
    double l() const {
        return target_state_((int)MModel::State::L);
    }
    double h() const {
        return target_state_((int)MModel::State::H);
    }

    inline bool checkTargetAppear() {
        bool appear = is_tracking
            && time_utils::durationSec(timestamp_, time_utils::now()) < target_config_.lost_dt;
        return appear;
    }
    bool diverged() const {
        return diverged(target_state_);
    }
    bool diverged(Eigen::VectorXd target_state) const {
        auto r_ok =
            target_state[(int)MModel::State::R] > 0.05 && target_state[(int)MModel::State::R] < 0.5;
        auto l_ok = target_state[(int)MModel::State::R] + target_state[(int)MModel::State::L] > 0.05
            && target_state[(int)MModel::State::R] + target_state[(int)MModel::State::L] < 0.5;
        if (tracked_id_ == armor::ArmorNumber::OUTPOST) {
            l_ok = true;
        }
        auto v_yaw_ok = std::abs(target_state[(int)MModel::State::VYAW]) < 30.0;
        Eigen::Vector3d vel = velocity();
        auto v_xyz_ok = std::abs(vel.norm()) < 10.0;
        auto pos_ok = position().norm() < 10.0 && position().norm() > 0.5;
        if (r_ok && l_ok && v_xyz_ok && v_yaw_ok && pos_ok)
            return false;

        return true;
    }
    inline void clampState(Eigen::VectorXd& state) noexcept {
        for (int i = 0; i < 3; i++) {
            state[i] = std::clamp(state[i], -10.0, 10.0);
        }

        for (int i = 3; i < 6; i++) {
            state[i] = std::clamp(state[i], -10.0, 10.0);
        }

        // state[6] = std::remainder(state[6], 2 * M_PI);
        state[(int)MModel::State::VYAW] = std::clamp(state[(int)MModel::State::VYAW], -30.0, 30.0);
        state[(int)MModel::State::R] = std::clamp(state[(int)MModel::State::R], 0.05, 0.5);
        state[(int)MModel::State::L] = std::clamp(state[(int)MModel::State::L], -0.45, 0.45);

        if (tracked_id_ == armor::ArmorNumber::OUTPOST) {
            state[(int)MModel::State::R] = std::clamp(state[(int)MModel::State::R], 0.05, 0.5);
        }

        double r_plus_l = state[(int)MModel::State::R] + state[(int)MModel::State::L];
        if (r_plus_l < 0.05) {
            state[(int)MModel::State::L] = 0.05 - state[(int)MModel::State::R];
        } else if (r_plus_l > 0.5) {
            state[(int)MModel::State::L] = 0.5 - state[(int)MModel::State::R];
        }
        esekf_ypd_.setState(state);
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
        auto angle =
            angles::normalize_angle(x[(int)MModel::State::YAW] + id * 2 * CV_PI / armor_num_);
        auto use_l_h = (armor_num_ == 4) && (id == 1 || id == 3);
        auto outpost = armor_num_ == 3;

        auto r = (use_l_h) ? x[(int)MModel::State::R] + x[(int)MModel::State::L]
                           : x[(int)MModel::State::R];
        auto armor_x = x[(int)MModel::State::CX] - r * std::cos(angle);
        auto armor_y = x[(int)MModel::State::CY] - r * std::sin(angle);
        auto armor_z = (outpost) ? getoutpost_armor_z(id, x)
            : (use_l_h)          ? x[(int)MModel::State::CZ] + x[(int)MModel::State::H]
                                 : x[(int)MModel::State::CZ];

        return { armor_x, armor_y, armor_z };
    }
    double getoutpost_armor_z(int id, const Eigen::VectorXd x) const {
        return (id == 0) ? x[(int)MModel::State::CZ]
            : (id == 1)  ? x[(int)MModel::State::CZ] + x[(int)MModel::State::outpost01DZ]
            : (id == 2)  ? x[(int)MModel::State::CZ] + x[(int)MModel::State::outpost02DZ]
                         : x[(int)MModel::State::CZ];
    }

    Eigen::Vector3d h_armor_vxyz(const Eigen::VectorXd& x, int id) const {
        Eigen::Vector3d v_center(
            x[(int)MModel::State::CX],
            x[(int)MModel::State::CY],
            x[(int)MModel::State::CZ]
        );

        auto angle =
            angles::normalize_angle(x[(int)MModel::State::YAW] + id * 2 * CV_PI / armor_num_);
        auto use_l_h = (armor_num_ == 4) && (id == 1 || id == 3);

        auto r = (use_l_h) ? x[(int)MModel::State::R] + x[(int)MModel::State::L]
                           : x[(int)MModel::State::R];

        Eigen::Vector3d p(
            -r * std::cos(angle),
            -r * std::sin(angle),
            (use_l_h ? x[(int)MModel::State::H] : 0.0)
        );

        Eigen::Vector3d omega(0.0, 0.0, x[(int)MModel::State::VYAW]);

        Eigen::Vector3d v_rot = omega.cross(p);
        return v_center + v_rot;
    }
    Eigen::Matrix<double, MModel::Z_N, 1> getmean(const armor::Armor& a) {
        auto p = a.target_pos;
        double measured_yaw = orientationToYaw(a.target_ori);
        double ypd_y = std::atan2(p.y(), p.x());
        ypd_y = this->last_ypd_y + angles::shortest_angular_distance(this->last_ypd_y, ypd_y);
        this->last_ypd_y = ypd_y;
        double ypd_p = std::atan2(p.z(), std::sqrt(p.x() * p.x() + p.y() * p.y()));
        double ypd_d = std::sqrt(p.x() * p.x() + p.y() * p.y() + p.z() * p.z());

        return Eigen::Vector4d(ypd_y, ypd_p, ypd_d, measured_yaw);
    }
};