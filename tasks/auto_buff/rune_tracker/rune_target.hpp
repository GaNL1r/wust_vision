#pragma once

#include "spd_fitter.hpp"
#include "tasks/auto_buff/rune_tracker/motion_models/motion_modelrypd.hpp"
#include "tasks/auto_buff/type.hpp"
#include "wust_vl/common/utils/math.hpp"
#include <wust_vl/common/utils/timer.hpp>
namespace rune {
struct RuneTargetConfig {
    int esekf_iter_num = 2;
    double q_roll = 1;
    double q_xyz = 0.5;
    double q_yaw = 0.1;
    double yp_r = 0.01;
    double dis_r = 0.05;
    double yaw_r = 0.1;
    double roll_r = 0.1;
    double match_gate = 10;
    double lost_dt = 0.5;
    // 从 YAML::Node 加载配置
    void loadFromYaml(const YAML::Node& node) {
        if (node["esekf_iter_num"])
            esekf_iter_num = node["esekf_iter_num"].as<int>();
        if (node["q_roll"])
            q_roll = node["q_roll"].as<double>();
        if (node["q_xyz"])
            q_xyz = node["q_xyz"].as<double>();
        if (node["q_yaw"])
            q_yaw = node["q_yaw"].as<double>();
        if (node["yp_r"])
            yp_r = node["yp_r"].as<double>();
        if (node["dis_r"])
            dis_r = node["dis_r"].as<double>();
        if (node["yaw_r"])
            yaw_r = node["yaw_r"].as<double>();
        if (node["roll_r"])
            roll_r = node["roll_r"].as<double>();
        if (node["match_gate"])
            match_gate = node["match_gate"].as<double>();
        if (node["lost_time_thres"])
            lost_dt = node["lost_time_thres"].as<double>();
    }
};
class RuneTarget {
public:
    RuneTarget() = default;
    RuneTarget& operator=(const RuneTarget&) = default;
    RuneTarget(const rune::RuneFan& fan, const RuneTargetConfig& target_config, double pre_v_roll);
    bool is_big_ = false;
    double last_yaw_ = 0;
    double last_ypd_y = 0;
    double last_roll_ = 0;
    bool is_inited = false;
    bool is_tracking = false;
    bool is_temp_lost_ = false;
    int last_id;
    std::vector<int> update_ids;
    RuneTargetConfig target_config_;
    std::chrono::steady_clock::time_point last_t_;
    std::chrono::steady_clock::time_point timestamp_;
    std::chrono::steady_clock::time_point start_time_;
    double dt_;
    double last_time_ = 0;
    SinSpeedFitter fitter_;
    ypdrune_motion_model::RuneESKF esekf_ypd_;
    Eigen::Matrix<double, ypdrune_motion_model::Z_N, 1> measurement_ =
        Eigen::Matrix<double, ypdrune_motion_model::Z_N, 1>::Zero();
    Eigen::Matrix<double, ypdrune_motion_model::X_N, 1> target_state_ =
        Eigen::Matrix<double, ypdrune_motion_model::X_N, 1>::Zero();
    cv::Rect expanded(
        Eigen::Matrix4d T_camera_to_odom,
        const cv::Mat& camera_intrinsic,
        const cv::Mat& camera_distortion,
        const cv::Size& image_size
    );
    bool update(const rune::RuneFan& fan);
    void predict(std::chrono::steady_clock::time_point t);
    void predict(double dt);
    Eigen::Matrix<double, ypdrune_motion_model::Z_N, ypdrune_motion_model::Z_N>
    computeMeasurementCovariance(const Eigen::Matrix<double, ypdrune_motion_model::Z_N, 1>& z
    ) const;
    Eigen::Matrix<double, ypdrune_motion_model::X_N, ypdrune_motion_model::X_N>
    computeProcessNoise(double dt) const;
    double orientationToYaw(const Eigen::Quaterniond& q) noexcept {
        double roll, pitch, yaw;
        Eigen::Vector3d euler = utils::quatToEuler(q, utils::EulerOrder::ZYX, false);
        yaw = euler[0];
        yaw = this->last_yaw_ + angles::shortest_angular_distance(this->last_yaw_, yaw);
        this->last_yaw_ = yaw;
        return yaw;
    }
    double orientationToRoll(const Eigen::Quaterniond& q) noexcept {
        double roll, pitch, yaw;
        Eigen::Vector3d euler = utils::quatToEuler(q, utils::EulerOrder::ZYX, false);
        roll = euler[2];
        roll = this->last_roll_ + angles::shortest_angular_distance(this->last_roll_, roll);
        this->last_roll_ = roll;
        return roll;
    }
    inline bool checkTargetAppear() {
        bool appear = is_tracking
            && time_utils::durationSec(timestamp_, time_utils::now()) < target_config_.lost_dt;
        return appear;
    }
    double predictAngle(std::chrono::steady_clock::time_point t) const {
        double to_start = time_utils::durationSec(start_time_, t);
        return fitter_.predictAngle(to_start);
    }
    double predictAngle(double dt) const {
        return fitter_.predictAngle(last_time_ + dt);
    }
    void predictWithFitter(std::chrono::steady_clock::time_point t) {
        if (is_big_) {
            double to_start = time_utils::durationSec(start_time_, t);
            double angle = fitter_.predictAngle(to_start);
            double speed = fitter_.predictSpeed(to_start);
            auto state = esekf_ypd_.getState();
            state[(int)ypdrune_motion_model::State::ROLL] = angles::normalize_angle(angle);
            state[(int)ypdrune_motion_model::State::VROLL] = speed;
            esekf_ypd_.setState(state);
        } else {
            predict(t);
        }
    }
    double getFitterSpd(std::chrono::steady_clock::time_point t) {
        double to_start = time_utils::durationSec(start_time_, t);
        return fitter_.predictSpeed(to_start);
    }

    Eigen::Vector3d centerPos() const {
        return { target_state_((int)ypdrune_motion_model::State::CX),
                 target_state_((int)ypdrune_motion_model::State::CY),
                 target_state_((int)ypdrune_motion_model::State::CZ) };
    }
    std::vector<double> getAngles() {
        std::vector<double> angles;
        for (int i = 0; i < 5; i++) {
            auto angle = angles::normalize_angle(
                target_state_[(int)ypdrune_motion_model::State::ROLL] + i * 2 * M_PI / 5
            );
            angles.push_back(angle);
        }
        return angles;
    }
    bool diverged() const {
        return diverged(target_state_);
    }
    bool diverged(Eigen::VectorXd target_state) const {
        return false;
    }
    double roll() const {
        return target_state_[(int)ypdrune_motion_model::State::ROLL];
    }
    double curr_roll() const {
        return roll() + last_id * 2 * M_PI / 5;
    }
    double real_roll(int id) const {
        return roll() + id * 2 * M_PI / 5;
    }
    double yaw() const {
        return target_state_[(int)ypdrune_motion_model::State::YAW];
    }
    double v_roll() const {
        return target_state_[(int)ypdrune_motion_model::State::VROLL];
    }
    std::vector<std::pair<int, rune::RuneFan::Simple>>
    match(const std::vector<rune::RuneFan::Simple>& fans) {
        std::vector<std::pair<int, rune::RuneFan::Simple>> result;
        const int n_obs = (int)(fans.size());
        const int armors_num = 5;
        const double GATE = target_config_.match_gate;
        const double max_cost = 1e9;
        std::vector<std::vector<double>> cost(n_obs, std::vector<double>(armors_num, max_cost + 1));
        std::vector<ypdrune_motion_model::VecZ> meas_list(n_obs);
        for (int j = 0; j < n_obs; ++j) {
            meas_list[j] = getmean(fans[j]);
        }
        for (int j = 0; j < n_obs; ++j) {
            for (int id = 0; id < armors_num; ++id) {
                ypdrune_motion_model::Measure measure(id);
                ypdrune_motion_model::VecZ z_pred;
                measure.h(target_state_, z_pred);

                ypdrune_motion_model::VecZ nu = meas_list[j] - z_pred;
                nu[(int)ypdrune_motion_model::Mean::YPD_Y] =
                    angles::normalize_angle(nu[(int)ypdrune_motion_model::Mean::YPD_Y]);
                nu[(int)ypdrune_motion_model::Mean::YPD_P] =
                    angles::normalize_angle(nu[(int)ypdrune_motion_model::Mean::YPD_P]);
                nu[(int)ypdrune_motion_model::Mean::ORI_YAW] =
                    angles::normalize_angle(nu[(int)ypdrune_motion_model::Mean::ORI_YAW]);
                nu[(int)ypdrune_motion_model::Mean::ORI_ROLL] =
                    angles::normalize_angle(nu[(int)ypdrune_motion_model::Mean::ORI_ROLL]);
                auto R = computeMeasurementCovariance(z_pred);
                auto Rinv = R.inverse();

                double d2 = (nu.transpose() * Rinv * nu)(0, 0);

                // 门控
                if (std::isfinite(d2) && d2 < GATE) {
                    cost[j][id] = d2;
                }
            }
        }
        std::vector<bool> used_obs(n_obs, false);
        std::vector<bool> used_id(armors_num, false);

        while (true) {
            double best = max_cost;
            int best_j = -1;
            int best_id = -1;

            for (int j = 0; j < n_obs; ++j) {
                if (used_obs[j])
                    continue;
                for (int id = 0; id < armors_num; ++id) {
                    if (used_id[id])
                        continue;
                    if (cost[j][id] < best) {
                        best = cost[j][id];
                        best_j = j;
                        best_id = id;
                    }
                }
            }

            if (best_j < 0 || best_id < 0) {
                break;
            }

            used_obs[best_j] = true;
            used_id[best_id] = true;
            result.push_back(std::make_pair(best_id, fans[best_j]));
        }

        // for (auto fan: fans) {
        //     int id;
        //     auto min_angle_error = 1e10;
        //     const auto angles = getAngles();
        //     for (int i = 0; i < angles.size(); i++) {
        //         auto angle_error = std::abs(angles::normalize_angle(
        //             angles::normalize_angle(orientationToRoll(fan.target_ori)) - angles[i]
        //         ));
        //         if (angle_error < min_angle_error) {
        //             min_angle_error = angle_error;
        //             id = i;
        //         }
        //     }
        //     result.push_back(std::make_pair(id, fan));
        // }
        return result;
    }
    std::vector<std::pair<Eigen::Vector3d, Eigen::Quaterniond>> getAllPose() const {
        std::vector<std::pair<Eigen::Vector3d, Eigen::Quaterniond>> poses;
        for (int i = 0; i < 5; i++) {
            poses.emplace_back(getPose(i));
        }
        return poses;
    }
    std::pair<Eigen::Vector3d, Eigen::Quaterniond> getPose(int id) const {
        Eigen::Vector3d euler = Eigen::Vector3d(yaw(), 0.0, real_roll(id));
        auto q = utils::eulerToQuat(euler, utils::EulerOrder::ZYX);
        return computeBladeTipPose(centerPos(), q, id);
    }

    std::pair<Eigen::Vector3d, Eigen::Quaterniond>
    computeBladeTipPose(const Eigen::Vector3d& center_pos, const Eigen::Quaterniond& q, int id)
        const {
        // tip 的局部坐标（沿 local X 方向）
        Eigen::Vector3d local_tip(-0.15, 0.0, 0.7);

        Eigen::Vector3d tip_pos = center_pos + q * local_tip;

        Eigen::Vector3d euler = Eigen::Vector3d(yaw(), 0.0, real_roll(id));
        return { tip_pos, utils::eulerToQuat(euler, utils::EulerOrder::ZYX) };
    }
    std::pair<Eigen::Vector3d, Eigen::Quaterniond> getHitPoint() const {
        return getPose(last_id);
    }
    rune::PowerRune getPowerRune() const {
        rune::PowerRune power_rune;
        if (!is_inited) {
            return power_rune;
        }
        power_rune.center.pos = centerPos();
        Eigen::Vector3d euler = Eigen::Vector3d(yaw(), 0.0, real_roll(last_id));
        auto q = Eigen::Quaterniond();
        power_rune.center.ori = q;
        auto all_pose = getAllPose();
        for (int i = 0; i < all_pose.size(); i++) {
            rune::PowerRune::Pose pose;
            pose.pos = all_pose[i].first;
            pose.ori = all_pose[i].second;
            power_rune.fans.push_back(pose);
        }
        power_rune.hit_id = last_id;
        return power_rune;
    }
    Eigen::Matrix<double, ypdrune_motion_model::Z_N, 1> getmean(const rune::RuneFan::Simple& fan) {
        auto p = fan.target_pos;

        double measured_yaw = orientationToYaw(fan.target_ori);
        double measured_roll = orientationToRoll(fan.target_ori);
        double ypd_y = std::atan2(p.y(), p.x());
        ypd_y = this->last_ypd_y + angles::shortest_angular_distance(this->last_ypd_y, ypd_y);
        this->last_ypd_y = ypd_y;
        double ypd_p = std::atan2(p.z(), std::sqrt(p.x() * p.x() + p.y() * p.y()));
        double ypd_d = std::sqrt(p.x() * p.x() + p.y() * p.y() + p.z() * p.z());
        Eigen::Matrix<double, ypdrune_motion_model::Z_N, 1> measure;
        measure << ypd_y, ypd_p, ypd_d, measured_yaw, measured_roll;
        return measure;
    }
};
} // namespace rune
