// Copyright 2025 Xiaojian Wu
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
#include "tracker_manager.hpp"
#include "tasks/auto_aim/type.hpp"
#include "tasks/utils.hpp"
TrackerManager::TrackerManager(
    const YAML::Node& config_,
    std::shared_ptr<ConfigBinder> config_binder
) {
    config_binder_ = config_binder;
    track_one_num_ = config_["armor_tracker"]["track_one_num"].as<int>(2);

    double max_match_distance = config_["armor_tracker"]["max_match_distance"].as<double>(0.2);
    double max_match_yaw_diff = config_["armor_tracker"]["max_match_yaw_diff"].as<double>(1.0);
    double max_match_z_diff = config_["armor_tracker"]["max_match_z_diff"].as<double>(0.1);
    double jump_thresh = config_["armor_tracker"]["jump_thresh"].as<double>(0.4);
    tracker_ = std::make_unique<Tracker>(
        max_match_distance,
        max_match_yaw_diff,
        max_match_z_diff,
        jump_thresh
    );

    for (int i = 0; i < track_one_num_; i++) {
        auto o_tracker_ = std::make_unique<OneTracker>(
            max_match_distance,
            max_match_yaw_diff,
            max_match_z_diff,
            jump_thresh
        );
        o_tracker_->tracking_thres_ = config_["armor_tracker"]["one_tracking_thres"].as<int>(1);
        one_trackers_.push_back(std::move(o_tracker_));
    }

    v_yaw_to_one_thres_high_ = config_["armor_tracker"]["v_yaw_to_one_thres_high"].as<float>(1.0);
    v_yaw_to_one_thres_low_ = config_["armor_tracker"]["v_yaw_to_one_thres_low"].as<float>(0.7);

    tracker_->tracking_thres_ = config_["armor_tracker"]["tracking_thres"].as<int>(5);

    lost_time_thres_ = config_["armor_tracker"]["lost_time_thres"].as<double>();
    one_lost_time_thres_ = config_["armor_tracker"]["one_lost_time_thres"].as<double>(0.1);

    // iteration_num
    bindConfig(config_binder_, { "armor_tracker", "ekf", "iteration_num" }, &iteration_num_, 1);
    bool use_esekf = config_["armor_tracker"]["ekf"]["use_esekf"].as<bool>(false);
    bool fusion_esekf_ekf = config_["armor_tracker"]["ekf"]["fusion_esekf_ekf"].as<bool>(false);
    tracker_->use_esekf_ = use_esekf;

    // EKF 噪声参数
    bindConfig(config_binder_, { "armor_tracker", "ekf", "ys2qx_a" }, &ys2qx_a_, 20.0);
    bindConfig(config_binder_, { "armor_tracker", "ekf", "ys2qy_a" }, &ys2qy_a_, 20.0);
    bindConfig(config_binder_, { "armor_tracker", "ekf", "ys2qz_a" }, &ys2qz_a_, 20.0);
    bindConfig(config_binder_, { "armor_tracker", "ekf", "ys2qyaw_a" }, &ys2qyaw_a_, 100.0);
    bindConfig(config_binder_, { "armor_tracker", "ekf", "ys2qr_a" }, &ys2qr_a_, 800.0);
    bindConfig(config_binder_, { "armor_tracker", "ekf", "ys2qd_zc_a" }, &ys2qd_zc_a_, 800.0);

    bindConfig(config_binder_, { "armor_tracker", "ekf", "yr_y_a" }, &yr_y_a_, 0.05);
    bindConfig(config_binder_, { "armor_tracker", "ekf", "yr_p_a" }, &yr_p_a_, 0.05);
    bindConfig(config_binder_, { "armor_tracker", "ekf", "yr_d_front_a" }, &yr_d_front_a_, 0.05);
    bindConfig(config_binder_, { "armor_tracker", "ekf", "yr_d_side_a" }, &yr_d_side_a_, 0.05);
    bindConfig(
        config_binder_,
        { "armor_tracker", "ekf", "yr_yaw_front_a" },
        &yr_yaw_front_a_,
        0.02
    );
    bindConfig(config_binder_, { "armor_tracker", "ekf", "yr_yaw_side_a" }, &yr_yaw_side_a_, 0.02);

    bindConfig(config_binder_, { "armor_tracker", "ekf", "ys2qx_c" }, &ys2qx_c_, 20.0);
    bindConfig(config_binder_, { "armor_tracker", "ekf", "ys2qy_c" }, &ys2qy_c_, 20.0);
    bindConfig(config_binder_, { "armor_tracker", "ekf", "ys2qz_c" }, &ys2qz_c_, 20.0);
    bindConfig(config_binder_, { "armor_tracker", "ekf", "ys2qyaw_c" }, &ys2qyaw_c_, 100.0);
    bindConfig(config_binder_, { "armor_tracker", "ekf", "ys2qr_c" }, &ys2qr_c_, 800.0);
    bindConfig(config_binder_, { "armor_tracker", "ekf", "ys2qd_zc_c" }, &ys2qd_zc_c_, 800.0);

    bindConfig(config_binder_, { "armor_tracker", "ekf", "yr_y_c" }, &yr_y_c_, 0.05);
    bindConfig(config_binder_, { "armor_tracker", "ekf", "yr_p_c" }, &yr_p_c_, 0.05);
    bindConfig(config_binder_, { "armor_tracker", "ekf", "yr_d_front_c" }, &yr_d_front_c_, 0.05);
    bindConfig(config_binder_, { "armor_tracker", "ekf", "yr_d_side_c" }, &yr_d_side_c_, 0.05);
    bindConfig(
        config_binder_,
        { "armor_tracker", "ekf", "yr_yaw_front_c" },
        &yr_yaw_front_c_,
        0.02
    );
    bindConfig(config_binder_, { "armor_tracker", "ekf", "yr_yaw_side_c" }, &yr_yaw_side_c_, 0.02);

    bindConfig(config_binder_, { "armor_tracker", "ekf", "oys2qx" }, &oys2qx_, 20.0);
    bindConfig(config_binder_, { "armor_tracker", "ekf", "oys2qy" }, &oys2qy_, 20.0);
    bindConfig(config_binder_, { "armor_tracker", "ekf", "oys2qz" }, &oys2qz_, 20.0);
    bindConfig(config_binder_, { "armor_tracker", "ekf", "oys2qyaw" }, &oys2qyaw_, 100.0);

    bindConfig(config_binder_, { "armor_tracker", "ekf", "oyr_y" }, &oyr_y_, 0.05);
    bindConfig(config_binder_, { "armor_tracker", "ekf", "oyr_p" }, &oyr_p_, 0.05);
    bindConfig(config_binder_, { "armor_tracker", "ekf", "oyr_d_front" }, &oyr_d_front_, 0.05);
    bindConfig(config_binder_, { "armor_tracker", "ekf", "oyr_d_side" }, &oyr_d_side_, 0.05);
    bindConfig(config_binder_, { "armor_tracker", "ekf", "oyr_yaw_front" }, &oyr_yaw_front_, 0.02);
    bindConfig(config_binder_, { "armor_tracker", "ekf", "oyr_yaw_side" }, &oyr_yaw_side_, 0.02);

    bindConfig(config_binder_, { "armor_tracker", "ekf", "r_v" }, &r_v_, 0.01);
    bindConfig(config_binder_, { "armor_tracker", "ekf", "q_v" }, &q_v_, 0.01);
    bindConfig(config_binder_, { "armor_tracker", "ekf", "q_a" }, &q_a_, 0.01);

    // EKF 状态预测函数
    auto acc_f = acc_model::Predict(0.005);
    auto yf = ypdarmor_motion_model::Predict(0.005);
    auto oyf = oneypdarmor_motion_model::Predict(0.005);

    // EKF 观测函数
    auto acc_h = acc_model::Measure();
    auto yh = ypdarmor_motion_model::Measure();
    auto oyh = oneypdarmor_motion_model::Measure();

    // EKF 过程噪声协方差 Q
    auto acc_q = [this]() {
        Eigen::Matrix<double, acc_model::X_N, acc_model::X_N> q;
        q(0, 0) = q(2, 2) = q(4, 4) = q_v_;
        q(1, 1) = q(3, 3) = q(5, 5) = q_a_;
        return q;
    };
    auto yu_q = [this]() {
        Eigen::Matrix<double, ypdarmor_motion_model::X_N, ypdarmor_motion_model::X_N> q;
        double t = dt_;
        double x, y, z, yaw, r, d_zc;
        // auto gobal_state = gobal::stringanything.get_value<GobalState>("gobal_state");
        // if (gobal_state.armor_slove_state == GobalState::ArmorSloveState::TRACKING_ARMOR) {
        x = ys2qx_a_, y = ys2qy_a_, z = ys2qz_a_, yaw = ys2qyaw_a_, r = ys2qr_a_,
        d_zc = ys2qd_zc_a_;
        // } else {
        //     x = ys2qx_c_, y = ys2qy_c_, z = ys2qz_c_, yaw = ys2qyaw_c_, r = ys2qr_c_,
        //     d_zc = ys2qd_zc_c_;
        // }

        double q_x_x = pow(t, 4) / 4 * x, q_x_vx = pow(t, 3) / 2 * x, q_vx_vx = pow(t, 2) * x;
        double q_y_y = pow(t, 4) / 4 * y, q_y_vy = pow(t, 3) / 2 * y, q_vy_vy = pow(t, 2) * y;
        double q_z_z = pow(t, 4) / 4 * z, q_z_vz = pow(t, 3) / 2 * z, q_vz_vz = pow(t, 2) * z;
        double q_yaw_yaw = pow(t, 4) / 4 * yaw, q_yaw_vyaw = pow(t, 3) / 2 * yaw,
               q_vyaw_vyaw = pow(t, 2) * yaw;
        double q_r = pow(t, 4) / 4 * r;
        double q_d_zc = pow(t, 4) / 4 * d_zc;
        // clang-format off
        //      xc      v_xc    yc      v_yc    zc      v_zc    yaw         v_yaw       r       d_za
        q <<    q_x_x,  q_x_vx, 0,      0,      0,      0,      0,          0,          0,      0,
                q_x_vx, q_vx_vx,0,      0,      0,      0,      0,          0,          0,      0,
                0,      0,      q_y_y,  q_y_vy, 0,      0,      0,          0,          0,      0,
                0,      0,      q_y_vy, q_vy_vy,0,      0,      0,          0,          0,      0,
                0,      0,      0,      0,      q_z_z,  q_z_vz, 0,          0,          0,      0,
                0,      0,      0,      0,      q_z_vz, q_vz_vz,0,          0,          0,      0,
                0,      0,      0,      0,      0,      0,      q_yaw_yaw,  q_yaw_vyaw, 0,      0,
                0,      0,      0,      0,      0,      0,      q_yaw_vyaw, q_vyaw_vyaw,0,      0,
                0,      0,      0,      0,      0,      0,      0,          0,          q_r,    0,
                0,      0,      0,      0,      0,      0,      0,          0,          0,      q_d_zc;

        // clang-format on
        return q;
    };

    auto oyu_q = [this]() {
        Eigen::Matrix<double, oneypdarmor_motion_model::X_N, oneypdarmor_motion_model::X_N> q;
        double t = dt_, x = oys2qx_, y = oys2qy_, z = oys2qz_, yaw = oys2qyaw_;
        double q_x_x = pow(t, 4) / 4 * x, q_x_vx = pow(t, 3) / 2 * x, q_vx_vx = pow(t, 2) * x;
        double q_y_y = pow(t, 4) / 4 * y, q_y_vy = pow(t, 3) / 2 * y, q_vy_vy = pow(t, 2) * y;
        double q_z_z = pow(t, 4) / 4 * z, q_z_vz = pow(t, 3) / 2 * z, q_vz_vz = pow(t, 2) * z;
        double q_yaw_yaw = pow(t, 4) / 4 * yaw, q_yaw_vyaw = pow(t, 3) / 2 * yaw,
               q_vyaw_vyaw = pow(t, 2) * yaw;

        // clang-format off
        //    xc      v_xc    yc      v_yc    zc      v_zc    yaw         v_yaw       
        q <<    q_x_x,  q_x_vx, 0,      0,      0,      0,      0,          0,         
                q_x_vx, q_vx_vx,0,      0,      0,      0,      0,          0,         
                0,      0,      q_y_y,  q_y_vy, 0,      0,      0,          0,          
                0,      0,      q_y_vy, q_vy_vy,0,      0,      0,          0,          
                0,      0,      0,      0,      q_z_z,  q_z_vz, 0,          0,          
                0,      0,      0,      0,      q_z_vz, q_vz_vz,0,          0,          
                0,      0,      0,      0,      0,      0,      q_yaw_yaw,  q_yaw_vyaw, 
                0,      0,      0,      0,      0,      0,      q_yaw_vyaw, q_vyaw_vyaw;

        // clang-format on
        return q;
    };

    // EKF 观测噪声协方差 R（基于测量值调整）
    auto acc_r = [this](const Eigen::Matrix<double, acc_model::Z_N, 1>& z) {
        Eigen::Matrix<double, acc_model::Z_N, acc_model::Z_N> r;
        r *= r_v_;
        return r;
    };
    auto yu_r = [this](const Eigen::Matrix<double, ypdarmor_motion_model::Z_N, 1>& z) {
        Eigen::Matrix<double, ypdarmor_motion_model::Z_N, ypdarmor_motion_model::Z_N> r;
        Eigen::Vector3d dir_odom(std::cos(z[3]), std::sin(z[3]), 0.0);

        Eigen::Vector3d dir_gimbal = this->R_gimbal2odom_.transpose() * dir_odom;

        double camera_yaw = std::atan2(dir_gimbal.y(), dir_gimbal.x()) * 180.0 / M_PI;
        double ry, rp, rd_front, rd_side, ryaw_front, ryaw_side;
        // auto gobal_state = gobal::stringanything.get_value<GobalState>("gobal_state");
        // if (gobal_state.armor_slove_state == GobalState::ArmorSloveState::TRACKING_ARMOR) {
        ry = yr_y_a_, rp = yr_p_a_, rd_front = yr_d_front_a_, rd_side = yr_d_side_a_,
        ryaw_front = yr_yaw_front_a_, ryaw_side = yr_yaw_side_a_;
        // } else {
        //     ry = yr_y_c_, rp = yr_p_c_, rd_front = yr_d_front_c_, rd_side = yr_d_side_c_,
        //     ryaw_front = yr_yaw_front_c_, ryaw_side = yr_yaw_side_c_;
        // }
        // clang-format off
        r <<pow(ry * M_PI / 180.0, 2), 0, 0, 0,
                0, pow(rp * M_PI / 180.0, 2) , 0, 0,
                0, 0, utils::getNoiseVarFromCameraYaw(camera_yaw ,rd_front , rd_side) * std::abs(z[2]) *std::abs(z[2]), 0,//pnp得到的distance的误差与distance的平方正相关
                0, 0, 0, utils::getNoiseVarFromCameraYaw(camera_yaw ,ryaw_front , ryaw_side);//相机系下yaw正对误差大
        // clang-format on
        return r;
    };

    auto oyu_r = [this](const Eigen::Matrix<double, oneypdarmor_motion_model::Z_N, 1>& z) {
        Eigen::Matrix<double, oneypdarmor_motion_model::Z_N, oneypdarmor_motion_model::Z_N> r;
        Eigen::Vector3d dir_odom(std::cos(z[3]), std::sin(z[3]), 0.0);

        Eigen::Vector3d dir_gimbal = this->R_gimbal2odom_.transpose() * dir_odom;

        double camera_yaw = std::atan2(dir_gimbal.y(), dir_gimbal.x()) * 180.0 / M_PI;
        // clang-format off
            r <<pow(oyr_y_ * M_PI / 180.0, 2), 0, 0, 0,
                0, pow(oyr_p_ * M_PI / 180.0, 2) , 0, 0,
                0, 0, utils::getNoiseVarFromCameraYaw(camera_yaw ,oyr_d_front_ , oyr_d_side_) * std::abs(z[2]) *std::abs(z[2]), 0,
                0, 0, 0, utils::getNoiseVarFromCameraYaw(camera_yaw ,oyr_yaw_front_ , oyr_yaw_side_);
        // clang-format on
        return r;
    };

    // 初始协方差
    Eigen::DiagonalMatrix<double, acc_model::X_N> accp0;
    accp0.setIdentity();

    Eigen::DiagonalMatrix<double, ypdarmor_motion_model::X_N> yp0;
    yp0.setIdentity();

    Eigen::DiagonalMatrix<double, oneypdarmor_motion_model::X_N> oyp0;
    oyp0.setIdentity();
    auto tracker_predict_func =
        [&](const std::unique_ptr<ypdarmor_motion_model::RobotStateEKF>& ekf,
            const std::unique_ptr<ypdarmor_motion_model::RobotStateESEKF>& esekf) {
            Eigen::VectorXd x1 = ekf->predict();
            Eigen::VectorXd x2 = esekf->predict();
            // auto gobal_state = gobal::stringanything.get_value<GobalState>("gobal_state");
            // if (gobal_state.armor_slove_state == GobalState::ArmorSloveState::TRACKING_ARMOR) {
            if (fusion_esekf_ekf) {
                const double eps = 1e-6;
                Eigen::MatrixXd P1 = ekf->getPriorCovariance();
                Eigen::MatrixXd P2 = esekf->getPriorCovariance();
                Eigen::MatrixXd info1 = P1.inverse();
                Eigen::MatrixXd info2 = P2.inverse();
                double norm1 = ekf->getResidualNorm();
                double norm2 = esekf->getResidualNorm();
                double w1 = 1.0 / (norm1 + eps);
                double w2 = 1.0 / (norm2 + eps);
                double sum_w = w1 + w2;
                w1 /= sum_w;
                w2 /= sum_w;
                Eigen::MatrixXd info_fused = w1 * info1 + w2 * info2;
                Eigen::MatrixXd P_fused = info_fused.inverse();
                Eigen::VectorXd x_fused = P_fused * (w1 * info1 * x1 + w2 * info2 * x2);
                return x_fused;
            } else {
                return use_esekf ? x2 : x1;
            }
            // } else {
            //     return x1;
            // }
        };
    auto tracker_update_func =
        [&](const std::unique_ptr<ypdarmor_motion_model::RobotStateEKF>& ekf,
            const std::unique_ptr<ypdarmor_motion_model::RobotStateESEKF>& esekf,
            const Eigen::VectorXd& measurement) {
            if (fusion_esekf_ekf) {
                ekf->update(measurement);
                esekf->update(measurement);
            } else {
                use_esekf ? esekf->update(measurement) : ekf->update(measurement);
            }
        };
    auto tracker_setstate_func =
        [&](const std::unique_ptr<ypdarmor_motion_model::RobotStateEKF>& ekf,
            const std::unique_ptr<ypdarmor_motion_model::RobotStateESEKF>& esekf,
            const Eigen::VectorXd& target_state) {
            if (fusion_esekf_ekf) {
                ekf->setState(target_state);
                esekf->setState(target_state);
            } else {
                use_esekf ? esekf->setState(target_state) : ekf->setState(target_state);
            }
        };
    tracker_->predict_func_ = tracker_predict_func;
    tracker_->update_func_ = tracker_update_func;
    tracker_->setstate_func_ = tracker_setstate_func;
    tracker_->ekf_ypd_ =
        std::make_unique<ypdarmor_motion_model::RobotStateEKF>(yf, yh, yu_q, yu_r, yp0);
    tracker_->ekf_ypd_->setAngleDims({ 0, 3 });
    tracker_->ekf_ypd_->setIterationNum(iteration_num_);
    tracker_->esekf_ypd_ =
        std::make_unique<ypdarmor_motion_model::RobotStateESEKF>(yf, yh, yu_q, yu_r, yp0);
    tracker_->esekf_ypd_->setAngleDims({ 0, 3 });
    tracker_->esekf_ypd_->setIterationNum(iteration_num_);
    tracker_->esekf_ypd_->setInjectFunc(
        [](const Eigen::Matrix<double, ypdarmor_motion_model::X_N, 1>& delta,
           Eigen::Matrix<double, ypdarmor_motion_model::X_N, 1>& nominal) {
            for (int i = 0; i < ypdarmor_motion_model::X_N; i++) {
                if (i == 6)
                    continue;
                nominal[i] += delta[i];
            }
            nominal[6] = angles::normalize_angle(nominal[6] + delta[6]);
        }
    );
    tracker_->acc_ekf_ =
        std::make_unique<acc_model::VelocityAccelEKF>(acc_f, acc_h, acc_q, acc_r, accp0);

    for (auto& o_tracker: one_trackers_) {
        o_tracker->ekf_ypd_ =
            std::make_unique<oneypdarmor_motion_model::RobotStateEKF>(oyf, oyh, oyu_q, oyu_r, oyp0);
        o_tracker->ekf_ypd_->setAngleDims({ 0, 3 });
        o_tracker->ekf_ypd_->setIterationNum(iteration_num_);
        o_tracker->acc_ekf_ =
            std::make_unique<acc_model::VelocityAccelEKF>(acc_f, acc_h, acc_q, acc_r, accp0);
    }
    attack_state_ = ATTACKWHOLECAR;
}

void TrackerManager::updateTracker(
    armor::Target& target_,
    armor::Armors armors_,
    std::chrono::steady_clock::time_point time,
    const Eigen::Vector3d& v
) {
    target_.timestamp = time;
    if (tracker_->tracker_state == Tracker::LOST) {
        tracker_->init(armors_);

        target_.tracking = false;

    } else {
        dt_ = std::chrono::duration<double>(time - last_time_).count();
        tracker_->lost_thres_ = std::abs(static_cast<int>(lost_time_thres_ / dt_));
        double vx = v.x(), vy = v.y(), vz = v.z();

        if (tracker_->tracked_id_ == armor::ArmorNumber::OUTPOST) {
            tracker_->ekf_ypd_->setPredictFunc(ypdarmor_motion_model::Predict {
                dt_,
                ypdarmor_motion_model::MotionModel::CONSTANT_ROTATION,
                vx,
                vy,
                vz });
            tracker_->esekf_ypd_->setPredictFunc(ypdarmor_motion_model::Predict {
                dt_,
                ypdarmor_motion_model::MotionModel::CONSTANT_ROTATION,
                vx,
                vy,
                vz });
        } else {
            tracker_->ekf_ypd_->setPredictFunc(ypdarmor_motion_model::Predict {
                dt_,
                ypdarmor_motion_model::MotionModel::CONSTANT_VEL_ROT,
                vx,
                vy,
                vz });
            tracker_->esekf_ypd_->setPredictFunc(ypdarmor_motion_model::Predict {
                dt_,
                ypdarmor_motion_model::MotionModel::CONSTANT_VEL_ROT,
                vx,
                vy,
                vz });
        }
        tracker_->acc_ekf_->setPredictFunc(acc_model::Predict { dt_ });
        tracker_->update(armors_);

        if (tracker_->tracker_state == Tracker::DETECTING) {
            target_.tracking = false;
            target_.armors_num = static_cast<int>(tracker_->tracked_armors_num_);
        } else if (tracker_->tracker_state == Tracker::TRACKING || tracker_->tracker_state == Tracker::TEMP_LOST)
        {
            target_.tracking = true;
            const auto& state = tracker_->target_state_;
            const auto& acc_state = tracker_->acc_state_;
            target_.id = tracker_->tracked_id_;
            target_.armors_num = static_cast<int>(tracker_->tracked_armors_num_);
            target_.position_.x() = state(0);
            target_.velocity_.x() = state(1);
            target_.acceleration_.x() = acc_state(1);
            target_.position_.y() = state(2);
            target_.velocity_.y() = state(3);
            target_.acceleration_.y() = acc_state(3);
            target_.position_.z() = state(4);
            target_.velocity_.z() = state(5);
            target_.acceleration_.z() = acc_state(5);
            target_.yaw = state(6);
            target_.v_yaw = state(7);
            target_.radius_1 = state(8);
            target_.radius_2 = tracker_->another_r_;
            target_.d_zc = state(9);
            target_.d_za = tracker_->d_za_;
            target_.type = tracker_->type_;

            //target_.acceleration_ = tf::Position(0, 0, 0);
        }
    }
}
void TrackerManager::updateOneTrackers(
    std::vector<armor::OneTarget>& one_targets_,
    armor::Armors armors_,
    std::chrono::steady_clock::time_point time,
    const Eigen::Vector3d& v
) {
    std::vector<bool> armor_assigned(armors_.armors.size(), false);

    for (auto& otracker: one_trackers_) {
        armor::OneTarget target;

        if (otracker->tracker_state == Tracker::LOST) {
            int best_i = -1;
            double min_dist_center = std::numeric_limits<double>::max();
            for (size_t i = 0; i < armors_.armors.size(); ++i) {
                if (!armor_assigned[i]) {
                    double dist_center = armors_.armors[i].distance_to_image_center;
                    if (dist_center < min_dist_center) {
                        min_dist_center = dist_center;
                        best_i = static_cast<int>(i);
                    }
                }
            }

            if (best_i >= 0) {
                otracker->init({ armors_.armors[best_i] });
                armor_assigned[best_i] = true;
            }

            target.tracking = false;
            one_targets_.push_back(target);
            continue;
        }

        otracker->lost_thres_ = std::abs(static_cast<int>(one_lost_time_thres_ / dt_));
        Eigen::VectorXd ekf_prediction;
        double vx = v.x(), vy = v.y(), vz = v.z();
        if (otracker->tracked_id_ == armor::ArmorNumber::OUTPOST) {
            otracker->ekf_ypd_->setPredictFunc(oneypdarmor_motion_model::Predict {
                dt_,
                oneypdarmor_motion_model::MotionModel::CONSTANT_ROTATION,
                vx,
                vy,
                vz });
        } else {
            otracker->ekf_ypd_->setPredictFunc(oneypdarmor_motion_model::Predict {
                dt_,
                oneypdarmor_motion_model::MotionModel::CONSTANT_VEL_ROT,
                vx,
                vy,
                vz });
        }
        otracker->acc_ekf_->setPredictFunc(acc_model::Predict { dt_ });
        ekf_prediction = otracker->ekf_ypd_->predict();

        Eigen::Vector3d predicted_position = otracker->getArmorPositionFromState(ekf_prediction);

        int best_i = -1;
        double min_dist = std::numeric_limits<double>::max();

        for (size_t i = 0; i < armors_.armors.size(); ++i) {
            if (armor_assigned[i])
                continue;

            if (!isSameTarget(armors_.armors[i].number, otracker->tracked_id_)) {
                continue;
            }

            const auto& armor = armors_.armors[i];
            Eigen::Vector3d obs_pos(
                armor.target_pos.x(),
                armor.target_pos.y(),
                armor.target_pos.z()
            );
            double obs_yaw = otracker->orientationToYaw(armor.target_ori);

            double pos_dist = (obs_pos - predicted_position).norm();
            double yaw_diff = std::abs(normalizeAngle(obs_yaw - otracker->target_state_(6)));
            double z_diff = std::abs(obs_pos.z() - predicted_position.z());
            if (pos_dist > otracker->max_match_distance_)
                continue;

            if (yaw_diff > otracker->max_match_yaw_diff_)
                continue;
            if (z_diff > otracker->max_match_z_diff_)
                continue;
            if (pos_dist < min_dist) {
                min_dist = pos_dist;
                best_i = static_cast<int>(i);
            }
        }

        if (best_i >= 0) {
            otracker->update({ armors_.armors[best_i] });
            armor_assigned[best_i] = true;
        } else {
            armor::Armor empty_armor;
            otracker->update(empty_armor);
        }
        if (otracker->tracker_state == Tracker::TRACKING
            || otracker->tracker_state == Tracker::TEMP_LOST) {
            const auto& state = otracker->target_state_;
            const auto& acc_state = otracker->acc_state_;
            target.tracking = true;
            target.id = otracker->tracked_id_;
            target.position_.x() = state(0);
            target.velocity_.x() = state(1);
            target.acceleration_.x() = acc_state(1);
            target.position_.y() = state(2);
            target.velocity_.y() = state(3);
            target.acceleration_.y() = acc_state(3);
            target.position_.z() = state(4);
            target.velocity_.z() = state(5);
            target.acceleration_.z() = acc_state(5);
            target.yaw = state(6);
            target.v_yaw = state(7);
            target.type = otracker->type_;

            //target.acceleration_ = tf::Position(0, 0, 0);

            target.distance_to_image_center = otracker->distance_to_image_center_;

        } else {
            target.tracking = false;
        }
        target.timestamp = time;
        target.is_omni = false;
        one_targets_.push_back(target);
    }
}
void TrackerManager::updateAttackState(const double& v_yaw_abs) {
    switch (attack_state_) {
        case ATTACKONE:
            if (v_yaw_abs > v_yaw_to_one_thres_high_ || tracker_->tracker_state == Tracker::LOST) {
                attack_state_ = ATTACKWHOLECAR;
            }
            break;

        case ATTACKWHOLECAR:
            if (v_yaw_abs < v_yaw_to_one_thres_low_) {
                attack_state_ = ATTACKONE;
            }
            break;

        default:
            if (v_yaw_abs > v_yaw_to_one_thres_high_) {
                attack_state_ = ATTACKWHOLECAR;
            } else {
                attack_state_ = ATTACKONE;
            }

            break;
    }
}

void TrackerManager::update(
    armor::Target& target_,
    std::vector<armor::OneTarget>& one_targets_,
    armor::Armors armors_,
    std::chrono::steady_clock::time_point time,
    const Eigen::Matrix3d& R_gimbal2odom,
    const Eigen::Vector3d& v
) {
    this->R_gimbal2odom_ = R_gimbal2odom;
    updateTracker(target_, armors_, time, v);
    updateAttackState(std::abs(target_.v_yaw));

    armor::Armors armors_empty;
    switch (attack_state_) {
        case ATTACKONE: {
            updateOneTrackers(one_targets_, armors_, time, v);
        } break;
        case ATTACKWHOLECAR: {
            std::vector<armor::OneTarget> one_targets_fake;
            updateOneTrackers(one_targets_fake, armors_empty, time, v);
        } break;
    }

    last_time_ = time;
}
