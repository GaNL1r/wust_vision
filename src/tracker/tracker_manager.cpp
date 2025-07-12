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
#include "tracker/tracker_manager.hpp"
#include "common/gobal.hpp"

TrackerManager::TrackerManager(const YAML::Node& config_) {
    track_one_num = config_["track_one_num"].as<int>(2);

    double max_match_distance = config_["max_match_distance"].as<double>(0.2);
    double max_match_yaw_diff = config_["max_match_yaw_diff"].as<double>(1.0);
    double max_match_z_diff = config_["max_match_z_diff"].as<double>(0.1);
    double jump_thresh = config_["jump_thresh"].as<double>(0.4);
    tracker_ = std::make_unique<Tracker>(
        max_match_distance,
        max_match_yaw_diff,
        max_match_z_diff,
        jump_thresh
    );

    for (int i = 0; i < track_one_num; i++) {
        auto o_tracker_ = std::make_unique<OneTracker>(
            max_match_distance,
            max_match_yaw_diff,
            max_match_z_diff,
            jump_thresh
        );
        o_tracker_->tracking_thres = config_["one_tracking_thres"].as<int>(1);
        one_trackers_.push_back(std::move(o_tracker_));
    }

    v_yaw_to_one_thres_high = config_["v_yaw_to_one_thres_high"].as<float>(1.0);
    v_yaw_to_one_thres_low = config_["v_yaw_to_one_thres_low"].as<float>(0.7);

    tracker_->tracking_thres = config_["tracking_thres"].as<int>(5);

    lost_time_thres_ = config_["lost_time_thres"].as<double>();
    one_lost_time_thres_ = config_["one_lost_time_thres"].as<double>(0.1);

    iteration_num_ = config_["ekf"]["iteration_num"].as<int>(1);
    bool use_esekf = config_["ekf"]["use_esekf"].as<bool>(false);
    bool fusion_esekf_ekf = config_["ekf"]["fusion_esekf_ekf"].as<bool>(false);
    tracker_->use_esekf = use_esekf;
    // EKF 噪声参数

    ys2qx_ = config_["ekf"]["ys2qx"].as<double>(20.0);
    ys2qy_ = config_["ekf"]["ys2qy"].as<double>(20.0);
    ys2qz_ = config_["ekf"]["ys2qz"].as<double>(20.0);
    ys2qyaw_ = config_["ekf"]["ys2qyaw"].as<double>(100.0);
    ys2qr_ = config_["ekf"]["ys2qr"].as<double>(800.0);
    ys2qd_zc_ = config_["ekf"]["ys2qd_zc"].as<double>(800.0);

    yr_y_ = config_["ekf"]["yr_y"].as<double>(0.05);
    yr_p_ = config_["ekf"]["yr_p"].as<double>(0.05);
    yr_d_front_ = config_["ekf"]["yr_d_front"].as<double>(0.05);
    yr_d_side_ = config_["ekf"]["yr_d_side"].as<double>(0.05);
    yr_yaw_front_ = config_["ekf"]["yr_yaw_front"].as<double>(0.02);
    yr_yaw_side_ = config_["ekf"]["yr_yaw_side"].as<double>(0.02);

    oys2qx_ = config_["ekf"]["oys2qx"].as<double>(20.0);
    oys2qy_ = config_["ekf"]["oys2qy"].as<double>(20.0);
    oys2qz_ = config_["ekf"]["oys2qz"].as<double>(20.0);
    oys2qyaw_ = config_["ekf"]["oys2qyaw"].as<double>(100.0);

    oyr_y_ = config_["ekf"]["oyr_y"].as<double>(0.05);
    oyr_p_ = config_["ekf"]["oyr_p"].as<double>(0.05);
    oyr_d_front_ = config_["ekf"]["oyr_d_front"].as<double>(0.05);
    oyr_d_side_ = config_["ekf"]["oyr_d_side"].as<double>(0.05);
    oyr_yaw_front_ = config_["ekf"]["oyr_yaw_front"].as<double>(0.02);
    oyr_yaw_side_ = config_["ekf"]["oyr_yaw_side"].as<double>(0.02);

    // EKF 状态预测函数

    auto yf = ypdarmor_motion_model::Predict(0.005);
    auto oyf = oneypdarmor_motion_model::Predict(0.005);

    // EKF 观测函数

    auto yh = ypdarmor_motion_model::Measure();
    auto oyh = oneypdarmor_motion_model::Measure();

    // EKF 过程噪声协方差 Q

    auto yu_q = [this]() {
        Eigen::Matrix<double, ypdarmor_motion_model::X_N, ypdarmor_motion_model::X_N> q;
        double t = dt_, x = ys2qx_, y = ys2qy_, z = ys2qz_, yaw = ys2qyaw_, r = ys2qr_,
               d_zc = ys2qd_zc_;
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

    auto yu_r = [this](const Eigen::Matrix<double, ypdarmor_motion_model::Z_N, 1>& z) {
        Eigen::Matrix<double, ypdarmor_motion_model::Z_N, ypdarmor_motion_model::Z_N> r;
        Eigen::Vector3d dir_odom(std::cos(z[3]), std::sin(z[3]), 0.0);

        Eigen::Vector3d dir_gimbal = this->R_gimbal2odom_.transpose() * dir_odom;

        double camera_yaw = std::atan2(dir_gimbal.y(), dir_gimbal.x()) * 180.0 / M_PI;

        // clang-format off
        r <<pow(yr_y_ * M_PI / 180.0, 2), 0, 0, 0,
                0, pow(yr_p_ * M_PI / 180.0, 2) , 0, 0,
                0, 0, getNoiseVarFromCameraYaw(camera_yaw ,yr_d_front_ , yr_d_side_) * std::abs(z[2]) *std::abs(z[2]), 0,//pnp得到的distance的误差与distance的平方正相关
                0, 0, 0, getNoiseVarFromCameraYaw(camera_yaw ,yr_yaw_front_ , yr_yaw_side_);//相机系下yaw正对误差大
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
                0, 0, getNoiseVarFromCameraYaw(camera_yaw ,oyr_d_front_ , oyr_d_side_) * std::abs(z[2]) *std::abs(z[2]), 0,
                0, 0, 0, getNoiseVarFromCameraYaw(camera_yaw ,oyr_yaw_front_ , oyr_yaw_side_);
        // clang-format on
        return r;
    };

    // 初始协方差
    Eigen::DiagonalMatrix<double, ypdarmor_motion_model::X_N> yp0;
    yp0.setIdentity();

    Eigen::DiagonalMatrix<double, oneypdarmor_motion_model::X_N> oyp0;
    oyp0.setIdentity();
    auto tracker_predict_func =
        [&](const std::unique_ptr<ypdarmor_motion_model::RobotStateEKF>& ekf,
            const std::unique_ptr<ypdarmor_motion_model::RobotStateESEKF>& esekf) {
            Eigen::VectorXd x1 = ekf->predict();
            Eigen::VectorXd x2 = esekf->predict();

            if (fusion_esekf_ekf) {
                Eigen::MatrixXd P1 = ekf->getPriorCovariance();
                Eigen::MatrixXd P2 = esekf->getPriorCovariance();
                Eigen::MatrixXd info1 = P1.inverse();
                Eigen::MatrixXd info2 = P2.inverse();
                Eigen::MatrixXd info_fused = info1 + info2;
                Eigen::MatrixXd P_fused = info_fused.inverse();
                Eigen::VectorXd x_fused = P_fused * (info1 * x1 + info2 * x2);

                return x_fused;
            } else {
                return use_esekf ? x2 : x1;
            }
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
    tracker_->predict_func = tracker_predict_func;
    tracker_->update_func = tracker_update_func;
    tracker_->setstate_func = tracker_setstate_func;
    tracker_->ekf_ypd =
        std::make_unique<ypdarmor_motion_model::RobotStateEKF>(yf, yh, yu_q, yu_r, yp0);
    tracker_->ekf_ypd->setAngleDims({ 0, 3 });
    tracker_->ekf_ypd->setIterationNum(iteration_num_);
    tracker_->esekf_ypd =
        std::make_unique<ypdarmor_motion_model::RobotStateESEKF>(yf, yh, yu_q, yu_r, yp0);
    tracker_->esekf_ypd->setAngleDims({ 0, 3 });
    tracker_->esekf_ypd->setIterationNum(iteration_num_);
    tracker_->esekf_ypd->setInjectFunc(
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

    for (auto& o_tracker: one_trackers_) {
        o_tracker->ekf_ypd =
            std::make_unique<oneypdarmor_motion_model::RobotStateEKF>(oyf, oyh, oyu_q, oyu_r, oyp0);
        o_tracker->ekf_ypd->setAngleDims({ 0, 3 });
        o_tracker->ekf_ypd->setIterationNum(iteration_num_);
    }

    gobal::attack_state = gobal::AttackState::ATTACKONE;
}
double
TrackerManager::getNoiseFromCameraYaw(double camera_yaw_deg, double r_front, double r_side) const {
    double yaw_rad = camera_yaw_deg * M_PI / 180.0;
    double cos2 = std::cos(yaw_rad);
    cos2 *= cos2;
    return cos2 * r_front + (1.0 - cos2) * r_side;
}
double
TrackerManager::getNoiseVarFromCameraYaw(double camera_yaw_deg, double r_front, double r_side)
    const {
    double noise_deg = getNoiseFromCameraYaw(camera_yaw_deg, r_front, r_side);
    return std::pow(noise_deg * M_PI / 180.0, 2);
}

void TrackerManager::updateTracker(
    Target& target_,
    Armors armors_,
    std::chrono::steady_clock::time_point time
) {
    target_.timestamp = time;
    if (tracker_->tracker_state == Tracker::LOST) {
        tracker_->init(armors_);
        target_.tracking = false;
    } else {
        dt_ = std::chrono::duration<double>(time - last_time_).count();
        tracker_->lost_thres = std::abs(static_cast<int>(lost_time_thres_ / dt_));

        if (tracker_->tracked_id == ArmorNumber::OUTPOST) {
            tracker_->ekf_ypd->setPredictFunc(ypdarmor_motion_model::Predict {
                dt_,
                ypdarmor_motion_model::MotionModel::CONSTANT_ROTATION });
            tracker_->esekf_ypd->setPredictFunc(ypdarmor_motion_model::Predict {
                dt_,
                ypdarmor_motion_model::MotionModel::CONSTANT_ROTATION });
        } else {
            tracker_->ekf_ypd->setPredictFunc(ypdarmor_motion_model::Predict {
                dt_,
                ypdarmor_motion_model::MotionModel::CONSTANT_VEL_ROT });
            tracker_->esekf_ypd->setPredictFunc(ypdarmor_motion_model::Predict {
                dt_,
                ypdarmor_motion_model::MotionModel::CONSTANT_VEL_ROT });
        }

        tracker_->update(armors_);

        if (tracker_->tracker_state == Tracker::DETECTING) {
            target_.tracking = false;
        } else if (tracker_->tracker_state == Tracker::TRACKING || tracker_->tracker_state == Tracker::TEMP_LOST)
        {
            target_.tracking = true;
            const auto& state = tracker_->target_state;
            target_.id = tracker_->tracked_id;
            target_.armors_num = static_cast<int>(tracker_->tracked_armors_num);
            target_.position_.x = state(0);
            target_.velocity_.x = state(1);
            target_.position_.y = state(2);
            target_.velocity_.y = state(3);
            target_.position_.z = state(4);
            target_.velocity_.z = state(5);
            target_.yaw = state(6);
            target_.v_yaw = state(7);
            target_.radius_1 = state(8);
            target_.radius_2 = tracker_->another_r;
            target_.d_zc = state(9);
            target_.d_za = tracker_->d_za;
            target_.type = tracker_->type;
        }
    }
}
void TrackerManager::updateOneTrackers(
    std::vector<OneTarget>& one_targets_,
    Armors armors_,
    std::chrono::steady_clock::time_point time
) {
    std::vector<bool> armor_assigned(armors_.armors.size(), false);

    for (auto& otracker: one_trackers_) {
        OneTarget target;

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

        otracker->lost_thres = std::abs(static_cast<int>(one_lost_time_thres_ / dt_));
        Eigen::VectorXd ekf_prediction;

        if (otracker->tracked_id == ArmorNumber::OUTPOST) {
            otracker->ekf_ypd->setPredictFunc(oneypdarmor_motion_model::Predict {
                dt_,
                oneypdarmor_motion_model::MotionModel::CONSTANT_ROTATION });
        } else {
            otracker->ekf_ypd->setPredictFunc(oneypdarmor_motion_model::Predict {
                dt_,
                oneypdarmor_motion_model::MotionModel::CONSTANT_VEL_ROT });
        }

        ekf_prediction = otracker->ekf_ypd->predict();

        Eigen::Vector3d predicted_position = otracker->getArmorPositionFromState(ekf_prediction);

        int best_i = -1;
        double min_dist = std::numeric_limits<double>::max();

        for (size_t i = 0; i < armors_.armors.size(); ++i) {
            if (armor_assigned[i])
                continue;

            if (!isSameTarget(armors_.armors[i].number, otracker->tracked_id)) {
                continue;
            }

            const auto& armor = armors_.armors[i];
            Eigen::Vector3d obs_pos(armor.target_pos.x, armor.target_pos.y, armor.target_pos.z);
            double obs_yaw = otracker->orientationToYaw(armor.target_ori);

            double pos_dist = (obs_pos - predicted_position).norm();
            double yaw_diff = std::abs(normalizeAngle(obs_yaw - otracker->target_state(6)));
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
            Armor empty_armor;
            otracker->update(empty_armor);
        }
        if (otracker->tracker_state == Tracker::TRACKING
            || otracker->tracker_state == Tracker::TEMP_LOST) {
            const auto& state = otracker->target_state;
            target.tracking = true;
            target.id = otracker->tracked_id;
            target.position_.x = state(0);
            target.velocity_.x = state(1);
            target.position_.y = state(2);
            target.velocity_.y = state(3);
            target.position_.z = state(4);
            target.velocity_.z = state(5);
            target.yaw = state(6);
            target.v_yaw = state(7);
            target.type = otracker->type;
            target.distance_to_image_center = otracker->distance_to_image_center;
        } else {
            target.tracking = false;
        }
        target.timestamp = time;
        one_targets_.push_back(target);
    }
}
void TrackerManager::updateAttackState(const double& v_yaw_abs) {
    switch (gobal::attack_state) {
        case gobal::AttackState::ATTACKONE:
            if (v_yaw_abs > v_yaw_to_one_thres_high || tracker_->tracker_state == Tracker::LOST) {
                gobal::attack_state = gobal::AttackState::ATTACKWHOLECAR;
            }
            break;

        case gobal::AttackState::ATTACKWHOLECAR:
            if (v_yaw_abs < v_yaw_to_one_thres_low) {
                gobal::attack_state = gobal::AttackState::ATTACKONE;
            }
            break;

        default:
            if (v_yaw_abs > v_yaw_to_one_thres_high) {
                gobal::attack_state = gobal::AttackState::ATTACKWHOLECAR;
            } else {
                gobal::attack_state = gobal::AttackState::ATTACKONE;
            }

            break;
    }
}

void TrackerManager::update(
    Target& target_,
    std::vector<OneTarget>& one_targets_,
    Armors armors_,
    std::chrono::steady_clock::time_point time,
    Eigen::Matrix3d R_gimbal2odom
) {
    this->R_gimbal2odom_ = R_gimbal2odom;

    updateTracker(target_, armors_, time);
    updateAttackState(std::abs(target_.v_yaw));

    Armors armors_empty;
    switch (gobal::attack_state) {
        case gobal::AttackState::ATTACKONE: {
            updateOneTrackers(one_targets_, armors_, time);
        } break;
        case gobal::AttackState::ATTACKWHOLECAR: {
            std::vector<OneTarget> one_targets_fake;
            updateOneTrackers(one_targets_fake, armors_empty, time);
        } break;
    }

    last_time_ = time;
}
