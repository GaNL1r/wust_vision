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

TrackerManager::TrackerManager(const YAML::Node& config_) {
    track_one_num = config_["track_one_num"].as<int>(2);
    // Tracker 基础参数
    double max_match_distance = config_["max_match_distance"].as<double>(0.2);
    double max_match_yaw_diff = config_["max_match_yaw_diff"].as<double>(1.0);
    double max_match_z_diff = config_["max_match_z_diff"].as<double>(0.1);
    double jump_thresh = config_["jump_thresh"].as<double>(0.4);
    use_ypd_tracker_ = config_["use_ypd_tracker"].as<bool>(false);
    if (use_ypd_tracker_) {
        ypd_tracker_ = std::make_unique<YpdTracker>(
            max_match_distance,
            max_match_yaw_diff,
            max_match_z_diff,
            jump_thresh
        );
    } else {
        tracker_ = std::make_unique<Tracker>(
            max_match_distance,
            max_match_yaw_diff,
            max_match_z_diff,
            jump_thresh
        );
    }

    // one_tracker_ = std::make_unique<OneTracker>(
    //     max_match_distance,
    //     max_match_yaw_diff,
    //     max_match_z_diff,
    //     jump_thresh
    // );


    // for (int i = 0; i < track_one_num; i++) {
    //     auto o_tracker_ = std::make_unique<OneTracker>(
    //         max_match_distance,
    //         max_match_yaw_diff,
    //         max_match_z_diff,
    //         jump_thresh
    //     );
    //o_tracker_->tracking_thres = config_["tracking_thres"].as<int>(5);
    //     one_trackers_.push_back(std::move(o_tracker_));
    // }
    // one_ca_tracker_ = std::make_unique<OneCaTracker>(
    //     max_match_distance,
    //     max_match_yaw_diff,
    //     max_match_z_diff,
    //     jump_thresh
    // );
    for (int i = 0; i < track_one_num; i++) {
      auto oy_tracker_ = std::make_unique<OneYpdTracker>(
          max_match_distance, max_match_yaw_diff,
          max_match_z_diff,jump_thresh);
          oy_tracker_->tracking_thres = config_["tracking_thres"].as<int>(5);
      one_ypd_trackers_.push_back(std::move(oy_tracker_));
    }

    v_yaw_update_thres_ = config_["v_yaw_update_thres"].as<float>(0.01);
    v_yaw_to_one_thres_ = config_["v_yaw_to_one_thres"].as<float>(0.01);

    // 跟踪判定参数
    if (use_ypd_tracker_) {
        ypd_tracker_->tracking_thres = config_["tracking_thres"].as<int>(5);
    } else {
        tracker_->tracking_thres = config_["tracking_thres"].as<int>(5);
    }

  

    // one_ypd_tracker_->tracking_thres = config_["tracking_thres"].as<int>(5);
    lost_time_thres_ = config_["lost_time_thres"].as<double>();
    one_lost_time_thres_ = config_["one_lost_time_thres"].as<double>(0.1);

    iteration_num_ = config_["ekf"]["iteration_num"].as<int>(1);
    // EKF 噪声参数
    s2qx_ = config_["ekf"]["s2qx"].as<double>(20.0);
    s2qy_ = config_["ekf"]["s2qy"].as<double>(20.0);
    s2qz_ = config_["ekf"]["s2qz"].as<double>(20.0);
    s2qyaw_ = config_["ekf"]["s2qyaw"].as<double>(100.0);
    s2qr_ = config_["ekf"]["s2qr"].as<double>(800.0);
    s2qd_zc_ = config_["ekf"]["s2qd_zc"].as<double>(800.0);

    r_x_ = config_["ekf"]["r_x"].as<double>(0.05);
    r_y_ = config_["ekf"]["r_y"].as<double>(0.05);
    r_z_ = config_["ekf"]["r_z"].as<double>(0.05);
    r_yaw_ = config_["ekf"]["r_yaw"].as<double>(0.02);

    ys2qx_ = config_["ekf"]["ys2qx"].as<double>(20.0);
    ys2qy_ = config_["ekf"]["ys2qy"].as<double>(20.0);
    ys2qz_ = config_["ekf"]["ys2qz"].as<double>(20.0);
    ys2qyaw_ = config_["ekf"]["ys2qyaw"].as<double>(100.0);
    ys2qr_ = config_["ekf"]["ys2qr"].as<double>(800.0);
    ys2qd_zc_ = config_["ekf"]["ys2qd_zc"].as<double>(800.0);

    yr_y_ = config_["ekf"]["yr_y"].as<double>(0.05);
    yr_p_ = config_["ekf"]["yr_p"].as<double>(0.05);
    yr_d_ = config_["ekf"]["yr_d"].as<double>(0.05);
    yr_yaw_ = config_["ekf"]["yr_yaw"].as<double>(0.02);

    os2qx_ = config_["ekf"]["os2qx"].as<double>(20.0);
    os2qy_ = config_["ekf"]["os2qy"].as<double>(20.0);
    os2qz_ = config_["ekf"]["os2qz"].as<double>(20.0);
    os2qyaw_ = config_["ekf"]["os2qyaw"].as<double>(100.0);

    or_x_ = config_["ekf"]["or_x"].as<double>(0.05);
    or_y_ = config_["ekf"]["or_y"].as<double>(0.05);
    or_z_ = config_["ekf"]["or_z"].as<double>(0.05);
    or_yaw_ = config_["ekf"]["or_yaw"].as<double>(0.02);

    oys2qx_ = config_["ekf"]["oys2qx"].as<double>(20.0);
    oys2qy_ = config_["ekf"]["oys2qy"].as<double>(20.0);
    oys2qz_ = config_["ekf"]["oys2qz"].as<double>(20.0);
    oys2qyaw_ = config_["ekf"]["oys2qyaw"].as<double>(100.0);

    oyr_y_ = config_["ekf"]["oyr_y"].as<double>(0.05);
    oyr_p_ = config_["ekf"]["oyr_p"].as<double>(0.05);
    oyr_d_ = config_["ekf"]["oyr_d"].as<double>(0.05);
    oyr_yaw_ = config_["ekf"]["oyr_yaw"].as<double>(0.02);

    ocas2qx_ = config_["ekf"]["ocas2qx"].as<double>(20.0);
    ocas2qy_ = config_["ekf"]["ocas2qy"].as<double>(20.0);
    ocas2qz_ = config_["ekf"]["ocas2qz"].as<double>(20.0);
    ocas2qyaw_ = config_["ekf"]["ocas2qyaw"].as<double>(100.0);

    ocar_x_ = config_["ekf"]["ocar_x"].as<double>(0.05);
    ocar_y_ = config_["ekf"]["ocar_y"].as<double>(0.05);
    ocar_z_ = config_["ekf"]["ocar_z"].as<double>(0.05);
    ocar_yaw_ = config_["ekf"]["ocar_yaw"].as<double>(0.02);

    // EKF 状态预测函数
    auto f = armor_motion_model::Predict(0.005); // dt 固定为 5ms
    auto yf = ypdarmor_motion_model::Predict(0.005);
    auto of = onearmor_motion_model::Predict(0.005);
    auto oyf = oneypdarmor_motion_model::Predict(0.005);
    auto ocaf = onecaarmor_motion_model::Predict(0.005);

    // EKF 观测函数
    auto h = armor_motion_model::Measure();
    auto yh = ypdarmor_motion_model::Measure();
    auto oh = onearmor_motion_model::Measure();
    auto oyh = oneypdarmor_motion_model::Measure();
    auto ocah = onecaarmor_motion_model::Measure();
    // EKF 过程噪声协方差 Q
    auto u_q = [this]() {
        Eigen::Matrix<double, armor_motion_model::X_N, armor_motion_model::X_N> q;
        double t = dt_, x = s2qx_, y = s2qy_, z = s2qz_, yaw = s2qyaw_, r = s2qr_, d_zc = s2qd_zc_;
        double q_x_x = pow(t, 4) / 4 * x, q_x_vx = pow(t, 3) / 2 * x, q_vx_vx = pow(t, 2) * x;
        double q_y_y = pow(t, 4) / 4 * y, q_y_vy = pow(t, 3) / 2 * y, q_vy_vy = pow(t, 2) * y;
        double q_z_z = pow(t, 4) / 4 * z, q_z_vz = pow(t, 3) / 2 * z, q_vz_vz = pow(t, 2) * z;
        double q_yaw_yaw = pow(t, 4) / 4 * yaw, q_yaw_vyaw = pow(t, 3) / 2 * yaw,
               q_vyaw_vyaw = pow(t, 2) * yaw;
        double q_r = pow(t, 4) / 4 * r;
        double q_d_zc = pow(t, 4) / 4 * d_zc;
        // clang-format off
      //    xc      v_xc    yc      v_yc    zc      v_zc    yaw         v_yaw       r       d_za
      q <<  q_x_x,  q_x_vx, 0,      0,      0,      0,      0,          0,          0,      0,
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
      //    xc      v_xc    yc      v_yc    zc      v_zc    yaw         v_yaw       r       d_za
      q <<  q_x_x,  q_x_vx, 0,      0,      0,      0,      0,          0,          0,      0,
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
    auto ou_q = [this]() {
        Eigen::Matrix<double, onearmor_motion_model::X_N, onearmor_motion_model::X_N> q;
        double t = dt_, x = os2qx_, y = os2qy_, z = os2qz_, yaw = os2qyaw_;
        double q_x_x = pow(t, 4) / 4 * x, q_x_vx = pow(t, 3) / 2 * x, q_vx_vx = pow(t, 2) * x;
        double q_y_y = pow(t, 4) / 4 * y, q_y_vy = pow(t, 3) / 2 * y, q_vy_vy = pow(t, 2) * y;
        double q_z_z = pow(t, 4) / 4 * z, q_z_vz = pow(t, 3) / 2 * z, q_vz_vz = pow(t, 2) * z;
        double q_yaw_yaw = pow(t, 4) / 4 * yaw, q_yaw_vyaw = pow(t, 3) / 2 * yaw,
               q_vyaw_vyaw = pow(t, 2) * yaw;

        // clang-format off
      //    xc      v_xc    yc      v_yc    zc      v_zc    yaw         v_yaw       
      q <<  q_x_x,  q_x_vx, 0,      0,      0,      0,      0,          0,         
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
      q <<  q_x_x,  q_x_vx, 0,      0,      0,      0,      0,          0,         
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
    auto ocau_q = [this]() {
        using Mat =
            Eigen::Matrix<double, onecaarmor_motion_model::X_N, onecaarmor_motion_model::X_N>;
        Mat q = Mat::Zero();
        double t = dt_;

        double t2 = t * t;
        double t3 = t2 * t;
        double t4 = t3 * t;

        // 白噪声强度
        double qax = ocas2qx_;
        double qay = ocas2qy_;
        double qvz = ocas2qz_;
        double qvyaw = ocas2qyaw_;

        // clang-format off
        q <<
        // x,   vx,     ax,     y,     vy,     ay,     z,     vz,     yaw,   vyaw
         t4/4*qax, t3/2*qax, t2/2*qax, 0,     0,     0,      0,     0,     0,     0,       // x
         t3/2*qax, t2*qax,   t*qax,    0,     0,     0,      0,     0,     0,     0,       // vx
         t2/2*qax, t*qax,    1.0*qax,  0,     0,     0,      0,     0,     0,     0,       // ax
    
         0,       0,        0,        t4/4*qay, t3/2*qay, t2/2*qay, 0,     0,     0,     0,  // y
         0,       0,        0,        t3/2*qay, t2*qay,   t*qay,    0,     0,     0,     0,  // vy
         0,       0,        0,        t2/2*qay, t*qay,    1.0*qay,  0,     0,     0,     0,  // ay
    
         0,       0,        0,        0,       0,        0,        t3/3*qvz, t2/2*qvz, 0,     0,  // z
         0,       0,        0,        0,       0,        0,        t2/2*qvz, t*qvz,    0,     0,  // vz
    
         0,       0,        0,        0,       0,        0,        0,     0,     t3/3*qvyaw, t2/2*qvyaw, // yaw
         0,       0,        0,        0,       0,        0,        0,     0,     t2/2*qvyaw, t*qvyaw;    // vyaw
        // clang-format on

        return q;
    };

    // EKF 观测噪声协方差 R（基于测量值调整）
    auto u_r = [this](const Eigen::Matrix<double, armor_motion_model::Z_N, 1>& z) {
        Eigen::Matrix<double, armor_motion_model::Z_N, armor_motion_model::Z_N> r;
        // clang-format off
      r << r_x_ * std::abs(z[0]), 0, 0, 0,
           0, r_y_ * std::abs(z[1]), 0, 0,
           0, 0, r_z_ * std::abs(z[2]), 0,
           0, 0, 0, r_yaw_;
        // clang-format on
        return r;
    };
    // EKF 观测噪声协方差 R（基于测量值调整）
    auto yu_r = [this](const Eigen::Matrix<double, ypdarmor_motion_model::Z_N, 1>& z) {
        Eigen::Matrix<double, ypdarmor_motion_model::Z_N, ypdarmor_motion_model::Z_N> r;
        // clang-format off
      r << yr_y_      * std::abs(z[0]), 0, 0, 0,
                0, yr_p_ * std::abs(z[1]), 0, 0,
                0, 0, yr_d_ * std::abs(z[2]), 0,
                0, 0, 0, yr_yaw_;
        // clang-format on
        return r;
    };

    auto ou_r = [this](const Eigen::Matrix<double, onearmor_motion_model::Z_N, 1>& z) {
        Eigen::Matrix<double, onearmor_motion_model::Z_N, onearmor_motion_model::Z_N> r;
        // clang-format off
      r << or_x_ * std::abs(z[0]), 0, 0, 0,
           0, or_y_ * std::abs(z[1]), 0, 0,
           0, 0, or_z_ * std::abs(z[2]), 0,
           0, 0, 0, or_yaw_;
        // clang-format on
        return r;
    };
    auto oyu_r = [this](const Eigen::Matrix<double, oneypdarmor_motion_model::Z_N, 1>& z) {
        Eigen::Matrix<double, oneypdarmor_motion_model::Z_N, oneypdarmor_motion_model::Z_N> r;
        // clang-format off
       r << oyr_y_* std::abs(z[0]), 0, 0, 0,
                    0, oyr_p_ * std::abs(z[1]), 0, 0,
                    0, 0, oyr_d_ * std::abs(z[2]), 0,
                    0, 0, 0, oyr_yaw_;
        // clang-format on
        return r;
    };
    auto ocau_r = [this](const Eigen::Matrix<double, onecaarmor_motion_model::Z_N, 1>& z) {
        Eigen::Matrix<double, onecaarmor_motion_model::Z_N, onecaarmor_motion_model::Z_N> r;
        // clang-format off
      r << or_x_ * std::abs(z[0]), 0, 0, 0,
           0, or_y_ * std::abs(z[1]), 0, 0,
           0, 0, or_z_ * std::abs(z[2]), 0,
           0, 0, 0, or_yaw_;
        // clang-format on
        return r;
    };
    // 初始协方差
    Eigen::DiagonalMatrix<double, armor_motion_model::X_N> p0;
    p0.setIdentity();
    Eigen::DiagonalMatrix<double, ypdarmor_motion_model::X_N> yp0;
    yp0.setIdentity();
    Eigen::DiagonalMatrix<double, onearmor_motion_model::X_N> op0;
    op0.setIdentity();
    Eigen::DiagonalMatrix<double, oneypdarmor_motion_model::X_N> oyp0;
    oyp0.setIdentity();
    Eigen::DiagonalMatrix<double, onecaarmor_motion_model::X_N> ocap0;
    ocap0.setIdentity();

    // 初始化 EKF 滤波器
    if (use_ypd_tracker_) {
        ypd_tracker_->ekf =
            std::make_unique<ypdarmor_motion_model::RobotStateEKF>(yf, yh, yu_q, yu_r, yp0);
        ypd_tracker_->ekf->setAngleDims({ 0, 3 });
        ypd_tracker_->ekf->setIterationNum(iteration_num_);
    } else {
        tracker_->ekf = std::make_unique<armor_motion_model::RobotStateEKF>(f, h, u_q, u_r, p0);
        tracker_->ekf->setAngleDims({ 3 });
        tracker_->ekf->setIterationNum(iteration_num_);
    }

    // one_tracker_->ekf =
    //     std::make_unique<onearmor_motion_model::RobotStateEKF>(of, oh, ou_q, ou_r, op0);
    // one_tracker_->ekf->setAngleDims({3});
    // one_ypd_tracker_->ekf =
    //     std::make_unique<oneypdarmor_motion_model::RobotStateEKF>(oyf, oyh,
    //     oyu_q, oyu_r, oyp0);
    //
    // one_ypd_tracker_->ekf->setAngleDims({0});
    // for (auto& o_tracker: one_trackers_) {
    //     o_tracker->ekf =
    //         std::make_unique<onearmor_motion_model::RobotStateEKF>(of, oh, ou_q, ou_r, op0);
    //     o_tracker->ekf->setAngleDims({ 3 });
    //     o_tracker->ekf->setIterationNum(iteration_num_);
    // }
    // one_ca_tracker_->ekf =
    //     std::make_unique<onecaarmor_motion_model::RobotStateEKF>(ocaf, ocah, ocau_q, ocau_r, ocap0);
    // one_ca_tracker_->ekf->setAngleDims({ 3 });
    for (auto &oy_tracker : one_ypd_trackers_) {
      oy_tracker->ekf =
      std::make_unique<oneypdarmor_motion_model::RobotStateEKF>(
          oyf, oyh, oyu_q, oyu_r, oyp0);
          oy_tracker->ekf->setAngleDims({ 0, 3 });
          oy_tracker->ekf->setIterationNum(iteration_num_);
    }
}

void TrackerManager::update(
    Target& target_,
    std::vector<OneTarget>& one_targets_,
    Armors armors_,
    std::chrono::steady_clock::time_point time
) {
    static int init_count_ = 0;
    if (use_ypd_tracker_) {
        if (ypd_tracker_->tracker_state == Tracker::LOST || init_count_ == 500) {
            ypd_tracker_->init(armors_);
            target_.tracking = false;
            ++init_count_;
        } else {
            dt_ = std::chrono::duration<double>(time - last_time_).count();
            ypd_tracker_->lost_thres = std::abs(static_cast<int>(lost_time_thres_ / dt_));
            if (ypd_tracker_->tracked_id == ArmorNumber::OUTPOST) {
                ypd_tracker_->ekf->setPredictFunc(ypdarmor_motion_model::Predict {
                    dt_,
                    ypdarmor_motion_model::MotionModel::CONSTANT_ROTATION });
            } else {
                ypd_tracker_->ekf->setPredictFunc(ypdarmor_motion_model::Predict {
                    dt_,
                    ypdarmor_motion_model::MotionModel::CONSTANT_VEL_ROT });
            }
            ypd_tracker_->update(armors_);

            ++init_count_;

            if (ypd_tracker_->tracker_state == Tracker::DETECTING) {
                target_.tracking = false;
            } else if (ypd_tracker_->tracker_state == Tracker::TRACKING || ypd_tracker_->tracker_state == Tracker::TEMP_LOST)
            {
                target_.tracking = true;

                const auto& state = ypd_tracker_->target_state;
                target_.id = ypd_tracker_->tracked_id;
                target_.armors_num = static_cast<int>(ypd_tracker_->tracked_armors_num);

                target_.position_.x = state(0);
                target_.velocity_.x = state(1);
                target_.position_.y = state(2);
                target_.velocity_.y = state(3);
                target_.position_.z = state(4);
                target_.velocity_.z = state(5);
                target_.yaw = state(6);
                target_.v_yaw = state(7);
                target_.radius_1 = state(8);
                target_.radius_2 = ypd_tracker_->another_r;
                target_.d_zc = state(9);
                target_.d_za = ypd_tracker_->d_za;
                target_.type = ypd_tracker_->type;
            }
        }
    } else {
        if (tracker_->tracker_state == Tracker::LOST || init_count_ == 500) {
            tracker_->init(armors_);
            target_.tracking = false;
            ++init_count_;
        } else {
            dt_ = std::chrono::duration<double>(time - last_time_).count();
            tracker_->lost_thres = std::abs(static_cast<int>(lost_time_thres_ / dt_));
            if (tracker_->tracked_id == ArmorNumber::OUTPOST) {
                tracker_->ekf->setPredictFunc(armor_motion_model::Predict {
                    dt_,
                    armor_motion_model::MotionModel::CONSTANT_ROTATION });
            } else {
                tracker_->ekf->setPredictFunc(armor_motion_model::Predict {
                    dt_,
                    armor_motion_model::MotionModel::CONSTANT_VEL_ROT });
            }
            tracker_->update(armors_);

            ++init_count_;

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

    // OneTarget one_target;
    // if (one_ca_tracker_->tracker_state == Tracker::LOST ) {
    //   one_ca_tracker_->init(armors_);
    //   one_target.tracking = false;

    // } else {
    //   dt_ = std::chrono::duration<double>(time - last_time_).count();
    //   one_ca_tracker_->lost_thres =
    //       std::abs(static_cast<int>(one_lost_time_thres_ / dt_));

    //     one_ca_tracker_->ekf->setPredictFunc(onecaarmor_motion_model::Predict{
    //         dt_, onecaarmor_motion_model::MotionModel::CONSTANT_ACCEL_ROT});

    //   one_ca_tracker_->update(armors_);

    //   if (one_ca_tracker_->tracker_state == Tracker::DETECTING) {
    //     one_target.tracking = false;
    //   } else if (one_ca_tracker_->tracker_state == Tracker::TRACKING ||
    //     one_ca_tracker_->tracker_state == Tracker::TEMP_LOST) {
    //       one_target.tracking = true;

    //     const auto &state = one_ca_tracker_->target_state;
    //     one_target.id = one_ca_tracker_->tracked_id;

    //     one_target.position_.x = state(0);
    //     one_target.velocity_.x = state(1);
    //     one_target.acceleration_.x = state(2);
    //     one_target.position_.y = state(3);
    //     one_target.velocity_.y = state(4);
    //     one_target.acceleration_.y = state(5);
    //     one_target.position_.z = state(6);
    //     one_target.velocity_.z = state(7);
    //     one_target.yaw = state(8);
    //     one_target.v_yaw = state(9);
    //     bool hasNaN = false;
    //     for (int i = 0; i < 9; ++i) {
    //         if (std::isnan(state[i])) {
    //             hasNaN = true;
    //             break;
    //         }
    //     }
    //     if (hasNaN) {
    //         std::cerr << "State vector contains NaN!" << std::endl;
    //         one_ca_tracker_->tracker_state = OneCaTracker::State::LOST;

    //     }

    //     one_target.type = one_ca_tracker_->type;
    //   }
    // }
    // one_targets_.push_back(one_target);
    // std::cout<<one_target.position_<<" "<<one_target.velocity_<<"
    // "<<one_target.acceleration_<<" "<<one_target.yaw<<"
    // "<<one_target.v_yaw<<std::endl; one_targets_.push_back(one_target);
    // if (!target_.tracking || std::abs(target_.v_yaw) < v_yaw_to_one_thres_) {
    //     std::vector<bool> armor_assigned(armors_.armors.size(), false);

    //     for (auto& otracker: one_trackers_) {
    //         OneTarget target;

    //         if (otracker->tracker_state == Tracker::LOST) {
    //             int best_i = -1;
    //             double min_dist_center = std::numeric_limits<double>::max();
    //             for (size_t i = 0; i < armors_.armors.size(); ++i) {
    //                 if (!armor_assigned[i]) {
    //                     double dist_center = armors_.armors[i].distance_to_image_center;
    //                     if (dist_center < min_dist_center) {
    //                         min_dist_center = dist_center;
    //                         best_i = static_cast<int>(i);
    //                     }
    //                 }
    //             }

    //             if (best_i >= 0) {
    //                 otracker->init({ armors_.armors[best_i] });
    //                 armor_assigned[best_i] = true;
    //             }

    //             target.tracking = false;
    //             one_targets_.push_back(target);
    //             continue;
    //         }

    //         // 设置预测函数
    //         otracker->lost_thres = std::abs(static_cast<int>(one_lost_time_thres_ / dt_));
    //         if (otracker->tracked_id == ArmorNumber::OUTPOST) {
    //             otracker->ekf->setPredictFunc(onearmor_motion_model::Predict {
    //                 dt_,
    //                 onearmor_motion_model::MotionModel::CONSTANT_ROTATION });
    //         } else {
    //             otracker->ekf->setPredictFunc(onearmor_motion_model::Predict {
    //                 dt_,
    //                 onearmor_motion_model::MotionModel::CONSTANT_VEL_ROT });
    //         }

    //         // 预测当前状态
    //         Eigen::VectorXd ekf_prediction = otracker->ekf->predict();
    //         Eigen::Vector3d predicted_position =
    //             otracker->getArmorPositionFromState(ekf_prediction);

    //         // 匹配观测中最近的装甲板
    //         int best_i = -1;
    //         double min_dist = std::numeric_limits<double>::max();

    //         for (size_t i = 0; i < armors_.armors.size(); ++i) {
    //             if (armor_assigned[i])
    //                 continue;

    //             // 类型不匹配
    //             if (!isSameTarget(armors_.armors[i].number, otracker->tracked_id))
    //                 continue;
    //             // 提取观测装甲板的位置和 yaw
    //             const auto& armor = armors_.armors[i];
    //             Eigen::Vector3d obs_pos(armor.target_pos.x, armor.target_pos.y, armor.target_pos.z);
    //             double obs_yaw = otracker->orientationToYaw(armor.target_ori);

    //             // 计算各项误差
    //             double pos_dist = (obs_pos - predicted_position).norm();
    //             double yaw_diff = std::abs(normalizeAngle(obs_yaw - otracker->target_state(6)));
    //             double z_diff = std::abs(obs_pos.z() - predicted_position.z());

    //             // 不满足阈值条件，跳过
    //             if (pos_dist > otracker->max_match_distance_)
    //                 continue;
    //             if (yaw_diff > otracker->max_match_yaw_diff_)
    //                 continue;
    //             if (z_diff > otracker->max_match_z_diff_)
    //                 continue;

    //             // 选择最小距离匹配
    //             if (pos_dist < min_dist) {
    //                 min_dist = pos_dist;
    //                 best_i = static_cast<int>(i);
    //             }
    //         }

    //         if (best_i >= 0) {
    //             otracker->update({ armors_.armors[best_i] });
    //             armor_assigned[best_i] = true;
    //         } else {
    //             // 无匹配，发送空观测
    //             Armor empty_armor;
    //             otracker->update(empty_armor);
    //         }

    //         // 状态同步到目标信息
    //         if (otracker->tracker_state == Tracker::TRACKING
    //             || otracker->tracker_state == Tracker::TEMP_LOST) {
    //             const auto& state = otracker->target_state;
    //             target.tracking = true;
    //             target.id = otracker->tracked_id;
    //             target.position_.x = state(0);
    //             target.velocity_.x = state(1);
    //             target.position_.y = state(2);
    //             target.velocity_.y = state(3);
    //             target.position_.z = state(4);
    //             target.velocity_.z = state(5);
    //             target.yaw = state(6);
    //             target.v_yaw = state(7);
    //             target.type = otracker->type;
    //             target.distance_to_image_center = otracker->distance_to_image_center;
    //         } else {
    //             target.tracking = false;
    //         }

    //         one_targets_.push_back(target);
    //     }
    // }

    if (!target_.tracking || std::abs(target_.v_yaw) < v_yaw_to_one_thres_) {
        std::vector<bool> armor_assigned(armors_.armors.size(), false);

        for (auto& otracker: one_ypd_trackers_) {
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

            // 设置预测函数
            otracker->lost_thres = std::abs(static_cast<int>(one_lost_time_thres_ / dt_));
            if (otracker->tracked_id == ArmorNumber::OUTPOST) {
                otracker->ekf->setPredictFunc(oneypdarmor_motion_model::Predict {
                    dt_,
                    oneypdarmor_motion_model::MotionModel::CONSTANT_ROTATION });
            } else {
                otracker->ekf->setPredictFunc(oneypdarmor_motion_model::Predict {
                    dt_,
                    oneypdarmor_motion_model::MotionModel::CONSTANT_VEL_ROT });
            }

            // 预测当前状态
            Eigen::VectorXd ekf_prediction = otracker->ekf->predict();
            Eigen::Vector3d predicted_position =
                otracker->getArmorPositionFromState(ekf_prediction);

            // 匹配观测中最近的装甲板
            int best_i = -1;
            double min_dist = std::numeric_limits<double>::max();

            for (size_t i = 0; i < armors_.armors.size(); ++i) {
                if (armor_assigned[i])
                    continue;

                // 类型不匹配
                if (!isSameTarget(armors_.armors[i].number, otracker->tracked_id))
                    continue;
                // 提取观测装甲板的位置和 yaw
                const auto& armor = armors_.armors[i];
                Eigen::Vector3d obs_pos(armor.target_pos.x, armor.target_pos.y, armor.target_pos.z);
                double obs_yaw = otracker->orientationToYaw(armor.target_ori);

                // 计算各项误差
                double pos_dist = (obs_pos - predicted_position).norm();
                double yaw_diff = std::abs(normalizeAngle(obs_yaw - otracker->target_state(6)));
                double z_diff = std::abs(obs_pos.z() - predicted_position.z());

                // 不满足阈值条件，跳过
                if (pos_dist > otracker->max_match_distance_)
                    continue;
                if (yaw_diff > otracker->max_match_yaw_diff_)
                    continue;
                if (z_diff > otracker->max_match_z_diff_)
                    continue;

                // 选择最小距离匹配
                if (pos_dist < min_dist) {
                    min_dist = pos_dist;
                    best_i = static_cast<int>(i);
                }
            }

            if (best_i >= 0) {
                otracker->update({ armors_.armors[best_i] });
                armor_assigned[best_i] = true;
            } else {
                // 无匹配，发送空观测
                Armor empty_armor;
                otracker->update(empty_armor);
            }

            // 状态同步到目标信息
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

            one_targets_.push_back(target);
        }
    }


    last_time_ = time;
}
