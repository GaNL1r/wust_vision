#include "tracker/tracker_manager.hpp"

TrackerManager::TrackerManager(const YAML::Node &config_) {
  track_one_num = config_["track_one_num"].as<int>(2);
  // Tracker 基础参数
  double max_match_distance = config_["max_match_distance"].as<double>(0.2);
  double max_match_yaw_diff = config_["max_match_yaw_diff"].as<double>(1.0);
  double max_match_z_diff = config_["max_match_z_diff"].as<double>(0.1);
  // tracker_ = std::make_unique<Tracker>(max_match_distance, max_match_yaw_diff,
  //                                      max_match_z_diff);
  ypd_tracker_ = std::make_unique<YpdTracker>(
      max_match_distance, max_match_yaw_diff, max_match_z_diff);
  one_tracker_ = std::make_unique<OneTracker>(
      max_match_distance, max_match_yaw_diff, max_match_z_diff);
  // one_ypd_tracker_ = std::make_unique<OneYpdTracker>(
  //     max_match_distance, max_match_yaw_diff, max_match_z_diff);

  for (int i = 0; i < track_one_num; i++) {
    auto o_tracker_ = std::make_unique<OneTracker>(
        max_match_distance, max_match_yaw_diff, max_match_z_diff);
    one_trackers_.push_back(std::move(o_tracker_));
  }
  // for (int i = 0; i < track_one_num; i++) {
  //   auto oy_tracker_ = std::make_unique<OneYpdTracker>(
  //       max_match_distance, max_match_yaw_diff, max_match_z_diff);
  //   one_ypd_trackers_.push_back(std::move(oy_tracker_));
  // }
  one_tracker_->buffer_size_ = config_["obs_vyaw_buffer_thres"].as<int>(5);
  one_tracker_->obs_yaw_stationary_thresh =
      config_["obs_yaw_stationary_thresh"].as<float>(1.0);
  one_tracker_->pred_yaw_stationary_thresh =
      config_["pred_yaw_stationary_thresh"].as<float>(0.5);
  one_tracker_->min_valid_velocity =
      config_["min_valid_velocity_thresh"].as<float>(0.01);
  one_tracker_->max_inconsistent_count_ =
      config_["max_inconsistent_count"].as<int>(3);
  one_tracker_->rotation_inconsistent_cooldown_limit_ =
      config_["rotation_inconsistent_cooldown_limit"].as<int>(5);
  one_tracker_->jump_thresh = config_["jump_thresh"].as<double>(0.4);


  v_yaw_update_thres_= config_["v_yaw_update_thres"].as<float>(0.01);
  v_yaw_to_one_thres_ = config_["v_yaw_to_one_thres"].as<float>(0.01);

  // 跟踪判定参数
  //tracker_->tracking_thres = config_["tracking_thres"].as<int>(5);
  one_tracker_->tracking_thres = config_["tracking_thres"].as<int>(5);
  ypd_tracker_->tracking_thres = config_["tracking_thres"].as<int>(5);
  //one_ypd_tracker_->tracking_thres = config_["tracking_thres"].as<int>(5);
  lost_time_thres_ = config_["lost_time_thres"].as<double>(0.3);
  one_lost_time_thres_ = config_["one_lost_time_thres"].as<double>(0.1);

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
    double t = dt_, x = s2qx_, y = s2qy_, z = s2qz_, yaw = s2qyaw_, r = s2qr_,
           d_zc = s2qd_zc_;
    double q_x_x = pow(t, 4) / 4 * x, q_x_vx = pow(t, 3) / 2 * x,
           q_vx_vx = pow(t, 2) * x;
    double q_y_y = pow(t, 4) / 4 * y, q_y_vy = pow(t, 3) / 2 * y,
           q_vy_vy = pow(t, 2) * y;
    double q_z_z = pow(t, 4) / 4 * z, q_z_vz = pow(t, 3) / 2 * z,
           q_vz_vz = pow(t, 2) * z;
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
    Eigen::Matrix<double, ypdarmor_motion_model::X_N,
                  ypdarmor_motion_model::X_N>
        q;
    double t = dt_, x = ys2qx_, y = ys2qy_, z = ys2qz_, yaw = ys2qyaw_,
           r = ys2qr_, d_zc = ys2qd_zc_;
    double q_x_x = pow(t, 4) / 4 * x, q_x_vx = pow(t, 3) / 2 * x,
           q_vx_vx = pow(t, 2) * x;
    double q_y_y = pow(t, 4) / 4 * y, q_y_vy = pow(t, 3) / 2 * y,
           q_vy_vy = pow(t, 2) * y;
    double q_z_z = pow(t, 4) / 4 * z, q_z_vz = pow(t, 3) / 2 * z,
           q_vz_vz = pow(t, 2) * z;
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
    Eigen::Matrix<double, onearmor_motion_model::X_N,
                  onearmor_motion_model::X_N>
        q;
    double t = dt_, x = os2qx_, y = os2qy_, z = os2qz_, yaw = os2qyaw_;
    double q_x_x = pow(t, 4) / 4 * x, q_x_vx = pow(t, 3) / 2 * x,
           q_vx_vx = pow(t, 2) * x;
    double q_y_y = pow(t, 4) / 4 * y, q_y_vy = pow(t, 3) / 2 * y,
           q_vy_vy = pow(t, 2) * y;
    double q_z_z = pow(t, 4) / 4 * z, q_z_vz = pow(t, 3) / 2 * z,
           q_vz_vz = pow(t, 2) * z;
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
    Eigen::Matrix<double, oneypdarmor_motion_model::X_N,
                  oneypdarmor_motion_model::X_N>
        q;
    double t = dt_, x = oys2qx_, y = oys2qy_, z = oys2qz_, yaw = oys2qyaw_;
    double q_x_x = pow(t, 4) / 4 * x, q_x_vx = pow(t, 3) / 2 * x,
           q_vx_vx = pow(t, 2) * x;
    double q_y_y = pow(t, 4) / 4 * y, q_y_vy = pow(t, 3) / 2 * y,
           q_vy_vy = pow(t, 2) * y;
    double q_z_z = pow(t, 4) / 4 * z, q_z_vz = pow(t, 3) / 2 * z,
           q_vz_vz = pow(t, 2) * z;
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
    Eigen::Matrix<double, onecaarmor_motion_model::X_N,
                  onecaarmor_motion_model::X_N>
        q;
    double t = dt_;
    double x = os2qx_, y = os2qy_, z = os2qz_, yaw = os2qyaw_;
    
    // 按照“加速度是白噪声”的协方差传播公式
    double q_x_x     = pow(t, 4) / 4.0 * x;
    double q_x_vx    = pow(t, 3) / 2.0 * x;
    double q_vx_vx   = pow(t, 2)       * x;

    double q_y_y     = pow(t, 4) / 4.0 * y;
    double q_y_vy    = pow(t, 3) / 2.0 * y;
    double q_vy_vy   = pow(t, 2)       * y;

    double q_z_z     = pow(t, 4) / 4.0 * z;
    double q_z_vz    = pow(t, 3) / 2.0 * z;
    double q_vz_vz   = pow(t, 2)       * z;

    double q_yaw_yaw     = pow(t, 4) / 4.0 * yaw;
    double q_yaw_vyaw    = pow(t, 3) / 2.0 * yaw;
    double q_vyaw_vyaw   = pow(t, 2)       * yaw;

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


  // EKF 观测噪声协方差 R（基于测量值调整）
  auto u_r = [this](
                 const Eigen::Matrix<double, armor_motion_model::Z_N, 1> &z) {
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
  auto yu_r =
      [this](const Eigen::Matrix<double, ypdarmor_motion_model::Z_N, 1> &z) {
        Eigen::Matrix<double, ypdarmor_motion_model::Z_N,
                      ypdarmor_motion_model::Z_N>
            r;
        // clang-format off
r << yr_y_      * std::abs(z[0]), 0, 0, 0,
     0, yr_p_ * std::abs(z[1]), 0, 0,
     0, 0, yr_d_ * std::abs(z[2]), 0,
     0, 0, 0, yr_yaw_;
        // clang-format on
        return r;
      };

  auto ou_r =
      [this](const Eigen::Matrix<double, onearmor_motion_model::Z_N, 1> &z) {
        Eigen::Matrix<double, onearmor_motion_model::Z_N,
                      onearmor_motion_model::Z_N>
            r;
        // clang-format off
      r << or_x_ * std::abs(z[0]), 0, 0, 0,
           0, or_y_ * std::abs(z[1]), 0, 0,
           0, 0, or_z_ * std::abs(z[2]), 0,
           0, 0, 0, or_yaw_;
        // clang-format on
        return r;
      };
  auto oyu_r =
      [this](const Eigen::Matrix<double, oneypdarmor_motion_model::Z_N, 1> &z) {
        Eigen::Matrix<double, oneypdarmor_motion_model::Z_N,
                      oneypdarmor_motion_model::Z_N>
            r;
        // clang-format off
r << oyr_y_      * std::abs(z[0]), 0, 0, 0,
     0, oyr_p_ * std::abs(z[1]), 0, 0,
     0, 0, oyr_d_ * std::abs(z[2]), 0,
     0, 0, 0, oyr_yaw_;
        // clang-format on
        return r;
      };
      auto ocau_r =
      [this](const Eigen::Matrix<double, onecaarmor_motion_model::Z_N, 1> &z) {
        Eigen::Matrix<double, onecaarmor_motion_model::Z_N,
                      onecaarmor_motion_model::Z_N>
            r;
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
  // tracker_->ekf =
  //     std::make_unique<armor_motion_model::RobotStateEKF>(f, h, u_q, u_r, p0);
  ypd_tracker_->ekf = std::make_unique<ypdarmor_motion_model::RobotStateEKF>(
      yf, yh, yu_q, yu_r, yp0);
  one_tracker_->ekf = std::make_unique<onearmor_motion_model::RobotStateEKF>(
      of, oh, ou_q, ou_r, op0);
  // one_ypd_tracker_->ekf =
  //     std::make_unique<oneypdarmor_motion_model::RobotStateEKF>(oyf, oyh, oyu_q,
  //                                                               oyu_r, oyp0);
  for (auto &o_tracker : one_trackers_) {
    o_tracker->ekf = std::make_unique<onearmor_motion_model::RobotStateEKF>(
        of, oh, ou_q, ou_r, op0);
  }
  // for (auto &oy_tracker : one_ypd_trackers_) {
  //   oy_tracker->ekf = std::make_unique<oneypdarmor_motion_model::RobotStateEKF>(
  //       oyf, oyh, oyu_q, oyu_r, oyp0);
  // }
  
  
}

void TrackerManager::update(Target &target_,
                            std::vector<OneTarget> &one_targets_,
                            Armors armors_,
                            std::chrono::steady_clock::time_point time) {
    static int init_count_ = -1;  

   if (ypd_tracker_->tracker_state == Tracker::LOST || init_count_ == 500) {
    ypd_tracker_->init(armors_);
    target_.tracking = false;

  } else {
    dt_ = std::chrono::duration<double>(time - last_time_).count();
    ypd_tracker_->lost_thres =
        std::abs(static_cast<int>(lost_time_thres_ / dt_));
    if (ypd_tracker_->tracked_id == ArmorNumber::OUTPOST) {
      ypd_tracker_->ekf->setPredictFunc(ypdarmor_motion_model::Predict{
          dt_, ypdarmor_motion_model::MotionModel::CONSTANT_ROTATION});
    } else {
      ypd_tracker_->ekf->setPredictFunc(ypdarmor_motion_model::Predict{
          dt_, ypdarmor_motion_model::MotionModel::CONSTANT_VEL_ROT});
    }
    ypd_tracker_->update(armors_);

    if (init_count_ >= 0)
      ++init_count_;  

    if (ypd_tracker_->tracker_state == Tracker::DETECTING) {
      target_.tracking = false;
    } else if (ypd_tracker_->tracker_state == Tracker::TRACKING ||
               ypd_tracker_->tracker_state == Tracker::TEMP_LOST) {
      target_.tracking = true;

      const auto &state = ypd_tracker_->target_state;
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

  if (!target_.tracking || std::abs(target_.v_yaw )<v_yaw_to_one_thres_ ) {
    std::vector<bool> armor_assigned(armors_.armors.size(), false);

    for (auto &otracker : one_trackers_) {
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
          otracker->init({armors_.armors[best_i]});
          armor_assigned[best_i] = true;
        }

        target.tracking = false;
        one_targets_.push_back(target);
        continue;
      }

      // 设置预测函数
      otracker->lost_thres =
          std::abs(static_cast<int>(one_lost_time_thres_ / dt_));
      if (otracker->tracked_id == ArmorNumber::OUTPOST) {
        otracker->ekf->setPredictFunc(onearmor_motion_model::Predict{
            dt_, onearmor_motion_model::MotionModel::CONSTANT_ROTATION});
      } else {
        otracker->ekf->setPredictFunc(onearmor_motion_model::Predict{
            dt_, onearmor_motion_model::MotionModel::CONSTANT_VEL_ROT});
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
        if (retypetotracker(armors_.armors[i].number) != otracker->retype)
          continue;

        // 提取观测装甲板的位置和 yaw
        const auto &armor = armors_.armors[i];
        Eigen::Vector3d obs_pos(armor.target_pos.x, armor.target_pos.y,
                                armor.target_pos.z);
        double obs_yaw =otracker->orientationToYaw(armor.target_ori) ;

        // 计算各项误差
        double pos_dist = (obs_pos - predicted_position).norm();
        double yaw_diff =
            std::abs(normalizeAngle(obs_yaw - otracker->target_state(6)));
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
        otracker->update({armors_.armors[best_i]});
        armor_assigned[best_i] = true;
      } else {
        // 无匹配，发送空观测
        Armor empty_armor;
        otracker->update(empty_armor);
      }

      // 状态同步到目标信息
      if (otracker->tracker_state == Tracker::TRACKING ||
          otracker->tracker_state == Tracker::TEMP_LOST) {
        const auto &state = otracker->target_state;
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


  // if (!target_.tracking || target_.v_yaw < 1.0 && target_.v_yaw > -1.0) {
  //   std::vector<bool> armor_assigned(armors_.armors.size(), false);

  //   for (auto &otracker : one_ypd_trackers_) {
  //     OneTarget target;

  //     if (otracker->tracker_state == Tracker::LOST) {
  //       int best_i = -1;
  //       double min_dist_center = std::numeric_limits<double>::max();
  //       for (size_t i = 0; i < armors_.armors.size(); ++i) {
  //         if (!armor_assigned[i]) {
  //           double dist_center = armors_.armors[i].distance_to_image_center;
  //           if (dist_center < min_dist_center) {
  //             min_dist_center = dist_center;
  //             best_i = static_cast<int>(i);
  //           }
  //         }
  //       }

  //       if (best_i >= 0) {
  //         otracker->init({armors_.armors[best_i]});
  //         armor_assigned[best_i] = true;
  //       }

  //       target.tracking = false;
  //       one_targets_.push_back(target);
  //       continue;
  //     }

  //     // 设置预测函数
  //     otracker->lost_thres =
  //         std::abs(static_cast<int>(one_lost_time_thres_ / dt_));
  //     if (otracker->tracked_id == ArmorNumber::OUTPOST) {
  //       otracker->ekf->setPredictFunc(oneypdarmor_motion_model::Predict{
  //           dt_, oneypdarmor_motion_model::MotionModel::CONSTANT_ROTATION});
  //     } else {
  //       otracker->ekf->setPredictFunc(oneypdarmor_motion_model::Predict{
  //           dt_, oneypdarmor_motion_model::MotionModel::CONSTANT_VEL_ROT});
  //     }

  //     // 预测当前状态
  //     Eigen::VectorXd ekf_prediction = otracker->ekf->predict();
  //     Eigen::Vector3d predicted_position =
  //         otracker->getArmorPositionFromState(ekf_prediction);

  //     // 匹配观测中最近的装甲板
  //     int best_i = -1;
  //     double min_dist = std::numeric_limits<double>::max();

  //     for (size_t i = 0; i < armors_.armors.size(); ++i) {
  //       if (armor_assigned[i])
  //         continue;

  //       // 类型不匹配
  //       if (retypetotracker(armors_.armors[i].number) != otracker->retype)
  //         continue;

  //       // 提取观测装甲板的位置和 yaw
  //       const auto &armor = armors_.armors[i];
  //       Eigen::Vector3d obs_pos(armor.target_pos.x, armor.target_pos.y,
  //                               armor.target_pos.z);
  //       double obs_yaw = armor.yaw;

  //       // 计算各项误差
  //       double pos_dist = (obs_pos - predicted_position).norm();
  //       double yaw_diff =
  //           std::abs(normalizeAngle(obs_yaw - otracker->target_state(6)));
  //       double z_diff = std::abs(obs_pos.z() - predicted_position.z());

  //       // 不满足阈值条件，跳过
  //       if (pos_dist > otracker->max_match_distance_)
  //         continue;
  //       if (yaw_diff > otracker->max_match_yaw_diff_)
  //         continue;
  //       if (z_diff > otracker->max_match_z_diff_)
  //         continue;

  //       // 选择最小距离匹配
  //       if (pos_dist < min_dist) {
  //         min_dist = pos_dist;
  //         best_i = static_cast<int>(i);
  //       }
  //     }

  //     if (best_i >= 0) {
  //       otracker->update({armors_.armors[best_i]});
  //       armor_assigned[best_i] = true;
  //     } else {
  //       // 无匹配，发送空观测
  //       Armor empty_armor;
  //       otracker->update(empty_armor);
  //     }

  //     // 状态同步到目标信息
  //     if (otracker->tracker_state == Tracker::TRACKING ||
  //         otracker->tracker_state == Tracker::TEMP_LOST) {
  //       const auto &state = otracker->target_state;
  //       target.tracking = true;
  //       target.id = otracker->tracked_id;
  //       target.position_.x = state(0);
  //       target.velocity_.x = state(1);
  //       target.position_.y = state(2);
  //       target.velocity_.y = state(3);
  //       target.position_.z = state(4);
  //       target.velocity_.z = state(5);
  //       target.yaw = state(6);
  //       target.v_yaw = state(7);
  //       target.type = otracker->type;
  //       target.distance_to_image_center = otracker->distance_to_image_center;
  //     } else {

  //       target.tracking = false;
  //     }

  //     one_targets_.push_back(target);
  //   }
  // }
  // OneTarget one_target_;
  // if (one_tracker_->tracker_state == Tracker::LOST) {
  //   one_tracker_->init(armors_);
  //   one_target_.tracking = false;
  // } else {
  //   dt_ = std::chrono::duration<double>(time - last_time_).count();
  //   one_tracker_->lost_thres = std::abs(static_cast<int>(lost_time_thres_
  //   / dt_)); if (one_tracker_->tracked_id == ArmorNumber::OUTPOST) {
  //     one_tracker_->ekf->setPredictFunc(onearmor_motion_model::Predict{
  //         dt_, onearmor_motion_model::MotionModel::CONSTANT_ROTATION});
  //   } else {
  //     one_tracker_->ekf->setPredictFunc(onearmor_motion_model::Predict{
  //         dt_, onearmor_motion_model::MotionModel::CONSTANT_VEL_ROT});
  //   }
  //   one_tracker_->update(armors_);

  //   if (one_tracker_->tracker_state == Tracker::DETECTING) {
  //     one_target_.tracking = false;
  //   } else if (one_tracker_->tracker_state == Tracker::TRACKING ||
  //     one_tracker_->tracker_state == Tracker::TEMP_LOST) {
  //       one_target_.tracking = true;
  //     // Fill target
  //     const auto &state = one_tracker_->target_state;
  //     one_target_.id = one_tracker_->tracked_id;

  //     one_target_.position_.x = state(0);
  //     one_target_.velocity_.x = state(1);
  //     one_target_.position_.y = state(2);
  //     one_target_.velocity_.y = state(3);
  //     one_target_.position_.z = state(4);
  //     one_target_.velocity_.z = state(5);
  //     one_target_.yaw = state(6);
  //     one_target_.v_yaw = state(7);
  //     one_target_.type = one_tracker_->type;
  //     //std::cout<<"v_yaw: "<<one_target_.v_yaw<<std::endl;

  //   }
  // }
  // //one_targets_.push_back(one_target_);

  // if(std::abs(one_target_.v_yaw-target_.v_yaw)>v_yaw_update_thres_)
  // {
  //   ypd_tracker_->updatev_yaw(one_target_.v_yaw);
  // }

  
  // if (one_ypd_tracker_->tracker_state == Tracker::LOST) {
  //   one_ypd_tracker_->init(armors_);
  //   one_target_.tracking = false;
  // } else {
  //   dt_ = std::chrono::duration<double>(time - last_time_).count();
  //   one_ypd_tracker_->lost_thres = std::abs(static_cast<int>(lost_time_thres_
  //   / dt_)); if (one_ypd_tracker_->tracked_id == ArmorNumber::OUTPOST) {
  //     one_ypd_tracker_->ekf->setPredictFunc(oneypdarmor_motion_model::Predict{
  //         dt_, oneypdarmor_motion_model::MotionModel::CONSTANT_ROTATION});
  //   } else {
  //     one_ypd_tracker_->ekf->setPredictFunc(oneypdarmor_motion_model::Predict{
  //         dt_, oneypdarmor_motion_model::MotionModel::CONSTANT_VEL_ROT});
  //   }
  //   one_ypd_tracker_->update(armors_);

  //   if (one_ypd_tracker_->tracker_state == Tracker::DETECTING) {
  //     one_target_.tracking = false;
  //   } else if (one_ypd_tracker_->tracker_state == Tracker::TRACKING ||
  //     one_ypd_tracker_->tracker_state == Tracker::TEMP_LOST) {
  //       one_target_.tracking = true;
  //     // Fill target
  //     const auto &state = one_ypd_tracker_->target_state;
  //     one_target_.id = one_ypd_tracker_->tracked_id;

  //     one_target_.position_.x = state(0);
  //     one_target_.velocity_.x = state(1);
  //     one_target_.position_.y = state(2);
  //     one_target_.velocity_.y = state(3);
  //     one_target_.position_.z = state(4);
  //     one_target_.velocity_.z = state(5);
  //     one_target_.yaw = state(6);
  //     one_target_.v_yaw = state(7);
  //     one_target_.type = one_ypd_tracker_->type;

  //   }
  // }
  // one_targets_.push_back(one_target_);
  //  if(std::abs(one_target_.v_yaw-target_.v_yaw)>1.0)
  // {
  //   ypd_tracker_->updatev_yaw(one_target_.v_yaw);
  // }


  last_time_ = time;
}
