#include "rune_target.hpp"
namespace rune {
RuneTarget::RuneTarget(
    bool is_big,
    const rune::RuneFan& fan,
    const RuneTargetConfig& target_config,
    double pre_v_roll
) {
    is_big_ = is_big;
    start_time_ = fan.timestamp;
    target_config_ = target_config;
    auto f = ypdrune_motion_model::Predict(0.005);
    auto h = ypdrune_motion_model::Measure(0);
    auto u_q = [this]() {
        Eigen::Matrix<double, ypdrune_motion_model::X_N, ypdrune_motion_model::X_N> q;
        return q;
    };

    auto u_r = [this](const Eigen::Matrix<double, ypdrune_motion_model::Z_N, 1>& z) {
        Eigen::Matrix<double, ypdrune_motion_model::Z_N, ypdrune_motion_model::Z_N> r;
        return r;
    };
    Eigen::DiagonalMatrix<double, ypdrune_motion_model::X_N> p0;
    p0.setIdentity();
    esekf_ypd_ = ypdrune_motion_model::RuneESKF(f, h, u_q, u_r, p0);
    esekf_ypd_.setAngleDims({ 0, 3, 4 });
    esekf_ypd_.setIterationNum(1);
    esekf_ypd_.setInjectFunc([](const Eigen::Matrix<double, ypdrune_motion_model::X_N, 1>& delta,
                                Eigen::Matrix<double, ypdrune_motion_model::X_N, 1>& nominal) {
        for (int i = 0; i < ypdrune_motion_model::X_N; i++) {
            if (i == 3 || i == 4)
                continue;
            nominal[i] += delta[i];
        }
        nominal[3] = angles::normalize_angle(nominal[3] + delta[3]);
        nominal[4] = angles::normalize_angle(nominal[4] + delta[4]);
    });
    esekf_ypd_.setNisThreshold(9.488);
    esekf_ypd_.setNeesThreshold(9.488);
    esekf_ypd_.setWindowSize(50);
    esekf_ypd_.setRecentFailRateThreshold(0.4);

    double xc = fan.target_pos.x();
    double yc = fan.target_pos.y();
    double zc = fan.target_pos.z();
    double yaw = orientationToYaw(fan.target_ori);
    double roll = orientationToRoll(fan.target_ori);
    target_state_ = Eigen::VectorXd::Zero(ypdrune_motion_model::X_N);
    target_state_ << xc, yc, zc, yaw, roll, pre_v_roll;
    esekf_ypd_.setState(target_state_);
    fitter_.update(0, 0);
    last_time_ = 0;
    is_inited = true;
    last_t_ = fan.timestamp;
    timestamp_ = fan.timestamp;
}
Eigen::Matrix<double, ypdrune_motion_model::Z_N, ypdrune_motion_model::Z_N>
RuneTarget::computeMeasurementCovariance(
    const Eigen::Matrix<double, ypdrune_motion_model::Z_N, 1>& z
) const {
    Eigen::Matrix<double, ypdrune_motion_model::Z_N, ypdrune_motion_model::Z_N> r;
    // clang-format off
    r << target_config_.yp_r , 0 , 0 ,  0 , 0,
         0 , target_config_.yp_r , 0 ,  0 , 0,
         0 , 0 , target_config_.dis_r , 0 , 0,
         0 , 0 , 0 , target_config_.yaw_r , 0,
         0 , 0 , 0 , 0 , target_config_.roll_r;
    // clang-format on
    return r;
}
Eigen::Matrix<double, ypdrune_motion_model::X_N, ypdrune_motion_model::X_N>
RuneTarget::computeProcessNoise(double dt) const {
    Eigen::Matrix<double, ypdrune_motion_model::X_N, ypdrune_motion_model::X_N> q;
    double t = dt;
    double v1 = target_config_.q_roll;
    double q_roll_roll = pow(t, 4) / 4 * v1, q_roll_vroll = pow(t, 3) / 2 * v1,
           q_vroll_vroll = pow(t, 2) * v1;
    double q_xyz = target_config_.q_xyz;
    double q_yaw = target_config_.q_yaw;
    // clang-format off
    //   xc   yc   zc  yaw  roll           v_roll
    q << q_xyz,      0,        0,        0,        0,              0,            
         0,          q_xyz,    0,        0,        0,              0,             
         0,          0,        q_xyz,    0,        0,              0,            
         0,          0,        0,        q_yaw,    0,              0,              
         0,          0,        0,        0,        q_roll_roll,    q_roll_vroll,  
         0,          0,        0,        0,        q_roll_vroll,   q_vroll_vroll;

    // clang-format on
    return q;
}
void RuneTarget::predict(std::chrono::steady_clock::time_point t) {
    double dt = time_utils::durationSec(last_t_, t);

    predict(dt);

    last_t_ = t;
}
void RuneTarget::predict(double dt) {
    dt_ = dt;

    esekf_ypd_.setPredictFunc(ypdrune_motion_model::Predict { dt });
    auto u_q = [dt, this]() { return computeProcessNoise(dt); };

    esekf_ypd_.setUpdateQ(u_q);

    target_state_ = esekf_ypd_.predict();

    if (centerPos().norm() < 0.5) {
        is_tracking = false;
    }
}
bool RuneTarget::update(const rune::RuneFan& fan) {
    timestamp_ = fan.timestamp;
    int id;
    auto min_angle_error = 1e10;
    const auto angles = getAngles();
    for (int i = 0; i < angles.size(); i++) {
        auto angle_error = std::abs(angles::normalize_angle(
            angles::normalize_angle(orientationToRoll(fan.target_ori)) - angles[i]
        ));
        if (angle_error < min_angle_error) {
            min_angle_error = angle_error;
            id = i;
        }
    }
    auto p = fan.target_pos;
    double measured_yaw = orientationToYaw(fan.target_ori);
    double measured_roll = orientationToRoll(fan.target_ori);
    double ypd_y = std::atan2(p.y(), p.x());
    ypd_y = this->last_ypd_y + angles::shortest_angular_distance(this->last_ypd_y, ypd_y);
    this->last_ypd_y = ypd_y;
    double ypd_p = std::atan2(p.z(), std::sqrt(p.x() * p.x() + p.y() * p.y()));
    double ypd_d = std::sqrt(p.x() * p.x() + p.y() * p.y() + p.z() * p.z());
    measurement_ << ypd_y, ypd_p, ypd_d, measured_yaw, measured_roll;
    auto tmp_esekf = esekf_ypd_;
    tmp_esekf.setMeasureFunc(ypdrune_motion_model::Measure { id });
    tmp_esekf.update(measurement_);
    auto tmp_state = tmp_esekf.predict();
    if (diverged(tmp_state) || std::abs(tmp_state[5] - v_roll()) > M_PI / 10) {
        WUST_WARN("target") << "This update make diverged skip!!";
        return false;
    }
    last_id = id;
    auto yu_rv2 = [this](const Eigen::Matrix<double, ypdrune_motion_model::Z_N, 1>& z) {
        return this->computeMeasurementCovariance(z);
    };
    esekf_ypd_.setUpdateR(yu_rv2);
    esekf_ypd_.setMeasureFunc(ypdrune_motion_model::Measure { id });

    esekf_ypd_.update(measurement_);
    double tostart = time_utils::durationSec(start_time_, fan.timestamp);
    fitter_.update(tostart, v_roll());
    fitter_.setAngleRef(tostart, roll());
    fitter_.fitAsync();
    last_time_ = tostart;
    return true;
}

} // namespace rune
