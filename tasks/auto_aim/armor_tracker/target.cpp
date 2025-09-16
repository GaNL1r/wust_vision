#pragma once
#include "target.hpp"
#include "wust_vl/common/utils/math.hpp"
#include <wust_vl/common/utils/timer.hpp>

Target::Target() {
    target_state_ = Eigen::Matrix<double, ypdv2armor_motion_model::X_N, 1>::Zero();
}

Target::Target(
    const armor::Armor& a,
    const TargetConfig& target_config,
    double radius,
    int armor_num,
    Eigen::DiagonalMatrix<double, ypdv2armor_motion_model::X_N> p0
):
    target_config_(target_config),
    armor_num_(armor_num),
    radius_pre_(radius) {
    using namespace ypdv2armor_motion_model;

    // 初始化状态
    target_state_ = Eigen::Matrix<double, X_N, 1>::Zero();

    // ESEKF 初始化
    Predict yfv2(0.005);
    Measure yhv2(0, Z_N);

    auto yu_qv2 = [this]() { return Eigen::Matrix<double, X_N, X_N>::Zero(); };

    esekf_ypd_ = RobotStateESEKF(
        yfv2,
        yhv2,
        yu_qv2,
        [this](const auto& z) { return this->computeMeasurementCovariance(z); },
        p0
    );
    esekf_ypd_.setAngleDims({ 0, 3 });
    esekf_ypd_.setIterationNum(2);
    esekf_ypd_.setInjectFunc([](const Eigen::Matrix<double, X_N, 1>& delta,
                                Eigen::Matrix<double, X_N, 1>& nominal) {
        for (int i = 0; i < X_N; i++) {
            if (i == 6)
                continue;
            nominal[i] += delta[i];
        }
        nominal[6] = angles::normalize_angle(nominal[6] + delta[6]);
    });
    esekf_ypd_.setNisThreshold(9.488);
    esekf_ypd_.setNeesThreshold(9.488);
    esekf_ypd_.setWindowSize(50);
    esekf_ypd_.setRecentFailRateThreshold(0.4);

    // 状态初始化
    double xa = a.target_pos.x();
    double ya = a.target_pos.y();
    double za = a.target_pos.z();
    double yaw = orientationToYaw(a.target_ori);
    last_yaw_ = yaw;
    double r = radius;
    double xc = xa + r * cos(yaw);
    double yc = ya + r * sin(yaw);
    double zc = za;
    target_state_ << xc, 0, yc, 0, zc, 0, yaw, 0, r, 0, 0;

    fg_estimator_ = RobotStateFACTOR();
    gtsam::Vector sigmas = gtsam::Vector::Constant(X_N, 1.0);
    sigmas(0) = sigmas(2) = sigmas(4) = 0.05; // pos
    sigmas(1) = sigmas(3) = sigmas(5) = 1.0; // vel
    sigmas(6) = 0.05; // yaw
    sigmas(7) = 1.0; // yaw_rate
    sigmas(8) = 0.1; // radius
    sigmas(9) = sigmas(10) = 1.0; // others
    step_ = 0;

    // 添加 Prior
    fg_estimator_.addPrior(
        gtsam_motion_generic::XKey(step_),
        target_state_,
        gtsam_motion_generic::noiseModel::Diagonal::Sigmas(sigmas)
    );

    esekf_ypd_.setState(target_state_);
    tracked_id_ = a.number;
    type_ = a.type;
    last_t_ = a.timestamp;
    timestamp_ = a.timestamp;
    is_inited = true;
    std::cout << "Target " << tracked_id_ << " inited." << std::endl;
}

Eigen::Matrix<double, ypdv2armor_motion_model::Z_N, ypdv2armor_motion_model::Z_N>
Target::computeMeasurementCovariance(
    const Eigen::Matrix<double, ypdv2armor_motion_model::Z_N, 1>& measurement
) const {
    Eigen::Matrix<double, ypdv2armor_motion_model::Z_N, ypdv2armor_motion_model::Z_N> R;

    double yaw_error = angles::normalize_angle(measurement[3] - measurement[0]);
    double abs_yaw_error = std::abs(yaw_error);

    // sin 插值函数，小值慢、大值快
    auto sinInterpolation =
        [](double value, double startVal, double endVal, double outputStart, double outputEnd
        ) -> double {
        double t = (value - startVal) / (endVal - startVal);
        t = std::clamp(t, 0.0, 1.0);
        double s = std::sin(t * M_PI / 2.0);
        return outputStart + s * (outputEnd - outputStart);
    };

    // clang-format off
    R << target_config_.yp_r, 0, 0, 0,
         0, target_config_.yp_r, 0, 0,
         0, 0, sinInterpolation(abs_yaw_error, 0.0, M_PI/2.0,
                                target_config_.dis_r_front, target_config_.dis_r_side)
                  + measurement[2]*measurement[2]*target_config_.dis2_r_ratio, 0,
         0, 0, 0,
         0, 0, 0,
         log(std::abs(measurement[2]) + 1) * target_config_.yaw_r_log_ratio
                  + sinInterpolation(M_PI/2.0 - abs_yaw_error, 0.0, M_PI/2.0,
                                     target_config_.yaw_r_base_side, target_config_.yaw_r_base_front);
    // clang-format on

    return R;
}

Eigen::Matrix<double, ypdv2armor_motion_model::X_N, ypdv2armor_motion_model::X_N>
Target::computeProcessNoise(double dt) const {
    using namespace ypdv2armor_motion_model;
    double v1 = (tracked_id_ == armor::ArmorNumber::OUTPOST) ? target_config_.qxyz_output
                                                             : target_config_.qxyz_common;
    double v2 = (tracked_id_ == armor::ArmorNumber::OUTPOST) ? target_config_.qyaw_output
                                                             : target_config_.qyaw_common;

    Eigen::Matrix<double, X_N, X_N> q = Eigen::Matrix<double, X_N, X_N>::Zero();
    q(0, 0) = q(2, 2) = q(4, 4) = pow(dt, 4) / 4 * v1;
    q(0, 1) = q(2, 3) = q(4, 5) = pow(dt, 3) / 2 * v1;
    q(1, 1) = q(3, 3) = q(5, 5) = pow(dt, 2) * v1;

    q(6, 6) = pow(dt, 4) / 4 * v2;
    q(6, 7) = pow(dt, 3) / 2 * v2;
    q(7, 7) = pow(dt, 2) * v2;

    return q;
}

std::vector<Eigen::Vector4d> Target::getArmorPosAndYaw() const {
    std::vector<Eigen::Vector4d> _armor_xyza_list;

    for (int i = 0; i < armor_num_; i++) {
        auto angle = angles::normalize_angle(target_state_[6] + i * 2 * CV_PI / armor_num_);
        Eigen::Vector3d xyz = h_armor_xyz(target_state_, i);
        _armor_xyza_list.push_back({ xyz[0], xyz[1], xyz[2], angle });
    }
    return _armor_xyza_list;
}
Eigen::Vector3d Target::h_armor_xyz(const Eigen::VectorXd& x, int id) const {
    auto angle = angles::normalize_angle(x[6] + id * 2 * CV_PI / armor_num_);
    auto use_l_h = (armor_num_ == 4) && (id == 1 || id == 3);

    auto r = (use_l_h) ? x[8] + x[9] : x[8];
    auto armor_x = x[0] - r * std::cos(angle);
    auto armor_y = x[2] - r * std::sin(angle);
    auto armor_z = (use_l_h) ? x[4] + x[10] : x[4];

    return { armor_x, armor_y, armor_z };
}
Eigen::Vector3d Target::h_armor_vxyz(const Eigen::VectorXd& x, int id) const {
    Eigen::Vector3d v_center(x[1], x[3], x[5]);

    auto angle = angles::normalize_angle(x[6] + id * 2 * CV_PI / armor_num_);
    auto use_l_h = (armor_num_ == 4) && (id == 1 || id == 3);

    auto r = (use_l_h) ? x[8] + x[9] : x[8];

    Eigen::Vector3d p(-r * std::cos(angle), -r * std::sin(angle), (use_l_h ? x[10] : 0.0));

    Eigen::Vector3d omega(0.0, 0.0, x[7]);

    Eigen::Vector3d v_rot = omega.cross(p);
    return v_center + v_rot;
}

void Target::predict(
    std::chrono::steady_clock::time_point t,
    Eigen::Vector3d self_v,
    bool use_lin_pre
) {
    double dt = time_utils::durationSec(last_t_, t);

    predict(dt, self_v, use_lin_pre);

    last_t_ = t;
}

void Target::predict(double dt, Eigen::Vector3d self_v, bool use_lin_pre) {
    using namespace ypdv2armor_motion_model;

    using VectorX = Eigen::Matrix<double, X_N, 1>; // 固定尺寸类型

    if (use_lin_pre) {
        // 线性预测
        Eigen::Vector3d pos = position();
        Eigen::Vector3d vel = velocity();
        pos += vel * dt;
        double yaw = target_state_[6] + target_state_[7] * dt;

        target_state_ << pos.x(), vel.x(), pos.y(), vel.y(), pos.z(), vel.z(), yaw,
            target_state_[7], target_state_[8], target_state_[9], target_state_[10];
        esekf_ypd_.setState(target_state_);
        return;
    }

    // ---------------- 非线性 ESEKF 预测 ----------------
    MotionModel mtype = (tracked_id_ == armor::ArmorNumber::OUTPOST)
        ? MotionModel::CONSTANT_ROTATION
        : MotionModel::CONSTANT_VEL_ROT;
    esekf_ypd_.setPredictFunc(Predict { dt, mtype, self_v.x(), self_v.y(), self_v.z() });

    esekf_ypd_.setUpdateQ([this, dt]() { return computeProcessNoise(dt); });

    // Predict 状态
    target_state_ = esekf_ypd_.predict();

    // ---------------- GTSAM MOTION FACTOR ----------------
    // auto key_prev = gtsam_motion_generic::XKey(step_);
    // auto key_curr = gtsam_motion_generic::XKey(step_ + 1);

    // if (!fg_estimator_.hasKey(key_prev))
    //     fg_estimator_.insertInitial(key_prev, target_state_);

    // if (!fg_estimator_.hasKey(key_curr))
    //     fg_estimator_.insertInitial(key_curr, target_state_);

    // VectorX sigmas = computeProcessNoise(dt).diagonal().cwiseSqrt(); // 固定尺寸
    // auto modelQ = gtsam_motion_generic::noiseModel::Diagonal::Sigmas(sigmas);

    // fg_estimator_.addMotionFactor(key_prev, key_curr, Predict{dt, mtype, self_v.x(), self_v.y(), self_v.z()}, modelQ);

    // step_++;  // 统一递增 step

    if (!jumped) {
        target_state_[8] = radius_pre_;
        esekf_ypd_.setState(target_state_);
    }
}

// ------------------- UPDATE -------------------
bool Target::update(const armor::Armor& armor) {
    using namespace ypdv2armor_motion_model;
    using VectorX = Eigen::Matrix<double, X_N, 1>;
    using VectorZ = Eigen::Matrix<double, Z_N, 1>;

    timestamp_ = armor.timestamp;

    // 找最匹配的 armor ID
    int id = 0;
    double min_angle_error = 1e10;
    const auto& xyza_list = getArmorPosAndYaw();
    std::vector<std::pair<Eigen::Vector4d, int>> xyza_i_list;
    for (int i = 0; i < armor_num_; i++)
        xyza_i_list.push_back({ xyza_list[i], i });

    std::sort(
        xyza_i_list.begin(),
        xyza_i_list.end(),
        [](const std::pair<Eigen::Vector4d, int>& a, const std::pair<Eigen::Vector4d, int>& b) {
            return utils::xyz2ypd(a.first.head<3>())[2] < utils::xyz2ypd(b.first.head<3>())[2];
        }
    );

    for (int i = 0; i < 3; i++) {
        const auto& xyza = xyza_i_list[i].first;
        Eigen::Vector3d ypd = utils::xyz2ypd(xyza.head<3>());
        Eigen::Vector3d ypd_target = utils::xyz2ypd(armor.target_pos);
        double angle_error =
            std::abs(angles::normalize_angle(orientationToYaw(armor.target_ori) - xyza[3]))
            + std::abs(angles::normalize_angle(ypd_target[0] - ypd[0]))
            + std::abs(angles::normalize_angle(ypd_target[1] - ypd[1]));
        if (angle_error < min_angle_error) {
            id = xyza_i_list[i].second;
            min_angle_error = angle_error;
        }
    }

    // ---------------- ESEKF 更新 ----------------
    auto p = armor.target_pos;
    double measured_yaw = orientationToYaw(armor.target_ori);
    double ypd_y = std::atan2(p.y(), p.x());
    ypd_y = last_ypd_y + angles::shortest_angular_distance(last_ypd_y, ypd_y);
    last_ypd_y = ypd_y;
    double ypd_p = std::atan2(p.z(), std::sqrt(p.x() * p.x() + p.y() * p.y()));
    double ypd_d = std::sqrt(p.x() * p.x() + p.y() * p.y() + p.z() * p.z());
    measurement_ = VectorZ(ypd_y, ypd_p, ypd_d, measured_yaw);

    esekf_ypd_.setMeasureFunc(Measure { id, armor_num_ });
    esekf_ypd_.update(measurement_);

    // ---------------- GTSAM MEASUREMENT FACTOR ----------------
    // auto key_meas = gtsam_motion_generic::XKey(step_);
    // if(!fg_estimator_.hasKey(key_meas))
    //     fg_estimator_.insertInitial(key_meas, target_state_);

    // Eigen::Matrix<double,Z_N,Z_N> R = yu_rv2(measurement_);
    // VectorZ sigmas = R.diagonal();
    // auto meas_model = gtsam_motion_generic::noiseModel::Diagonal::Sigmas(sigmas);

    // fg_estimator_.addMeasurementFactor(key_meas, measurement_, Measure{id,armor_num_}, meas_model);

    // fg_estimator_.update();
    // target_state_ = fg_estimator_.estimate(key_meas);

    if (id != 0)
        jumped = true;
    is_switch_ = (id != last_id);
    if (is_switch_)
        switch_count_++;
    last_id = id;
    update_count_++;

    return true;
}
