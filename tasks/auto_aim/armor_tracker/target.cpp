#include "target.hpp"
#include "wust_vl/common/utils/math.hpp"
#include <wust_vl/common/utils/timer.hpp>
Target::Target() {
    target_state_ = Eigen::VectorXd::Zero(ypdv2armor_motion_model::X_N);
}
Target::Target(
    const armor::Armor& a,
    const TargetConfig& target_config,
    double radius,
    int armor_num,
    Eigen::DiagonalMatrix<double, ypdv2armor_motion_model::X_N> p0
):

    armor_num_(armor_num),
    radius_pre_(radius) {
    target_config_ = target_config;
    target_state_ = Eigen::VectorXd::Zero(ypdv2armor_motion_model::X_N);
    auto yfv2 = ypdv2armor_motion_model::Predict(0.005);
    auto yhv2 = ypdv2armor_motion_model::Measure(0, 4);
    auto yu_qv2 = [this]() {
        Eigen::Matrix<double, ypdv2armor_motion_model::X_N, ypdv2armor_motion_model::X_N> q;
        return q;
    };

    auto yu_rv2 = [this](const Eigen::Matrix<double, ypdv2armor_motion_model::Z_N, 1>& z) {
        Eigen::Matrix<double, ypdv2armor_motion_model::Z_N, ypdv2armor_motion_model::Z_N> r;
        return r;
    };
    esekf_ypd_ = ypdv2armor_motion_model::RobotStateESEKF(yfv2, yhv2, yu_qv2, yu_rv2, p0);

    esekf_ypd_.setResidualFunc(
        [this](
            const Eigen::Matrix<double, ypdv2armor_motion_model::Z_N, 1>& z_pred,
            const Eigen::Matrix<double, ypdv2armor_motion_model::Z_N, 1>& z
        ) {
            Eigen::Matrix<double, ypdv2armor_motion_model::Z_N, 1> r = z - z_pred;
            r[0] = angles::shortest_angular_distance(z_pred[0], z[0]); // yaw
            r[3] = angles::shortest_angular_distance(z_pred[3], z[3]); // ori_yaw
            return r;
        }
    );
    esekf_ypd_.setIterationNum(target_config_.esekf_iter_num);
    esekf_ypd_.setInjectFunc(
        [this](
            const Eigen::Matrix<double, ypdv2armor_motion_model::X_N, 1>& delta,
            Eigen::Matrix<double, ypdv2armor_motion_model::X_N, 1>& nominal
        ) {
            for (int i = 0; i < ypdv2armor_motion_model::X_N; i++) {
                if (i == 6)
                    continue;
                nominal[i] += delta[i];
            }
            nominal[6] = angles::normalize_angle(nominal[6] + delta[6]);
        }
    );

    esekf_ypd_.setNisThreshold(9.488);
    esekf_ypd_.setNeesThreshold(9.488);
    esekf_ypd_.setWindowSize(50);
    esekf_ypd_.setRecentFailRateThreshold(0.4);
    double xa = a.target_pos.x();
    double ya = a.target_pos.y();
    double za = a.target_pos.z();
    last_yaw_ = 0;
    double yaw = orientationToYaw(a.target_ori);

    target_state_ = Eigen::VectorXd::Zero(ypdv2armor_motion_model::X_N);
    double r = radius;
    double xc = xa + r * cos(yaw);
    double yc = ya + r * sin(yaw);
    double zc = za;
    target_state_ << xc, 0, yc, 0, zc, 0, yaw, 0, r, 0, 0;

    esekf_ypd_.setState(target_state_);
    tracked_id_ = a.number;
    type_ = a.type;
    last_t_ = a.timestamp;
    timestamp_ = a.timestamp;
    is_inited = true;
}
Eigen::Matrix<double, ypdv2armor_motion_model::Z_N, ypdv2armor_motion_model::Z_N>
Target::computeMeasurementCovariance(const Eigen::Matrix<double, ypdv2armor_motion_model::Z_N, 1>& z
) const {
    Eigen::Matrix<double, ypdv2armor_motion_model::Z_N, ypdv2armor_motion_model::Z_N> r;
    double delta_angle = angles::normalize_angle(z[3] - z[0]);
    double abs_delta = std::abs(delta_angle);

    // sin插值函数，小值慢、大值快
    auto sinInterp = [](double x, double x0, double x1, double y0, double y1) -> double {
        double t = (x - x0) / (x1 - x0);
        if (t < 0)
            t = 0;
        if (t > 1)
            t = 1;
        double s = std::sin(t * M_PI / 2.0);
        return y0 + s * (y1 - y0);
    };
    // clang-format off
        r <<target_config_.yp_r, 0, 0, 0,
                0, target_config_.yp_r , 0, 0,
                0, 0, sinInterp(abs_delta, 0.0, M_PI/2.0, target_config_.dis_r_front, target_config_.dis_r_side)+z[2]*z[2]*target_config_.dis2_r_ratio, 0,
                0, 0, 0,log(std::abs(z[2]) + 1) *target_config_.yaw_r_log_ratio + sinInterp(M_PI/2.0-abs_delta, 0.0, M_PI/2.0, target_config_.yaw_r_base_side, target_config_.yaw_r_base_front);
    // clang-format on
    return r;
}
Eigen::Matrix<double, ypdv2armor_motion_model::X_N, ypdv2armor_motion_model::X_N>
Target::computeProcessNoise(double dt) const {
    Eigen::Matrix<double, ypdv2armor_motion_model::X_N, ypdv2armor_motion_model::X_N> q;
    double v1, v2;
    double q_l, q_h;
    if (tracked_id_ == armor::ArmorNumber::OUTPOST) {
        v1 = target_config_.qxyz_output; // 前哨站加速度方差
        v2 = target_config_.qyaw_output; // 前哨站角加速度方差
        q_l = target_config_.q_outpost_dz;
        q_h = target_config_.q_outpost_dz;
    } else {
        v1 = target_config_.qxyz_common; // 加速度方差
        v2 = target_config_.qyaw_common; // 角加速度方差
        q_l = target_config_.q_l;
        q_h = target_config_.q_h;
    }
    double t = dt;
    double q_x_x = pow(t, 4) / 4 * v1, q_x_vx = pow(t, 3) / 2 * v1, q_vx_vx = pow(t, 2) * v1;
    double q_y_y = pow(t, 4) / 4 * v1, q_y_vy = pow(t, 3) / 2 * v1, q_vy_vy = pow(t, 2) * v1;
    double q_z_z = pow(t, 4) / 4 * v1, q_z_vz = pow(t, 3) / 2 * v1, q_vz_vz = pow(t, 2) * v1;
    double q_yaw_yaw = pow(t, 4) / 4 * v2, q_yaw_vyaw = pow(t, 3) / 2 * v2,
           q_vyaw_vyaw = pow(t, 2) * v2;
    double q_r = target_config_.q_r;

    // clang-format off
            //      xc      v_xc    yc      v_yc    zc      v_zc    yaw         v_yaw       r       l   h
            q <<    q_x_x,  q_x_vx, 0,      0,      0,      0,      0,          0,          0,      0,  0,
                    q_x_vx, q_vx_vx,0,      0,      0,      0,      0,          0,          0,      0,  0,
                    0,      0,      q_y_y,  q_y_vy, 0,      0,      0,          0,          0,      0,  0,
                    0,      0,      q_y_vy, q_vy_vy,0,      0,      0,          0,          0,      0,  0,
                    0,      0,      0,      0,      q_z_z,  q_z_vz, 0,          0,          0,      0,  0,
                    0,      0,      0,      0,      q_z_vz, q_vz_vz,0,          0,          0,      0,  0,
                    0,      0,      0,      0,      0,      0,      q_yaw_yaw,  q_yaw_vyaw, 0,      0,  0,
                    0,      0,      0,      0,      0,      0,      q_yaw_vyaw, q_vyaw_vyaw,0,      0,  0,
                    0,      0,      0,      0,      0,      0,      0,          0,          q_r,      0,  0,
                    0,      0,      0,      0,      0,      0,      0,          0,          0,      q_l,  0,
                    0,      0,      0,      0,      0,      0,      0,          0,          0,      0,  q_h;
    // clang-format on
    return q;
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
    dt_ = dt;
    if (use_lin_pre) {
        double r = target_state_[8];
        double l = target_state_[9];
        double h = target_state_[10];
        auto pos = position();
        auto vel = velocity();
        pos += vel * dt;
        double yaw = target_state_[6];
        double v_yaw = target_state_[7];
        yaw += v_yaw * dt;
        target_state_ = Eigen::VectorXd::Zero(ypdv2armor_motion_model::X_N);
        target_state_ << pos.x(), vel.x(), pos.y(), vel.y(), pos.z(), vel.z(), yaw, v_yaw, r, l, h;
        esekf_ypd_.setState(target_state_);
    } else {
        if (tracked_id_ == armor::ArmorNumber::OUTPOST) {
            esekf_ypd_.setPredictFunc(ypdv2armor_motion_model::Predict {
                dt,
                ypdv2armor_motion_model::MotionModel::CONSTANT_ROTATION,
                self_v.x(),
                self_v.y(),
                self_v.z() });
        } else {
            esekf_ypd_.setPredictFunc(ypdv2armor_motion_model::Predict {
                dt,
                ypdv2armor_motion_model::MotionModel::CONSTANT_VEL_ROT,
                self_v.x(),
                self_v.y(),
                self_v.z() });
        }
        auto yu_qv2 = [dt, this]() { return computeProcessNoise(dt); };

        esekf_ypd_.setUpdateQ(yu_qv2);

        target_state_ = esekf_ypd_.predict();
    }
    if (!jumped) {
        target_state_[8] = radius_pre_;
        target_state_[9] = 0.0;
        target_state_[10] = 0.0;
        esekf_ypd_.setState(target_state_);
    }
    if (position().norm() < 0.5) {
        is_tracking = false;
    }
    if (tracked_id_ == armor::ArmorNumber::OUTPOST) {
        if (target_state_[8] < 0.25) {
            target_state_[8] = 0.25;
        }
        if (target_state_[8] > 0.35) {
            target_state_[8] = 0.35;
        }
        constexpr double outpost_v_yaw_err=0.1;
        if (target_state_[7] > armor::outpost_v_yaw + outpost_v_yaw_err) {
            target_state_[7] = armor::outpost_v_yaw + outpost_v_yaw_err;
        }
        if (target_state_[7] < armor::outpost_v_yaw - outpost_v_yaw_err) {
            target_state_[7] = armor::outpost_v_yaw - outpost_v_yaw_err;
        }
        esekf_ypd_.setState(target_state_);
    }
}

bool Target::update(const armor::Armor& a) {
    auto armor = a;
    timestamp_ = armor.timestamp;
    auto yu_rv2 = [this](const Eigen::Matrix<double, ypdv2armor_motion_model::Z_N, 1>& z) {
        return this->computeMeasurementCovariance(z);
    };
    esekf_ypd_.setUpdateR(yu_rv2);
    int id;
    auto min_angle_error = 1e10;
    const std::vector<Eigen::Vector4d>& xyza_list = getArmorPosAndYaw();

    std::vector<std::pair<Eigen::Vector4d, int>> xyza_i_list;
    for (int i = 0; i < armor_num_; i++) {
        xyza_i_list.push_back({ xyza_list[i], i });
    }

    std::sort(
        xyza_i_list.begin(),
        xyza_i_list.end(),
        [](const std::pair<Eigen::Vector4d, int>& a, const std::pair<Eigen::Vector4d, int>& b) {
            Eigen::Vector3d ypd1 = utils::xyz2ypd(a.first.head(3));
            Eigen::Vector3d ypd2 = utils::xyz2ypd(b.first.head(3));
            return ypd1[2] < ypd2[2];
        }
    );

    // 取前3个distance最小的装甲板
    if (armor_num_ > 3) {
        for (int i = 0; i < 3; i++) {
            const auto& xyza = xyza_i_list[i].first;
            Eigen::Vector3d ypd = utils::xyz2ypd(xyza.head(3));
            auto angle_error =
                std::abs(angles::normalize_angle(orientationToYaw(armor.target_ori) - xyza[3]))
                + std::abs(angles::normalize_angle(utils::xyz2ypd(armor.target_pos)[0] - ypd[0]))
                + std::abs(angles::normalize_angle(utils::xyz2ypd(armor.target_pos)[1] - ypd[1]));
            if (std::abs(angle_error) < std::abs(min_angle_error)) {
                id = xyza_i_list[i].second;
                min_angle_error = angle_error;
            }
        }
    } else {
        for (int i = 0; i < 3; i++) {
            const auto& xyza = xyza_i_list[i].first;
            Eigen::Vector3d ypd = utils::xyz2ypd(xyza.head(3));
            auto angle_error =
                std::abs(angles::normalize_angle(orientationToYaw(armor.target_ori) - xyza[3]))+ std::abs(angles::normalize_angle(utils::xyz2ypd(armor.target_pos)[0] - ypd[0]))
                + std::abs(angles::normalize_angle(utils::xyz2ypd(armor.target_pos)[1] - ypd[1]));
            if (std::abs(angle_error) < std::abs(min_angle_error)) {
                id = xyza_i_list[i].second;
                min_angle_error = angle_error;
            }
        }
    }

    auto p = armor.target_pos;
    double measured_yaw = orientationToYaw(armor.target_ori);

    double ypd_y = std::atan2(p.y(), p.x());
    ypd_y = this->last_ypd_y + angles::shortest_angular_distance(this->last_ypd_y, ypd_y);
    this->last_ypd_y = ypd_y;
    double ypd_p = std::atan2(p.z(), std::sqrt(p.x() * p.x() + p.y() * p.y()));
    double ypd_d = std::sqrt(p.x() * p.x() + p.y() * p.y() + p.z() * p.z());
    measurement_ = Eigen::Vector4d(ypd_y, ypd_p, ypd_d, measured_yaw);
    if (tracked_id_ != armor::ArmorNumber::OUTPOST) {
        auto tmp_esekf = esekf_ypd_;
        tmp_esekf.setMeasureFunc(ypdv2armor_motion_model::Measure { id, armor_num_ });
        auto tmp_state = tmp_esekf.update(measurement_);
        if (diverged(tmp_state)) {
            WUST_WARN("target") << "This update make diverged skip!!";
            return false;
        }
    }

    if (id != 0)
        jumped = true;

    if (id != last_id) {
        is_switch_ = true;
    } else {
        is_switch_ = false;
    }

    if (is_switch_)
        switch_count_++;

    last_id = id;
    update_count_++;

    esekf_ypd_.setMeasureFunc(ypdv2armor_motion_model::Measure { id, armor_num_ });

    target_state_ = esekf_ypd_.update(measurement_);

    return true;
}