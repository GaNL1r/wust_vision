#include "target.hpp"
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
    target_config_(target_config),
    armor_num_(armor_num) {
    target_state_ = Eigen::VectorXd::Zero(ypdv2armor_motion_model::X_N);
    auto yfv2 = ypdv2armor_motion_model::Predict(0.005);
    auto yhv2 = ypdv2armor_motion_model::Measure(0, 4);
    auto yu_qv2 = [this]() {
        Eigen::Matrix<double, ypdv2armor_motion_model::X_N, ypdv2armor_motion_model::X_N> q;
        return q;
    };
    auto yu_rv2 = [this](const Eigen::Matrix<double, ypdv2armor_motion_model::Z_N, 1>& z) {
        Eigen::Matrix<double, ypdv2armor_motion_model::Z_N, ypdv2armor_motion_model::Z_N> r;
        auto delta_angle = angles::normalize_angle(z[3] - z[0]);
        // clang-format off
        r <<4e-3, 0, 0, 0,
                0, 4e-3 , 0, 0,
                0, 0, log(std::abs(z[2]) + 1) / 200 + 9e-2, 0,//pnp得到的distance的误差与distance的平方正相关
                0, 0, 0, log(std::abs(delta_angle) + 1) + 1;//相机系下yaw正对误差大
        // clang-format on
        return r;
    };
    esekf_ypd_ = ypdv2armor_motion_model::RobotStateESEKF(yfv2, yhv2, yu_qv2, yu_rv2, p0);
    esekf_ypd_.setAngleDims({ 0, 3 });
    esekf_ypd_.setIterationNum(2);
    esekf_ypd_.setInjectFunc([](const Eigen::Matrix<double, ypdv2armor_motion_model::X_N, 1>& delta,
                                Eigen::Matrix<double, ypdv2armor_motion_model::X_N, 1>& nominal) {
        for (int i = 0; i < ypdv2armor_motion_model::X_N; i++) {
            if (i == 6)
                continue;
            nominal[i] += delta[i];
        }
        nominal[6] = angles::normalize_angle(nominal[6] + delta[6]);
    });
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
void Target::predict(std::chrono::steady_clock::time_point t, Eigen::Vector3d self_v) {
    double dt = time_utils::durationSec(last_t_, t);
    predict(dt, self_v);
    last_t_ = t;
}
void Target::predict(double dt, const Eigen::Vector3d& self_v) {
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
    auto yu_qv2 = [dt, this]() {
        Eigen::Matrix<double, ypdv2armor_motion_model::X_N, ypdv2armor_motion_model::X_N> q;
        double v1, v2;
        if (tracked_id_ == armor::ArmorNumber::OUTPOST) {
            v1 = target_config_.qxyz_output; // 前哨站加速度方差
            v2 = target_config_.qyaw_output; // 前哨站角加速度方差
        } else {
            v1 = target_config_.qxyz_common; // 加速度方差
            v2 = target_config_.qyaw_common; // 角加速度方差
        }
        double t = dt;
        double q_x_x = pow(t, 4) / 4 * v1, q_x_vx = pow(t, 3) / 2 * v1, q_vx_vx = pow(t, 2) * v1;
        double q_y_y = pow(t, 4) / 4 * v1, q_y_vy = pow(t, 3) / 2 * v1, q_vy_vy = pow(t, 2) * v1;
        double q_z_z = pow(t, 4) / 4 * v1, q_z_vz = pow(t, 3) / 2 * v1, q_vz_vz = pow(t, 2) * v1;
        double q_yaw_yaw = pow(t, 4) / 4 * v2, q_yaw_vyaw = pow(t, 3) / 2 * v2,
               q_vyaw_vyaw = pow(t, 2) * v2;
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
                0,      0,      0,      0,      0,      0,      0,          0,          0,      0,  0,
                0,      0,      0,      0,      0,      0,      0,          0,          0,      0,  0,
                0,      0,      0,      0,      0,      0,      0,          0,          0,      0,  0;
        // clang-format on
        return q;
    };

    esekf_ypd_.setUpdateQ(yu_qv2);
    Eigen::VectorXd x2 = esekf_ypd_.predict();

    target_state_ = esekf_ypd_.predict();
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

void Target::update(const armor::Armor& armor) {
    timestamp_ = armor.timestamp;
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
    for (int i = 0; i < 3; i++) {
        const auto& xyza = xyza_i_list[i].first;
        Eigen::Vector3d ypd = utils::xyz2ypd(xyza.head(3));
        auto angle_error =
            std::abs(angles::normalize_angle(orientationToYaw(armor.target_ori) - xyza[3]))
            + std::abs(angles::normalize_angle(orientationToYaw(armor.target_ori) - ypd[0]));

        if (std::abs(angle_error) < std::abs(min_angle_error)) {
            id = xyza_i_list[i].second;
            min_angle_error = angle_error;
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
    auto p = armor.target_pos;
    double measured_yaw = orientationToYaw(armor.target_ori);

    double ypd_y = std::atan2(p.y(), p.x());
    ypd_y = this->last_ypd_y + angles::shortest_angular_distance(this->last_ypd_y, ypd_y);
    this->last_ypd_y = ypd_y;
    double ypd_p = std::atan2(p.z(), std::sqrt(p.x() * p.x() + p.y() * p.y()));
    double ypd_d = std::sqrt(p.x() * p.x() + p.y() * p.y() + p.z() * p.z());
    measurement_ = Eigen::Vector4d(ypd_y, ypd_p, ypd_d, measured_yaw);

    esekf_ypd_.update(measurement_);
}