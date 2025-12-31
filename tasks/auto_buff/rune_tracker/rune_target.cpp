#include "rune_target.hpp"
namespace rune {
RuneTarget::RuneTarget(
    const rune::RuneFan& fan,
    const RuneTargetConfig& target_config,
    double pre_v_roll
) {
    is_big_ = false;
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
    esekf_ypd_.setResidualFunc([](const Eigen::VectorXd& z_pred, const Eigen::VectorXd& z) {
        Eigen::VectorXd r = z - z_pred;
        r[(int)ypdrune_motion_model::Mean::YPD_Y] = angles::shortest_angular_distance(
            z_pred[(int)ypdrune_motion_model::Mean::YPD_Y],
            z[(int)ypdrune_motion_model::Mean::YPD_Y]
        ); // yaw
        r[(int)ypdrune_motion_model::Mean::ORI_YAW] = angles::shortest_angular_distance(
            z_pred[(int)ypdrune_motion_model::Mean::ORI_YAW],
            z[(int)ypdrune_motion_model::Mean::ORI_YAW]
        ); // ori_yaw
        r[(int)ypdrune_motion_model::Mean::ORI_ROLL] = angles::shortest_angular_distance(
            z_pred[(int)ypdrune_motion_model::Mean::ORI_ROLL],
            z[(int)ypdrune_motion_model::Mean::ORI_ROLL]
        ); // ori_roll
        return r;
    });
    esekf_ypd_.setIterationNum(target_config_.esekf_iter_num);
    esekf_ypd_.setInjectFunc([](const Eigen::Matrix<double, ypdrune_motion_model::X_N, 1>& delta,
                                Eigen::Matrix<double, ypdrune_motion_model::X_N, 1>& nominal) {
        for (int i = 0; i < ypdrune_motion_model::X_N; i++) {
            if (i == (int)ypdrune_motion_model::Mean::ORI_YAW
                || i == (int)ypdrune_motion_model::Mean::ORI_ROLL)
                continue;
            nominal[i] += delta[i];
        }
        nominal[(int)ypdrune_motion_model::Mean::ORI_YAW] = angles::normalize_angle(
            nominal[(int)ypdrune_motion_model::Mean::ORI_YAW]
            + delta[(int)ypdrune_motion_model::Mean::ORI_YAW]
        );
        nominal[(int)ypdrune_motion_model::Mean::ORI_ROLL] = angles::normalize_angle(
            nominal[(int)ypdrune_motion_model::Mean::ORI_ROLL]
            + delta[(int)ypdrune_motion_model::Mean::ORI_ROLL]
        );
    });

    double xc = fan.fans.front().target_pos.x();
    double yc = fan.fans.front().target_pos.y();
    double zc = fan.fans.front().target_pos.z();
    double yaw = orientationToYaw(fan.fans.front().target_ori);
    double roll = orientationToRoll(fan.fans.front().target_ori);
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
bool RuneTarget::update(const rune::RuneFan& fans) {
    timestamp_ = fans.timestamp;
    if (fans.fans.empty()) {
        return false;
    }

    update_ids.clear();
    auto matched = match(fans.fans);
    bool has_match = false;
    for (auto [id, fan]: matched) {
        measurement_ = getmean(fan);
        update_ids.push_back(id);
        auto yu_rv2 = [this](const Eigen::Matrix<double, ypdrune_motion_model::Z_N, 1>& z) {
            return this->computeMeasurementCovariance(z);
        };
        esekf_ypd_.setUpdateR(yu_rv2);
        esekf_ypd_.setMeasureFunc(ypdrune_motion_model::Measure { id });

        esekf_ypd_.update(measurement_);
        if (!is_big_)
            last_id = id;
        has_match = true;
    }
    bool no_change = false;
    for (auto id: update_ids) {
        if (id == last_id)
            no_change = true;
    }
    if (!no_change && update_ids.size() > 1)
        last_id = update_ids[0];
    if (update_ids.size() > 1)
        is_big_ = true;
    double tostart = time_utils::durationSec(start_time_, fans.timestamp);
    fitter_.update(tostart, v_roll());
    fitter_.setAngleRef(tostart, roll());
    fitter_.fitAsync();
    last_time_ = tostart;

    return has_match;
}
cv::Rect RuneTarget::expanded(
    Eigen::Matrix4d T_camera_to_odom,
    const cv::Mat& camera_intrinsic,
    const cv::Mat& camera_distortion,
    const cv::Size& image_size
) {
    if (!is_inited
        || time_utils::durationSec(timestamp_, time_utils::now()) > target_config_.lost_dt) {
        return cv::Rect(0, 0, 0, 0);
    }

    const float car_box_half = 1.0;

    static std::vector<cv::Point3f> CAR_BOX;
    CAR_BOX = { { 0, car_box_half, -car_box_half },
                { 0, -car_box_half, -car_box_half },
                { 0, -car_box_half, car_box_half },
                { 0, car_box_half, car_box_half } };

    Eigen::Matrix4d T_odom_to_camera = T_camera_to_odom.inverse();
    Eigen::Vector4d pos_odom(centerPos().x(), centerPos().y(), centerPos().z(), 1.0);
    Eigen::Vector4d pos_cam = T_odom_to_camera * pos_odom;

    if (pos_cam.z() <= 0.2) {
        return cv::Rect(0, 0, 0, 0);
    }

    cv::Mat tvec = (cv::Mat_<double>(3, 1) << pos_cam.x(), pos_cam.y(), pos_cam.z());

    Eigen::Vector3d euler;
    euler.z() = M_PI / 2.0;
    euler.y() = 0;
    euler.x() = std::atan2(pos_odom.y(), pos_odom.x());

    Eigen::Quaterniond ori = utils::eulerToQuat(euler, utils::EulerOrder::ZYX);
    auto target_ori = utils::transformOrientation(ori, T_odom_to_camera);
    Eigen::Matrix3d tf_rot = target_ori.toRotationMatrix();

    cv::Mat rot_mat =
        (cv::Mat_<double>(3, 3) << tf_rot(0, 0),
         tf_rot(0, 1),
         tf_rot(0, 2),
         tf_rot(1, 0),
         tf_rot(1, 1),
         tf_rot(1, 2),
         tf_rot(2, 0),
         tf_rot(2, 1),
         tf_rot(2, 2));

    cv::Mat rvec;
    cv::Rodrigues(rot_mat, rvec);

    std::vector<cv::Point2f> pts_2d;
    cv::projectPoints(CAR_BOX, rvec, tvec, camera_intrinsic, camera_distortion, pts_2d);

    cv::Rect rect = cv::boundingRect(pts_2d);

    cv::Rect img_rect(0, 0, image_size.width, image_size.height);
    if ((rect & img_rect).area() <= 0) {
        return cv::Rect(0, 0, 0, 0);
    }

    int cx = rect.x + rect.width / 2;
    int cy = rect.y + rect.height / 2;
    int side = std::max(rect.width, rect.height);

    cv::Rect square(cx - side / 2, cy - side / 2, side, side);
    square &= img_rect;

    return square;
}
std::vector<std::pair<int, rune::RuneFan::Simple>>
RuneTarget::match(const std::vector<rune::RuneFan::Simple>& fans) {
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
} // namespace rune
