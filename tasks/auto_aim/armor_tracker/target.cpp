#include "target.hpp"
namespace wust_vision {
namespace auto_aim {
    Target::Target() {
        target_state_.x = Eigen::VectorXd::Zero(MModel::X_N);
    }
    Target::Target(const Armor& a, TargetConfig::Ptr target_config) {
        Eigen::DiagonalMatrix<double, ypdv2armor_motion_model::X_N> p0;
        if (a.number == ArmorNumber::OUTPOST) {
            p0.diagonal() << 1, 64, 1, 64, 1, 81, 0.4, 100, 1e-4, 0.1, 0.1;
            armor_num_ = 3;
            radius_pre_ = 0.2765;
        } else if (a.number == ArmorNumber::BASE) {
            p0.diagonal() << 1, 64, 1, 64, 1, 64, 0.4, 100, 1e-4, 0, 0;
            armor_num_ = 3;
            radius_pre_ = 0.3205;
        } else {
            p0.diagonal() << 1, 64, 1, 64, 1, 64, 0.4, 100, 1, 1, 1;
            armor_num_ = 4;
            radius_pre_ = 0.2;
        }
        target_config_ = target_config;
        const auto yfv2 = MModel::Predict(0.005);
        ctx_.armor_num = armor_num_;
        ctx_.id = 0;
        const auto yhv2 = MModel::Measure(ctx_);
        const auto yu_qv2 = [this]() {
            Eigen::Matrix<double, MModel::X_N, MModel::X_N> q;
            return q;
        };

        const auto yu_rv2 = [this](const Eigen::Matrix<double, MModel::Z_N, 1>& z) {
            Eigen::Matrix<double, MModel::Z_N, MModel::Z_N> r;
            return r;
        };
        esekf_ypd_ = MModel::RobotStateESEKF(yfv2, yhv2, yu_qv2, yu_rv2, p0);

        esekf_ypd_.setResidualFunc([this](
                                       const Eigen::Matrix<double, MModel::Z_N, 1>& z_pred,
                                       const Eigen::Matrix<double, MModel::Z_N, 1>& z
                                   ) {
            Eigen::Matrix<double, MModel::Z_N, 1> r = z - z_pred;
            r[0] = angles::shortest_angular_distance(
                z_pred[(int)MModel::MeasureID::YPD_Y],
                z[(int)MModel::MeasureID::YPD_Y]
            ); // yaw
            r[3] = angles::shortest_angular_distance(
                z_pred[(int)MModel::MeasureID::ORI_YAW],
                z[(int)MModel::MeasureID::ORI_YAW]
            ); // ori_yaw
            return r;
        });
        esekf_ypd_.setIterationNum(target_config_->esekf_iter_num_param.get());
        esekf_ypd_.setInjectFunc([this](
                                     const Eigen::Matrix<double, MModel::X_N, 1>& delta,
                                     Eigen::Matrix<double, MModel::X_N, 1>& nominal
                                 ) {
            for (int i = 0; i < MModel::X_N; i++) {
                if (i == (int)MModel::StateID::YAW)
                    continue;
                nominal[i] += delta[i];
            }
            nominal[(int)MModel::StateID::YAW] = angles::normalize_angle(
                nominal[(int)MModel::StateID::YAW] + delta[(int)MModel::StateID::YAW]
            );
        });

        const double xa = a.target_pos.x();
        const double ya = a.target_pos.y();
        const double za = a.target_pos.z();
        last_yaw_ = 0;
        const double yaw = orientationToYaw(a.target_ori);

        target_state_.x = Eigen::VectorXd::Zero(MModel::X_N);
        const double r = radius_pre_;
        const double xc = xa + r * cos(yaw);
        const double yc = ya + r * sin(yaw);
        const double zc = za;
        target_state_.x << xc, 0, yc, 0, zc, 0, yaw, 0, r, 0, 0;
        esekf_ypd_.setState(target_state_.x);
        tracked_id_ = a.number;
        type_ = a.type;
        last_t_ = a.timestamp;
        timestamp_ = a.timestamp;
        is_inited = true;
    }
    Eigen::Matrix<double, MModel::Z_N, MModel::Z_N>
    Target::computeMeasurementCovariance(const Eigen::Matrix<double, MModel::Z_N, 1>& z
    ) const noexcept {
        Eigen::Matrix<double, MModel::Z_N, MModel::Z_N> r;
        const double delta_angle = angles::normalize_angle(z[3] - z[0]);
        const double abs_delta = std::abs(delta_angle);

        // sin插值函数，小值慢、大值快
        const auto sinInterp = [](double x, double x0, double x1, double y0, double y1) -> double {
            double t = (x - x0) / (x1 - x0);
            if (t < 0)
                t = 0;
            if (t > 1)
                t = 1;
            double s = std::sin(t * M_PI / 2.0);
            return y0 + s * (y1 - y0);
        };
        // clang-format off
        r <<target_config_->yp_r_param.get(), 0, 0, 0,
                0, target_config_->yp_r_param.get() , 0, 0,
                0, 0, sinInterp(abs_delta, 0.0, M_PI/2.0, target_config_->dis_r_front_param.get(), target_config_->dis_r_side_param.get())+z[2]*z[2]*target_config_->dis2_r_ratio_param.get(), 0,
                0, 0, 0,log(std::abs(z[2]) + 1) *target_config_->yaw_r_log_ratio_param.get() + sinInterp(M_PI/2.0-abs_delta, 0.0, M_PI/2.0, target_config_->yaw_r_base_side_param.get(), target_config_->yaw_r_base_front_param.get());
        // clang-format on
        return r;
    }
    Eigen::Matrix<double, MModel::X_N, MModel::X_N> Target::computeProcessNoise(double dt
    ) const noexcept {
        Eigen::Matrix<double, MModel::X_N, MModel::X_N> q;
        Eigen::Vector3d q_xyz;
        double q_yaw;
        double q_l, q_h;
        if (tracked_id_ == ArmorNumber::OUTPOST) {
            q_xyz = target_config_->qxyz_output; // 前哨站加速度方差
            q_yaw = target_config_->qyaw_output_param.get(); // 前哨站角加速度方差
            q_l = target_config_->q_outpost_dz_param.get();
            q_h = target_config_->q_outpost_dz_param.get();
        } else {
            q_xyz = target_config_->qxyz_common; // 加速度方差
            q_yaw = target_config_->qyaw_common_param.get(); // 角加速度方差
            q_l = target_config_->q_l_param.get();
            q_h = target_config_->q_h_param.get();
        }
        const double t = dt;
        const double q_x_x = pow(t, 4) / 4 * q_xyz.x(), q_x_vx = pow(t, 3) / 2 * q_xyz.x(),
                     q_vx_vx = pow(t, 2) * q_xyz.x();
        const double q_y_y = pow(t, 4) / 4 * q_xyz.y(), q_y_vy = pow(t, 3) / 2 * q_xyz.y(),
                     q_vy_vy = pow(t, 2) * q_xyz.y();
        const double q_z_z = pow(t, 4) / 4 * q_xyz.z(), q_z_vz = pow(t, 3) / 2 * q_xyz.z(),
                     q_vz_vz = pow(t, 2) * q_xyz.z();
        const double q_yaw_yaw = pow(t, 4) / 4 * q_yaw, q_yaw_vyaw = pow(t, 3) / 2 * q_yaw,
                     q_vyaw_vyaw = pow(t, 2) * q_yaw;
        const double q_r = target_config_->q_r_param.get();

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
    MModel::Predict Target::getPredictFunc(double dt) const noexcept {
        MModel::Predict predict_func;
        if (tracked_id_ == ArmorNumber::OUTPOST) {
            predict_func = MModel::Predict {
                dt,
                MModel::MotionModel::CONSTANT_ROTATION,
            };
        } else {
            predict_func = MModel::Predict {
                dt,
                MModel::MotionModel::CONSTANT_VEL_ROT,
            };
        }
        return predict_func;
    }
    void Target::predict(std::chrono::steady_clock::time_point t) noexcept {
        const double dt = wust_vl::common::utils::time_utils::durationSec(last_t_, t);

        predict(dt);

        last_t_ = t;
    }
    void Target::predict(double dt) noexcept {
        MModel::Predict predict_func = getPredictFunc(dt);

        esekf_ypd_.setPredictFunc(predict_func);
        const auto yu_qv2 = [dt, this]() { return computeProcessNoise(dt); };

        esekf_ypd_.setUpdateQ(yu_qv2);

        target_state_.x = esekf_ypd_.predict();
        if (target_state_.pos().norm() < 0.5) {
            is_tracking = false;
        }
    }
    void Target::predictSimple(std::chrono::steady_clock::time_point t) noexcept {
        const double dt = wust_vl::common::utils::time_utils::durationSec(last_t_, t);

        predictSimple(dt);

        last_t_ = t;
    }
    void Target::predictSimple(double dt) noexcept {
        MModel::Predict predict_func = getPredictFunc(dt);
        predict_func.f(target_state_.x, target_state_.x);
        if (target_state_.pos().norm() < 0.5) {
            is_tracking = false;
        }
    }
    bool Target::update(const std::pair<int, Armor>& a) noexcept {
        const auto armor = a.second;
        const auto id = a.first;
        const auto yu_rv2 = [this](const Eigen::Matrix<double, MModel::Z_N, 1>& z) {
            return this->computeMeasurementCovariance(z);
        };
        esekf_ypd_.setUpdateR(yu_rv2);
        measurement_ = getMeasure(armor);

        if (id != 0)
            jumped = true;

        ctx_.id = id;
        esekf_ypd_.setMeasureFunc(MModel::Measure { ctx_ });

        target_state_.x = esekf_ypd_.update(measurement_);
        timestamp_ = armor.timestamp;
        last_t_ = timestamp_;
        return true;
    }
    cv::Rect Target::expanded(
        Eigen::Matrix4d T_camera_to_odom,
        const cv::Mat& camera_intrinsic,
        const cv::Mat& camera_distortion,
        const cv::Size& image_size
    ) const noexcept {
        const double dt = wust_vl::common::utils::time_utils::durationSec(
            timestamp_,
            wust_vl::common::utils::time_utils::now()
        );
        if (!is_inited || dt > target_config_->lost_time_thres_param.get()) {
            return cv::Rect(0, 0, 0, 0);
        }

        const float car_box_half =
            std::max(target_state_.r(), target_state_.r() + target_state_.l()) + 0.15;

        static std::vector<cv::Point3f> CAR_BOX;
        CAR_BOX = { { 0, car_box_half, -car_box_half },
                    { 0, -car_box_half, -car_box_half },
                    { 0, -car_box_half, car_box_half },
                    { 0, car_box_half, car_box_half } };

        const Eigen::Matrix4d T_odom_to_camera = T_camera_to_odom.inverse();
        const Eigen::Vector4d
            pos_odom(target_state_.cx(), target_state_.cy(), target_state_.cz(), 1.0);
        const Eigen::Vector4d pos_cam = T_odom_to_camera * pos_odom;

        if (pos_cam.z() <= 0.2) {
            return cv::Rect(0, 0, 0, 0);
        }

        const cv::Mat tvec = (cv::Mat_<double>(3, 1) << pos_cam.x(), pos_cam.y(), pos_cam.z());

        Eigen::Vector3d euler;
        euler.z() = M_PI / 2.0;
        euler.y() = 0;
        euler.x() = std::atan2(pos_odom.y(), pos_odom.x());

        const Eigen::Quaterniond ori = utils::eulerToQuat(euler, utils::EulerOrder::ZYX);
        const auto target_ori = utils::transformOrientation(ori, T_odom_to_camera);
        const Eigen::Matrix3d tf_rot = target_ori.toRotationMatrix();

        const cv::Mat rot_mat =
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

        const cv::Rect rect = cv::boundingRect(pts_2d);

        const cv::Rect img_rect(0, 0, image_size.width, image_size.height);
        if ((rect & img_rect).area() <= 0) {
            return cv::Rect(0, 0, 0, 0);
        }

        const int base_side = std::max(rect.width, rect.height);
        const int max_side = std::max(image_size.width, image_size.height);

        const double lost_dt = target_config_->lost_time_thres_param.get();
        const double dt_clamped = std::max(0.0, std::min(dt, lost_dt));

        int side = static_cast<int>(base_side + (max_side - base_side) * (dt_clamped / lost_dt));

        if (dt >= lost_dt) {
            side = max_side;
        }

        const int cx = rect.x + rect.width / 2;
        const int cy = rect.y + rect.height / 2;
        cv::Rect square(cx - side / 2, cy - side / 2, side, side);

        square &= img_rect;

        return square;
    }

    std::vector<std::pair<int, Armor>> Target::match(const std::vector<Armor>& armors) noexcept {
        std::vector<std::pair<int, Armor>> result;
        const int n_obs = static_cast<int>(armors.size());
        const int armors_num = armor_num_;
        const double GATE = target_config_->match_gate_param.get();
        const double max_cost = 1e9;
        std::vector<std::vector<double>> cost(n_obs, std::vector<double>(armors_num, max_cost + 1));
        std::vector<MModel::VecZ> meas_list(n_obs);
        for (int j = 0; j < n_obs; ++j) {
            meas_list[j] = getMeasure(armors[j]);
        }
        for (int j = 0; j < n_obs; ++j) {
            for (int id = 0; id < armors_num; ++id) {
                MModel::Measure::MeasureCtx tmp_ctx(id, armors_num);
                MModel::Measure measure(tmp_ctx);
                MModel::VecZ z_pred;
                measure.h(target_state_.x, z_pred);

                MModel::VecZ nu = meas_list[j] - z_pred;
                nu[(int)MModel::MeasureID::YPD_Y] =
                    angles::normalize_angle(nu[(int)MModel::MeasureID::YPD_Y]);
                nu[(int)MModel::MeasureID::YPD_P] =
                    angles::normalize_angle(nu[(int)MModel::MeasureID::YPD_P]);
                nu[(int)MModel::MeasureID::ORI_YAW] =
                    angles::normalize_angle(nu[(int)MModel::MeasureID::ORI_YAW]);
                auto R = computeMeasurementCovariance(z_pred);
                double d2 = nu.transpose() * R.ldlt().solve(nu);

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
            result.push_back(std::make_pair(best_id, armors[best_j]));
        }
        return result;
    }
} // namespace auto_aim
} // namespace wust_vision