#include "guidance_target.hpp"

namespace auto_guidance {
GuidanceTarget::GuidanceTarget() {
    target_state_ = Eigen::VectorXd::Zero(imgbox_model::X_N);
}
GuidanceTarget::GuidanceTarget(const GreenLight& light, TargetConfig target_config) {
    target_config_ = target_config;
    auto yfv2 = imgbox_model::Predict(0.01);
    auto yhv2 = imgbox_model::Measure();
    auto yu_qv2 = [this]() { return computeProcessNoise(0.01); };

    auto yu_rv2 = [this](const Eigen::Matrix<double, imgbox_model::Z_N, 1>& z) {
        return this->computeMeasurementCovariance(z);
    };
    Eigen::DiagonalMatrix<double, imgbox_model::X_N> p0;
    p0.diagonal() << 1000, 1000, 1000, 1000, 64000, 64000, 64000, 64000;
    esekf_ = imgbox_model::BBox8ESEKF(yfv2, yhv2, yu_qv2, yu_rv2, p0);

    esekf_.setResidualFunc([this](
                               const Eigen::Matrix<double, imgbox_model::Z_N, 1>& z_pred,
                               const Eigen::Matrix<double, imgbox_model::Z_N, 1>& z
                           ) {
        Eigen::Matrix<double, imgbox_model::Z_N, 1> r = z - z_pred;
        return r;
    });
    esekf_.setIterationNum(target_config_.iter_num);
    esekf_.setInjectFunc([this](
                             const Eigen::Matrix<double, imgbox_model::X_N, 1>& delta,
                             Eigen::Matrix<double, imgbox_model::X_N, 1>& nominal
                         ) {
        for (int i = 0; i < imgbox_model::X_N; i++) {
            nominal[i] += delta[i];
        }
    });

    double cx = light.center_point.x;
    double cy = light.center_point.y;
    double w = light.box.width;
    double h = light.box.height;
    target_state_ << cx, 0, cy, 0, w, 0, h, 0;
    esekf_.setState(target_state_);
    last_t_ = light.timestamp;
    position_ = light.position;
    timestamp_ = light.timestamp;
    image_size_ = light.image_size;
    is_inited_ = true;
}
Eigen::Matrix<double, imgbox_model::Z_N, imgbox_model::Z_N>
GuidanceTarget::computeMeasurementCovariance(const Eigen::Matrix<double, imgbox_model::Z_N, 1>& z
) const {
    Eigen::Matrix<double, imgbox_model::Z_N, imgbox_model::Z_N> r;
    // clang-format off
        r <<target_config_.xy_r, 0, 0, 0,
                0, target_config_.xy_r , 0, 0,
                0, 0, target_config_.wh_r, 0,
                0, 0, 0,target_config_.wh_r;
    // clang-format on
    return r;
}
Eigen::Matrix<double, imgbox_model::X_N, imgbox_model::X_N>
GuidanceTarget::computeProcessNoise(double dt) const {
    Eigen::Matrix<double, imgbox_model::X_N, imgbox_model::X_N> q;

    double t = dt;
    double q_pp = pow(t, 4) / 4.0 * target_config_.q_xy;
    double q_pv = pow(t, 3) / 2.0 * target_config_.q_xy;
    double q_vv = pow(t, 2) * target_config_.q_xy;

    double q_ss = pow(t, 4) / 4.0 * target_config_.q_wh;
    double q_sv = pow(t, 3) / 2.0 * target_config_.q_wh;
    double q_vvs = pow(t, 2) * target_config_.q_wh;

    // clang-format off
            //      cx      vx      cy      vy      w       vw      h           vh      
            q <<    q_pp,   q_pv,   0,      0,      0,      0,      0,          0,         
                    q_pv,   q_vv,   0,      0,      0,      0,      0,          0,          
                    0,      0,      q_pp,   q_pv,   0,      0,      0,          0,         
                    0,      0,      q_pv,   q_vv,   0,      0,      0,          0,          
                    0,      0,      0,      0,      q_ss,   q_sv,   0,          0,          
                    0,      0,      0,      0,      q_sv,   q_vvs,  0,          0,          
                    0,      0,      0,      0,      0,      0,      q_ss,       q_sv,       
                    0,      0,      0,      0,      0,      0,      q_sv,       q_vvs;

    // clang-format on
    return q;
}
void GuidanceTarget::predict(std::chrono::steady_clock::time_point t) {
    double dt = wust_vl::common::utils::time_utils::durationSec(last_t_, t);

    predict(dt);

    last_t_ = t;
}
void GuidanceTarget::predict(double dt) {
    dt_ = dt;

    esekf_.setPredictFunc(imgbox_model::Predict { dt });
    auto yu_qv2 = [dt, this]() { return computeProcessNoise(dt); };

    esekf_.setUpdateQ(yu_qv2);

    target_state_ = esekf_.predict();
}

bool GuidanceTarget::update(const GreenLights& lights) {
    auto ls = lights.lights;
    timestamp_ = lights.timestamp;

    auto yu_rv2 = [this](const Eigen::Matrix<double, imgbox_model::Z_N, 1>& z) {
        return this->computeMeasurementCovariance(z);
    };
    esekf_.setUpdateR(yu_rv2);
    int best_id = -1;
    double min_error = std::numeric_limits<double>::max();
    for (int i = 0; i < ls.size(); i++) {
        double centor_error = cv::norm(ls[i].center_point - center());
        double pos_error = (ls[i].position - position_).norm();
        if (centor_error < min_error && pos_error < target_config_.max_dis_diff) {
            min_error = centor_error;
            best_id = i;
        }
    }
    if (best_id == -1) {
        return false;
    }
    measurement_ = Eigen::Vector4d(
        ls[best_id].center_point.x,
        ls[best_id].center_point.y,
        ls[best_id].box.width,
        ls[best_id].box.height
    );
    esekf_.setMeasureFunc(imgbox_model::Measure());
    target_state_ = esekf_.update(measurement_);
    position_ = ls[best_id].position;
    image_size_ = ls[best_id].image_size;
    last_t_ = timestamp_;
    return true;
}

} // namespace auto_guidance