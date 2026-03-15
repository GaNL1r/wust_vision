#pragma once

#include <Eigen/Dense>
#include <vector>
#include <yaml-cpp/node/node.h>
#include <yaml-cpp/yaml.h>

namespace wust_vision::auto_sniper {

class OffsetHelper {
public:
    using Ptr = std::shared_ptr<OffsetHelper>;
    struct OffsetPoint {
        double distance;
        double yaw;
        double pitch;
    };
    OffsetHelper(const YAML::Node& config) {
        data_.clear();
        for (auto& v: config["offset_table"]) {
            OffsetPoint p;
            p.distance = v["distance"].as<double>();
            p.yaw = v["yaw"].as<double>();
            p.pitch = v["pitch"].as<double>();

            data_.push_back(p);
        }
        order_ = config["order"].as<int>();
        yaw_base_offset = config["yaw_base_offset"].as<double>();
        pitch_base_offset = config["pitch_base_offset"].as<double>();
        fit();
    }
    static Ptr create(const YAML::Node& config) {
        return std::make_shared<OffsetHelper>(config);
    }
    void fit() {
        int n = data_.size();
        Eigen::MatrixXd A(n, order_ + 1);

        Eigen::VectorXd y_yaw(n);
        Eigen::VectorXd y_pitch(n);

        for (int i = 0; i < n; ++i) {
            double x = data_[i].distance;

            double v = 1;
            for (int j = 0; j <= order_; ++j) {
                A(i, j) = v;
                v *= x;
            }

            y_yaw(i) = data_[i].yaw;
            y_pitch(i) = data_[i].pitch;
        }

        yaw_coeff_ = A.colPivHouseholderQr().solve(y_yaw);
        pitch_coeff_ = A.colPivHouseholderQr().solve(y_pitch);
    }

    double getYawOffset(double distance) const {
        return yaw_base_offset + eval(yaw_coeff_, distance);
    }
    double getPitchOffset(double distance) const {
        return pitch_base_offset + eval(pitch_coeff_, distance);
    }

    double eval(const Eigen::VectorXd& coeff, double x) const {
        double y = 0;
        double v = 1;

        for (int i = 0; i < coeff.size(); ++i) {
            y += coeff[i] * v;
            v *= x;
        }

        return y;
    }
    std::vector<OffsetPoint> data_;

    Eigen::VectorXd yaw_coeff_;
    Eigen::VectorXd pitch_coeff_;
    double yaw_base_offset = 0;
    double pitch_base_offset = 0;
    int order_ = 2;
};

} // namespace wust_vision::auto_sniper