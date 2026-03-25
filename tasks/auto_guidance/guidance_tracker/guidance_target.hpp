#pragma once
#include "motion_models/imgbox_model.hpp"
#include "tasks/auto_guidance/type.hpp"
#include <wust_vl/common/utils/timer.hpp>
#include <yaml-cpp/yaml.h>
namespace wust_vision {
namespace auto_guidance {
    struct TargetConfig {
        void load(const YAML::Node& config) {
            xy_r = config["xy_r"].as<double>();
            wh_r = config["wh_r"].as<double>();
            q_xy = config["q_xy"].as<double>();
            q_wh = config["q_wh"].as<double>();
            iter_num = config["iter_num"].as<int>();
            max_dis_diff = config["max_dis_diff"].as<double>();
        }
        double xy_r = 0.05;
        double wh_r = 0.05;
        double q_xy = 10;
        double q_wh = 10;
        int iter_num = 2;
        double max_dis_diff = 2.0;
    };
    class GuidanceTarget {
    public:
        GuidanceTarget();
        GuidanceTarget(const GreenLight& light, TargetConfig target_config);
        GuidanceTarget& operator=(const GuidanceTarget&) = default;
        bool update(const GreenLights& lights);

        void predict(std::chrono::steady_clock::time_point t);
        void predict(double dt);
        Eigen::Matrix<double, imgbox_model::Z_N, imgbox_model::Z_N>
        computeMeasurementCovariance(const Eigen::Matrix<double, imgbox_model::Z_N, 1>& z) const;
        Eigen::Matrix<double, imgbox_model::X_N, imgbox_model::X_N> computeProcessNoise(double dt
        ) const;
        std::chrono::steady_clock::time_point last_t_;
        std::chrono::steady_clock::time_point timestamp_;
        double dt_;
        cv::Size2d image_size_;
        imgbox_model::BBox8ESEKF esekf_;
        Eigen::Matrix<double, imgbox_model::Z_N, 1> measurement_ =
            Eigen::Matrix<double, imgbox_model::Z_N, 1>::Zero();
        Eigen::Matrix<double, imgbox_model::X_N, 1> target_state_ =
            Eigen::Matrix<double, imgbox_model::X_N, 1>::Zero();
        Eigen::Vector3d position_;
        TargetConfig target_config_;
        bool is_inited_ = false;
        bool is_tracking_ = false;
        bool checkappear() {
            return is_tracking_
                && wust_vl::common::utils::time_utils::durationSec(
                       timestamp_,
                       wust_vl::common::utils::time_utils::now()
                   )
                < 3.0;
        }
        cv::Point2d center() const {
            return cv::Point2d(target_state_(0), target_state_(2));
        }
        cv::Rect2d box() const {
            return cv::Rect2d(
                target_state_(0) - target_state_(4) / 2,
                target_state_(2) - target_state_(6) / 2,
                target_state_(4),
                target_state_(6)
            );
        }
        void draw(cv::Mat& img) const {
            cv::rectangle(img, box(), cv::Scalar(255, 50, 0), 2);
            cv::circle(img, center(), 3, cv::Scalar(255, 255, 255), -1);
            cv::line(
                img,
                cv::Point(center().x, center().y),
                cv::Point(img.cols / 2.0, center().y),
                cv::Scalar(0, 0, 255),
                2
            );
            cv::line(
                img,
                cv::Point2f(img.cols / 2.0, 0),
                cv::Point2f(img.cols / 2.0, img.rows),
                cv::Scalar(255, 255, 255),
                2
            );
        }
    };
} // namespace auto_guidance
} // namespace wust_vision