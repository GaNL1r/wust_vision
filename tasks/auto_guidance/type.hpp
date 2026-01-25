#pragma once
#include "Eigen/Dense"
#include "opencv2/opencv.hpp"
namespace wust_vision {
namespace auto_guidance {
    struct GreenLight {
        int id = -1;
        double score = 0.;
        cv::Rect2d box; // bounding box in pixel coordinates
        cv::Point2d center_point; // center in pixel coordinates
        double radius;
        Eigen::Vector3d position;
        std::chrono::steady_clock::time_point timestamp;
        cv::Size2d image_size;
        // PnP 估计位姿
        bool solvePnP(const cv::Mat& K, const cv::Mat& distCoeffs) {
            constexpr float half_w = 0.07;
            // 真实世界点，单位米
            std::vector<cv::Point3f> objectPoints = { { -half_w, -half_w, 0.f },
                                                      { half_w, -half_w, 0.f },
                                                      { half_w, half_w, 0.f },
                                                      { -half_w, half_w, 0.f } };

            // 像素点
            std::vector<cv::Point2f> imagePoints = {
                cv::Point2f(box.x, box.y),
                cv::Point2f(box.x + box.width, box.y),
                cv::Point2f(box.x + box.width, box.y + box.height),
                cv::Point2f(box.x, box.y + box.height)
            };

            cv::Mat rvec, tvec;

            bool ok = cv::solvePnP(
                objectPoints,
                imagePoints,
                K,
                distCoeffs,
                rvec,
                tvec,
                false,
                cv::SOLVEPNP_ITERATIVE
            );

            if (!ok)
                return false;

            // 转换到 Eigen 向量
            position = Eigen::Vector3d(tvec.at<double>(0), tvec.at<double>(1), tvec.at<double>(2));

            return true;
        }
    };

    struct GreenLights {
    public:
        std::vector<GreenLight> lights;
        std::chrono::steady_clock::time_point timestamp;
        int id;
        void drawFront(cv::Mat& img) const {
            for (const auto& light: lights) {
                cv::rectangle(img, light.box, cv::Scalar(0, 255, 255), 2);
                cv::circle(img, light.center_point, light.radius, cv::Scalar(0, 255, 0), 2);
                cv::circle(img, light.center_point, 3, cv::Scalar(255, 0, 0), -1);

                cv::putText(
                    img,
                    std::to_string(light.score),
                    light.center_point
                        + cv::Point2d(light.box.width / 2.0, -light.box.height / 2.0),
                    cv::FONT_HERSHEY_SIMPLEX,
                    0.5,
                    cv::Scalar(255, 0, 0),
                    2
                );
                cv::putText(
                    img,
                    std::to_string(light.position.norm()),
                    light.center_point + cv::Point2d(light.box.width / 2.0, light.box.height / 2.0),
                    cv::FONT_HERSHEY_SIMPLEX,
                    0.5,
                    cv::Scalar(255, 255, 255),
                    2
                );
            }
        }
        void drawBack(cv::Mat& img) const {
            for (const auto& light: lights) {
                cv::line(
                    img,
                    cv::Point(light.center_point.x, light.center_point.y),
                    cv::Point(img.cols / 2.0, light.center_point.y),
                    cv::Scalar(0, 0, 255),
                    2
                );
            }
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