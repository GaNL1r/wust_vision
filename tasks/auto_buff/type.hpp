#pragma once
#include "opencv2/opencv.hpp"
#include "tasks/type_common.hpp"
#include "tasks/utils.hpp"
#include <fmt/core.h>
#include <numeric>
#include <string>

namespace rune {

struct RuneFan {
    bool is_valid = false;
    int id;
    bool is_big = false;
    std::chrono::steady_clock::time_point timestamp;
    std::vector<cv::Point2f> points2d;
    std::vector<cv::Point3f> points3d;
    Eigen::Vector3d target_pos;
    Eigen::Quaterniond target_ori;

    std::vector<cv::Point2f> landmarks() const {
        return points2d;
    }
    void drawLandmarks(cv::Mat& image) const {
        std::vector<cv::Point2f> lm = landmarks();
        for (size_t i = 0; i < lm.size(); i++) {
            cv::circle(image, lm[i], 3, cv::Scalar(255, 255, 255), -1);
            if (i == 0) {
                cv::putText(
                    image,
                    "R",
                    lm[i],
                    cv::FONT_HERSHEY_SIMPLEX,
                    1.5,
                    cv::Scalar(40, 255, 40),
                    2
                );
            } else {
                cv::putText(
                    image,
                    std::to_string(i),
                    lm[i],
                    cv::FONT_HERSHEY_SIMPLEX,
                    0.5,
                    cv::Scalar(255, 255, 255),
                    2
                );
            }
        }
    }

    std::vector<cv::Point3f> getObjs() const {
        return points3d;
    }
};
static std::vector<cv::Point3f> FAN_BLOCK = {
    { -0.05f, -0.20f, -0.15f }, // 0: 左下前
    { 0.05f, -0.20f, -0.15f }, // 1: 右下前
    { 0.05f, 0.20f, -0.15f }, // 2: 右上前
    { -0.05f, 0.20f, -0.15f }, // 3: 左上前
    { -0.05f, -0.20f, 0.15f }, // 4: 左下后
    { 0.05f, -0.20f, 0.15f }, // 5: 右下后
    { 0.05f, 0.20f, 0.15f }, // 6: 右上后
    { -0.05f, 0.20f, 0.15f } // 7: 左上后
};

struct PowerRune {
    bool is_valid = false;
    struct Pose {
        Eigen::Vector3d pos;
        Eigen::Quaterniond ori;
        void tf(Eigen::Matrix4d T_camera_to_odom) {
            Eigen::Vector4d pos_camera(pos.x(), pos.y(), pos.z(), 1.0);
            Eigen::Vector4d pos_odom = T_camera_to_odom * pos_camera;

            pos.x() = pos_odom.x();
            pos.y() = pos_odom.y();
            pos.z() = pos_odom.z();
            Eigen::Matrix3d R_camera_to_odom = T_camera_to_odom.block<3, 3>(0, 0);
            Eigen::Quaterniond q_camera(ori.w(), ori.x(), ori.y(), ori.z());
            Eigen::Matrix3d R_ori_camera = q_camera.normalized().toRotationMatrix();

            Eigen::Matrix3d R_ori_odom = R_camera_to_odom * R_ori_camera;
            Eigen::Quaterniond q_odom(R_ori_odom);

            ori.w() = q_odom.w();
            ori.x() = q_odom.x();
            ori.y() = q_odom.y();
            ori.z() = q_odom.z();
        }
        std::vector<cv::Point2f> toPts(
            const cv::Mat& camera_intrinsic,
            const cv::Mat& camera_distortion,
            const std::vector<cv::Point3f>& obj_points = AIM_TARGET_BLOCK
        ) const {
            std::vector<cv::Point2f> pts;
            if (pos.norm() < 0.5) {
                return pts;
            }

            cv::Mat tvec = (cv::Mat_<double>(3, 1) << pos.x(), pos.y(), pos.z());
            Eigen::Matrix3d tf_rot = ori.toRotationMatrix();
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

            // 旋转矩阵 -> 旋转向量
            cv::Mat rvec;
            cv::Rodrigues(rot_mat, rvec);

            cv::projectPoints(obj_points, rvec, tvec, camera_intrinsic, camera_distortion, pts);

            return pts;
        }
        void draw(
            cv::Mat& img,
            const cv::Mat& camera_intrinsic,
            const cv::Mat& camera_distortion,
            const std::vector<cv::Point3f>& obj_points = AIM_TARGET_BLOCK,
            cv::Scalar color = cv::Scalar(255, 255, 255)
        ) const {
            auto pts = toPts(camera_intrinsic, camera_distortion, obj_points);
            if (!pts.empty()) {
                for (int i = 0; i < 4; i++)
                    cv::line(img, pts[i], pts[(i + 1) % 4], color, 2);

                // 后表面
                for (int i = 4; i < 8; i++)
                    cv::line(img, pts[i], pts[4 + (i + 1) % 4], color, 2);

                // 侧边
                for (int i = 0; i < 4; i++)
                    cv::line(img, pts[i], pts[i + 4], color, 2);
                cv::Point2f center(0.f, 0.f);
                for (auto pt: pts) {
                    center += pt;
                }
                center *= 1.0 / pts.size();
            }
        }
    };
    Pose center;
    std::vector<Pose> fans;
    int hit_id;
    void tf(Eigen::Matrix4d T_camera_to_odom) {
        center.tf(T_camera_to_odom);
        for (auto& fan: fans)
            fan.tf(T_camera_to_odom);
    }
    void
    draw(cv::Mat& img, const cv::Mat& camera_intrinsic, const cv::Mat& camera_distortion) const {
        center.draw(img, camera_intrinsic, camera_distortion);
        for (int i = 0; i < fans.size(); i++) {
            if (i == hit_id)
                fans[i].draw(
                    img,
                    camera_intrinsic,
                    camera_distortion,
                    FAN_BLOCK,
                    cv::Scalar(40, 255, 40)
                );
            else
                fans[i].draw(img, camera_intrinsic, camera_distortion, FAN_BLOCK);
        }
    }
};
} // namespace rune
