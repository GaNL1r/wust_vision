#pragma once
#include "opencv2/opencv.hpp"
#include "tasks/type_common.hpp"
#include "tasks/utils.hpp"
#include <fmt/core.h>
#include <numeric>
#include <string>

namespace rune {
constexpr double RUNE_PAN_REAL_DIS = 0.15;
constexpr double RUNE_FAN_REAL_W = 0.05;
constexpr double RUNE_FAN_REAL_H = 0.3;
constexpr double RUNE_R = 0.05;
constexpr double RUNE_R2PANCENTER = 0.7;
struct RuneCenter {
    cv::Point2f center;
    cv::RotatedRect rr;
    bool is_valid = false;
    RuneCenter() = default;
    RuneCenter(cv::RotatedRect rect): rr(rect) {
        center = rect.center;
        is_valid = rr.size.area() > 0;
    }
};

class RunePan {
public:
    cv::Point2f center;
    std::vector<cv::Point2f> corners;
    bool is_valid = false;
    bool has_refer = false;

    void draw(cv::Mat& img) const {
        if (!is_valid || corners.size() < 3)
            return;

        std::vector<cv::Point2f> sorted_corners = corners;

        // 画边
        for (size_t i = 0; i < sorted_corners.size(); ++i) {
            cv::line(
                img,
                sorted_corners[i],
                sorted_corners[(i + 1) % sorted_corners.size()],
                cv::Scalar(0, 255, 255),
                2
            );
        }

        // 画中心点
        cv::circle(img, center, 3, cv::Scalar(255, 0, 255), -1);
        if (has_refer) {
            // 画角点编号
            for (size_t i = 0; i < sorted_corners.size(); ++i) {
                cv::Point2f p = sorted_corners[i];

                // 绘制角点位置
                cv::circle(img, p, 3, cv::Scalar(0, 0, 255), -1);

                // 让数字稍微往右下偏移，避免盖到角点
                cv::Point2f text_pos = p + cv::Point2f(5, -5);

                cv::putText(
                    img,
                    std::to_string(i),
                    text_pos,
                    cv::FONT_HERSHEY_SIMPLEX,
                    0.5,
                    cv::Scalar(0, 255, 0),
                    1
                );
            }
        }
    }

    double getArea() const {
        if (corners.size() < 3)
            return 0.0;
        std::vector<cv::Point2f> sorted_corners = corners;
        std::sort(
            sorted_corners.begin(),
            sorted_corners.end(),
            [this](const cv::Point2f& a, const cv::Point2f& b) {
                double angA = std::atan2(a.y - center.y, a.x - center.x);
                double angB = std::atan2(b.y - center.y, b.x - center.x);
                return angA < angB;
            }
        );
        return cv::contourArea(sorted_corners);
    }
    void addReferRuneCenter(const RuneCenter& rc) {
        if (!rc.is_valid || !is_valid)
            return;
        if (corners.size() != 4)
            return;

        cv::Point2f down_vec = rc.center - center;
        float norm = std::sqrt(down_vec.x * down_vec.x + down_vec.y * down_vec.y);
        if (norm < 1e-6f)
            return;
        has_refer = true;
        float angle_ref = std::atan2(down_vec.y, down_vec.x);

        // 获取4个点在旋转后的角度
        struct Node {
            float ang;
            cv::Point2f p;
        };
        std::vector<Node> arr;
        arr.reserve(4);

        for (auto& p: corners) {
            cv::Point2f v = p - center;

            // 旋转坐标，使 down_vec 对齐 angle=0
            float ang = std::atan2(v.y, v.x) - angle_ref;

            // 归一化到 (-π, π]
            while (ang <= -CV_PI)
                ang += 2 * CV_PI;
            while (ang > CV_PI)
                ang -= 2 * CV_PI;

            arr.push_back({ ang, p });
        }

        // 按角度排序（从 -π 到 π）
        std::sort(arr.begin(), arr.end(), [](const Node& a, const Node& b) {
            return a.ang < b.ang;
        });

        // 准备象限变量并标记
        cv::Point2f lu(0, 0), ru(0, 0), rd(0, 0), ld(0, 0);
        bool has_lu = false, has_ru = false, has_rd = false, has_ld = false;

        for (const auto& n: arr) {
            float a = n.ang;

            if (a > CV_PI / 2 && a <= CV_PI) {
                lu = n.p;
                has_lu = true;
            } else if (a > 0 && a <= CV_PI / 2) {
                ru = n.p;
                has_ru = true;
            } else if (a > -CV_PI / 2 && a <= 0) {
                rd = n.p;
                has_rd = true;
            } else { // a > -CV_PI && a <= -CV_PI/2
                ld = n.p;
                has_ld = true;
            }
        }

        std::array<cv::Point2f, 4> ordered;

        if (has_lu && has_ru && has_rd && has_ld) {
            ordered[0] = lu;
            ordered[1] = ru;
            ordered[2] = rd;
            ordered[3] = ld;
            corners.assign(ordered.begin(), ordered.end());
            return;
        }

        float target = 3.0f * CV_PI / 4.0f; // 135°
        int best_idx = 0;
        float best_diff = std::numeric_limits<float>::max();
        for (int i = 0; i < (int)arr.size(); ++i) {
            float d = std::fabs(angles::shortest_angular_distance(target, arr[i].ang)
            ); // 如果没有 angles::shortest_angular_distance，可以用下面替代
            if (d < best_diff) {
                best_diff = d;
                best_idx = i;
            }
        }

        for (int i = 0; i < 4; ++i) {
            int idx = (best_idx + i) % 4;
            ordered[i] = arr[idx].p;
        }

        corners.assign(ordered.begin(), ordered.end());
    }
};

struct RuneFan {
    bool is_valid = false;
    int id;
    bool is_big = false;
    std::chrono::steady_clock::time_point timestamp;
    struct Simple {
        std::vector<cv::Point2f> points2d;
        std::vector<cv::Point3f> points3d = {
            { 0.0f, 0.0f, 0.0f }, // P0
            { 0.0f, -RUNE_PAN_REAL_DIS / 2.0f, RUNE_R2PANCENTER + RUNE_PAN_REAL_DIS / 2.0f }, // P1
            { 0.0f, -RUNE_PAN_REAL_DIS / 2.0f, RUNE_R2PANCENTER - RUNE_PAN_REAL_DIS / 2.0f }, // P2
            { 0.0f, RUNE_PAN_REAL_DIS / 2.0f, RUNE_R2PANCENTER - RUNE_PAN_REAL_DIS / 2.0f }, // P3
            { 0.0f, RUNE_PAN_REAL_DIS / 2.0f, RUNE_R2PANCENTER + RUNE_PAN_REAL_DIS / 2.0f }, // P4
            //{ 0.0f, 0.0f, RUNE_R2PANCENTER } // P5
        };
        Eigen::Vector3d pos;
        Eigen::Quaterniond ori;
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
    std::vector<Simple> fans;
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
