#pragma once
#include "tasks/type_common.hpp"
#include <fmt/core.h>
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

    void draw(
        cv::Mat& img,
        const cv::Point2f& offset,
        const Eigen::Matrix<float, 3, 3>& transform_matrix,
        bool is_up
    ) const;
    double getArea() const;
    void addReferRuneCenter(const RuneCenter& rc);
};

struct RuneFan {
    bool is_valid = false;
    int id;
    bool is_big = false;
    std::chrono::steady_clock::time_point timestamp;
    struct Simple {
        std::vector<double> angle_diffs = { 0,
                                            2 * M_PI / 5,
                                            2 * M_PI / 5 * 2,
                                            2 * M_PI / 5 * 3,
                                            2 * M_PI / 5 * 4 };
        std::vector<cv::Point2f> points2d;
        std::vector<cv::Point3f> points3d = {
            { 0.0f, 0.0f, 0.0f }, // P0
            { 0.0f, RUNE_PAN_REAL_DIS / 2.0f, RUNE_R2PANCENTER + RUNE_PAN_REAL_DIS / 2.0f }, // P1
            { 0.0f, RUNE_PAN_REAL_DIS / 2.0f, RUNE_R2PANCENTER - RUNE_PAN_REAL_DIS / 2.0f }, // P2
            { 0.0f, -RUNE_PAN_REAL_DIS / 2.0f, RUNE_R2PANCENTER - RUNE_PAN_REAL_DIS / 2.0f }, // P3
            { 0.0f, -RUNE_PAN_REAL_DIS / 2.0f, RUNE_R2PANCENTER + RUNE_PAN_REAL_DIS / 2.0f }, // P4
            { 0.0f, 0.0f, RUNE_R2PANCENTER } // P5
        };
        inline cv::Point3f rotateX(const cv::Point3f& p, double roll) {
            double c = std::cos(roll);
            double s = std::sin(roll);
            return { p.x, float(p.y * c - p.z * s), float(p.y * s + p.z * c) };
        }
        inline double normalizeAngle0to2pi(double a) {
            a = std::fmod(a, 2 * M_PI);
            if (a < 0)
                a += 2 * M_PI;
            return a;
        }

        Eigen::Vector3d pos;
        Eigen::Quaterniond ori;
        Eigen::Vector3d target_pos;
        Eigen::Quaterniond target_ori;
        void addOther(const Simple& other);
        std::vector<cv::Point2f> landmarks() const {
            return points2d;
        }
        void drawLandmarks(cv::Mat& image) const;
        std::vector<cv::Point3f> getObjs() const {
            return points3d;
        }
    };
    std::vector<Simple> fans;
    void addOffset(const cv::Point2f& offset);
    void transform(const Eigen::Matrix<float, 3, 3>& transform_matrix);
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
        void tf(Eigen::Matrix4d T_camera_to_odom);
        std::vector<cv::Point2f> toPts(
            const cv::Mat& camera_intrinsic,
            const cv::Mat& camera_distortion,
            const std::vector<cv::Point3f>& obj_points = AIM_TARGET_BLOCK
        ) const;
        void draw(
            cv::Mat& img,
            const cv::Mat& camera_intrinsic,
            const cv::Mat& camera_distortion,
            const std::vector<cv::Point3f>& obj_points = AIM_TARGET_BLOCK,
            cv::Scalar color = cv::Scalar(255, 255, 255)
        ) const;
    };
    Pose center;
    std::vector<Pose> fans;
    int hit_id;
    void tf(Eigen::Matrix4d T_camera_to_odom);
    void
    draw(cv::Mat& img, const cv::Mat& camera_intrinsic, const cv::Mat& camera_distortion) const;
};
} // namespace rune
