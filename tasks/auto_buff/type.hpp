#pragma once
#include "opencv2/opencv.hpp"
#include "tasks/type_common.hpp"
#include "tasks/utils.hpp"
#include <fmt/core.h>
#include <numeric>
#include <string>

constexpr double DEG_72 = 0.4 * CV_PI;
constexpr int ARMOR_KEYPOINTS_NUM = 5;
constexpr int KEYPOINTS_NUM = 6;

// Motion type
enum class MotionType { SMALL, BIG, UNKNOWN };

// Moving direction
enum Direction { CLOCKWISE = -1, ANTI_CLOCKWISE = 1, UNKNOWN = 0 };

// Rune arm length, Unit: m
constexpr double ARM_LENGTH = 0.700;

// Acceptable distance between robot and rune, Unit: m
// True value = 6.436 m
constexpr double MIN_RUNE_DISTANCE = 1.0;
constexpr double MAX_RUNE_DISTANCE = 19.0;

// Rune object points
// r_tag, bottom_left, top_left, top_right, bottom_right
// const std::vector<cv::Point3f> RUNE_OBJECT_POINTS = { cv::Point3f(0, 0, 0) / 1000,
//                                                       cv::Point3f(0, -541.5, 186) / 1000,
//                                                       cv::Point3f(0, -858.5, 160) / 1000,
//                                                       cv::Point3f(0, -858.5, -160) / 1000,
//                                                       cv::Point3f(0, -541.5, -186) / 1000 };
const std::vector<cv::Point3f> RUNE_OBJECT_POINTS = { cv::Point3f(0.0, 0, 0),
                                                      // cv::Point3f(0.69, 0.0, 0.0),
                                                      cv::Point3f(0.0, 0.845, 0.0),
                                                      cv::Point3f(0.0, 0.69, 0.155),

                                                      cv::Point3f(0.0, 0.535, 0.0),
                                                      cv::Point3f(0.0, 0.69, -0.155) };
#define BIG_RUNE_CURVE(x, a, omega, b, c, d, sign) \
    ((-((a) / (omega)*ceres::cos((omega) * ((x) + (d)))) + (b) * ((x) + (d)) + (c)) * (sign))

#define SMALL_RUNE_CURVE(x, a, b, c, sign) (((a) * ((x) + (b)) + (c)) * (sign))

namespace rune {
enum class RuneType { INACTIVATED = 0, ACTIVATED };

struct FeaturePoints {
    // 默认构造：全部初始化为 (-1, -1)
    FeaturePoints() {
        reset();
    }

    // 构造：全部初始化为同一个点
    explicit FeaturePoints(const cv::Point2f& p) {
        r_tag = fan_center = fan_top = fan_right = fan_bottom = fan_left = p;
    }

    // 构造：直接传 6 个点
    FeaturePoints(
        const cv::Point2f& rt,
        const cv::Point2f& fc,
        const cv::Point2f& ft,
        const cv::Point2f& fr,
        const cv::Point2f& fb,
        const cv::Point2f& fl
    ):
        r_tag(rt),
        fan_center(fc),
        fan_top(ft),
        fan_right(fr),
        fan_bottom(fb),
        fan_left(fl) {}

    // 构造：从 std::vector<cv::Point2f> 赋值
    explicit FeaturePoints(const std::vector<cv::Point2f>& pts) {
        reset();
        if (pts.size() >= 6) {
            r_tag = pts[0];
            fan_center = pts[1];
            fan_top = pts[2];
            fan_right = pts[3];
            fan_bottom = pts[4];
            fan_left = pts[5];
        }
    }
    void drawFan(
        cv::Mat& image,
        const cv::Scalar& color = cv::Scalar(0, 255, 0),
        int thickness = 2,
        int font_scale = 1,
        int font_thickness = 1
    ) const {
        // 关键点顺序
        std::vector<std::pair<cv::Point, std::string>> pts = {
            { r_tag, "r_tag" },         { fan_center, "fan_center" }, { fan_top, "fan_top" },
            { fan_right, "fan_right" }, { fan_bottom, "fan_bottom" }, { fan_left, "fan_left" }
        };

        // 1. r_tag ↔ fan_center
        if (r_tag.x >= 0 && r_tag.y >= 0 && fan_center.x >= 0 && fan_center.y >= 0)
            cv::line(image, r_tag, fan_center, color, thickness);

        // 2. fan_top → fan_right → fan_bottom → fan_left → fan_top
        std::vector<cv::Point> quad = { fan_top, fan_right, fan_bottom, fan_left, fan_top };
        for (size_t i = 0; i < quad.size() - 1; ++i) {
            if (quad[i].x >= 0 && quad[i].y >= 0 && quad[i + 1].x >= 0 && quad[i + 1].y >= 0)
                cv::line(image, quad[i], quad[i + 1], color, thickness);
        }

        // 绘制关键点名字
        for (const auto& pt_pair: pts) {
            cv::Point pt = pt_pair.first;
            if (pt.x >= 0 && pt.y >= 0) {
                cv::putText(
                    image,
                    pt_pair.second,
                    pt + cv::Point(2, -2),
                    cv::FONT_HERSHEY_SIMPLEX,
                    font_scale,
                    color,
                    font_thickness
                );
            }
        }
    }

    void reset() {
        cv::Point2f invalid(-1, -1);
        r_tag = fan_center = fan_top = fan_right = fan_bottom = fan_left = invalid;
    }

    FeaturePoints operator+(const FeaturePoints& other) const {
        FeaturePoints res;
        res.r_tag = r_tag + other.r_tag;
        res.fan_center = fan_center + other.fan_center;
        res.fan_top = fan_top + other.fan_top;
        res.fan_right = fan_right + other.fan_right;
        res.fan_bottom = fan_bottom + other.fan_bottom;
        res.fan_left = fan_left + other.fan_left;
        return res;
    }

    FeaturePoints operator/(float other) const {
        FeaturePoints res;
        res.r_tag = r_tag / other;
        res.fan_center = fan_center / other;
        res.fan_top = fan_top / other;
        res.fan_right = fan_right / other;
        res.fan_bottom = fan_bottom / other;
        res.fan_left = fan_left / other;
        return res;
    }

    std::vector<cv::Point2f> toVector2f() const {
        return { r_tag, fan_center, fan_top, fan_right, fan_bottom, fan_left };
    }

    std::vector<cv::Point> toVector2i() const {
        return { r_tag, fan_center, fan_top, fan_right, fan_bottom, fan_left };
    }

    cv::Point2f r_tag;
    cv::Point2f fan_center;
    cv::Point2f fan_top;
    cv::Point2f fan_right;
    cv::Point2f fan_bottom;
    cv::Point2f fan_left;

    std::vector<FeaturePoints> children;
};

struct RuneObject {
    EnemyColor color;
    RuneType type;
    float prob;
    FeaturePoints pts;
    cv::Rect box;
};
constexpr float R_height = 0.15;
constexpr float R2L = 0.2;
constexpr float L_height = 0.3;
constexpr float L_width = 0.05;
constexpr float L2C = 0.2;
constexpr float C_r = 0.15;
struct RuneFan {
    cv::Point2f fan_center;
    int radius;
    std::vector<cv::Point2f> fan_kpoints;
    cv::Point2f r_tag;
    cv::RotatedRect mid_rect;
    cv::Point2f base = cv::Point2f(0, 0);
    bool is_valid = false;
    void drawPoints(cv::Mat& img) const {
        if (!is_valid)
            return;
        cv::circle(img, fan_center + base, 5, cv::Scalar(0, 0, 255), -1);
        cv::circle(img, r_tag + base, 5, cv::Scalar(0, 255, 255), -1);
        for (auto& p: fan_kpoints)
            cv::circle(img, p + base, 5, cv::Scalar(255, 255, 0), -1);
        cv::Point2f rect_pts[4];
        mid_rect.points(rect_pts);
        cv::Point2f rect_center;
        for (auto& p: rect_pts) {
            cv::circle(img, p + base, 5, cv::Scalar(255, 255, 0), -1);
            rect_center += p + base;
        }
        rect_center /= 4.0;
        cv::circle(img, rect_center, 5, cv::Scalar(255, 255, 255), -1);
    }
    std::vector<cv::Point2f> toVectorP2f() {
        return {
            r_tag, fan_center, fan_kpoints[2], fan_kpoints[1], fan_kpoints[3], fan_kpoints[0]
        };
    }
    std::vector<cv::Point2f> sortRay() {
        std::vector<cv::Point2f> sorted_ray;
        if (fan_kpoints.size() != 4)
            return sorted_ray;

        // CR向量，r_tag 在 center 下方
        cv::Point2f down_dir = r_tag - fan_center;
        float angle_ref = std::atan2(down_dir.y, down_dir.x); // 参考角度

        // 将4个点按与 center 的相对方向分类
        cv::Point2f up, right, down, left;
        for (auto& p: fan_kpoints) {
            cv::Point2f v = p - fan_center;
            float angle = std::atan2(v.y, v.x) - angle_ref;

            // 将角度归一到 [-π, π]
            while (angle <= -CV_PI)
                angle += 2 * CV_PI;
            while (angle > CV_PI)
                angle -= 2 * CV_PI;

            if (angle > -CV_PI / 4 && angle <= CV_PI / 4)
                down = p; // 接近参考向下方向
            else if (angle > CV_PI / 4 && angle <= 3 * CV_PI / 4)
                left = p;
            else if (angle <= -CV_PI / 4 && angle > -3 * CV_PI / 4)
                right = p;
            else
                up = p;
        }

        sorted_ray = { up, right, down, left };
        return sorted_ray;
    }

    std::vector<cv::Point2f> sortRect() {
        std::vector<cv::Point2f> sorted_rect;
        cv::Point2f rect_pts[4];
        mid_rect.points(rect_pts);

        // 中心
        cv::Point2f c(0, 0);
        for (int i = 0; i < 4; i++)
            c += rect_pts[i];
        c *= 0.25f;

        cv::Point2f lt, rt, rb, lb;
        for (int i = 0; i < 4; i++) {
            auto& p = rect_pts[i];
            if (p.x < c.x && p.y < c.y)
                lt = p; // 左上
            if (p.x > c.x && p.y < c.y)
                rt = p; // 右上
            if (p.x > c.x && p.y > c.y)
                rb = p; // 右下
            if (p.x < c.x && p.y > c.y)
                lb = p; // 左下
        }

        sorted_rect = { lt, rt, rb, lb };
        return sorted_rect;
    }

    std::vector<cv::Point2f> landmarks() {
        auto sorted_rect = sortRect();
        auto sorted_ray = sortRay();
        auto l_center = utils::getCenter(sorted_rect);
        return {
            r_tag + base,         sorted_ray[0] + base, sorted_ray[1] + base,
            sorted_ray[2] + base, sorted_ray[3] + base,
            //center + base
        };
    }
    std::vector<cv::Point3f> getObjs() {
        return {
            cv::Point3f { 0.0, 0.0, 0.0 },
            //cv::Point3f { 0.0, R2L + L_height / 2.0, 0.0 },
            cv::Point3f { 0.0, 0.0, R2L + L_height + L2C + C_r },
            cv::Point3f { 0.0, C_r, R2L + L_height + L2C },
            cv::Point3f { 0.0, 0.0, R2L + L_height + L2C - C_r },
            cv::Point3f { 0.0, -C_r, R2L + L_height + L2C },
            // cv::Point3f { R2L + L_height + L2C + C_r, 0.0}
        };
    }
};

struct Rune {
    std::string frame_id;
    std::chrono::steady_clock::time_point timestamp;
    std::vector<cv::Point2f> pts;
    bool is_lost;
    bool is_big_rune;
    int id;
    Eigen::Matrix4d T_camera_to_odom;
};

} // namespace rune
