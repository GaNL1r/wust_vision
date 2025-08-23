#pragma once
#include "opencv2/opencv.hpp"
#include "tasks/type_common.hpp"
#include <fmt/core.h>
#include <numeric>
#include <string>

constexpr double DEG_72 = 0.4 * CV_PI;
constexpr int ARMOR_KEYPOINTS_NUM = 4;
constexpr int KEYPOINTS_NUM = 5;

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
const std::vector<cv::Point3f> RUNE_OBJECT_POINTS = { cv::Point3f(0, 0, 0) / 1000,
                                                      cv::Point3f(0, -541.5, 186) / 1000,
                                                      cv::Point3f(0, -858.5, 160) / 1000,
                                                      cv::Point3f(0, -858.5, -160) / 1000,
                                                      cv::Point3f(0, -541.5, -186) / 1000 };

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
        r_center = bottom_right = top_right = top_left = bottom_left = p;
    }

    // 构造：直接传 5 个点
    FeaturePoints(
        const cv::Point2f& rc,
        const cv::Point2f& br,
        const cv::Point2f& tr,
        const cv::Point2f& tl,
        const cv::Point2f& bl
    ):
        r_center(rc),
        bottom_right(br),
        top_right(tr),
        top_left(tl),
        bottom_left(bl) {}

    // 构造：从 std::vector<cv::Point2f> 赋值
    explicit FeaturePoints(const std::vector<cv::Point2f>& pts) {
        reset();
        if (pts.size() >= 5) {
            r_center = pts[0];
            bottom_left = pts[1];
            top_left = pts[2];
            top_right = pts[3];
            bottom_right = pts[4];
        }
    }

    void reset() {
        cv::Point2f invalid(-1, -1);
        r_center = bottom_right = top_right = top_left = bottom_left = invalid;
    }

    FeaturePoints operator+(const FeaturePoints& other) const {
        FeaturePoints res;
        res.r_center = r_center + other.r_center;
        res.bottom_right = bottom_right + other.bottom_right;
        res.top_right = top_right + other.top_right;
        res.top_left = top_left + other.top_left;
        res.bottom_left = bottom_left + other.bottom_left;
        return res;
    }

    FeaturePoints operator/(float other) const {
        FeaturePoints res;
        res.r_center = r_center / other;
        res.bottom_right = bottom_right / other;
        res.top_right = top_right / other;
        res.top_left = top_left / other;
        res.bottom_left = bottom_left / other;
        return res;
    }

    std::vector<cv::Point2f> toVector2f() const {
        return { r_center, bottom_left, top_left, top_right, bottom_right };
    }
    std::vector<cv::Point> toVector2i() const {
        return { r_center, bottom_left, top_left, top_right, bottom_right };
    }

    cv::Point2f r_center;
    cv::Point2f bottom_right;
    cv::Point2f top_right;
    cv::Point2f top_left;
    cv::Point2f bottom_left;

    std::vector<FeaturePoints> children;
};

struct RuneObject {
    EnemyColor color;
    RuneType type;
    float prob;
    FeaturePoints pts;
    cv::Rect box;
};
struct RunePoint {
    float x;
    float y;
};
struct Rune {
    std::string frame_id;
    std::chrono::steady_clock::time_point timestamp;
    std::array<RunePoint, 5> pts;
    bool is_lost;
    bool is_big_rune;
    int id;
    Eigen::Matrix4d T_camera_to_odom;
};

} // namespace rune
