// Copyright 2025 XiaoJian Wu
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
#pragma once

#include "common/tf.hpp"
#include "opencv2/opencv.hpp"
#include <fmt/core.h>
#include <numeric>
#include <string>

constexpr double SMALL_ARMOR_WIDTH = 133.0 / 1000.0; // 135
constexpr double SMALL_ARMOR_HEIGHT = 50.0 / 1000.0; // 55
constexpr double LARGE_ARMOR_WIDTH = 225.0 / 1000.0;
constexpr double LARGE_ARMOR_HEIGHT = 50.0 / 1000.0; // 55

constexpr double SMALL_ARMOR_WIDTH_NET = 135.0 / 1000.0; // 135
constexpr double SMALL_ARMOR_HEIGHT_NET = 55.0 / 1000.0; // 55
constexpr double LARGE_ARMOR_WIDTH_NET = 225.0 / 1000.0;
constexpr double LARGE_ARMOR_HEIGHT_NET = 55.0 / 1000.0; // 55
constexpr double FIFTTEN_DEGREE_RAD = 15 * CV_PI / 180;

struct Light: public cv::RotatedRect {
    Light() = default;
    explicit Light(const std::vector<cv::Point>& contour):
        cv::RotatedRect(cv::minAreaRect(contour)) {
        center = std::accumulate(
            contour.begin(),
            contour.end(),
            cv::Point2f(0, 0),
            [n = static_cast<float>(contour.size())](const cv::Point2f& a, const cv::Point& b) {
                return a + cv::Point2f(b.x, b.y) / n;
            }
        );

        cv::Point2f p[4];
        this->points(p);
        std::sort(p, p + 4, [](const cv::Point2f& a, const cv::Point2f& b) { return a.y < b.y; });
        top = (p[0] + p[1]) / 2;
        bottom = (p[2] + p[3]) / 2;

        length = cv::norm(top - bottom);
        width = cv::norm(p[0] - p[1]);

        axis = top - bottom;
        axis = axis / cv::norm(axis);

        // Calculate the tilt angle
        // The angle is the angle between the light bar and the horizontal line
        tilt_angle = std::atan2(std::abs(top.x - bottom.x), std::abs(top.y - bottom.y));
        tilt_angle = tilt_angle / CV_PI * 180;
    }

    cv::Point2f top, bottom, center;
    int color;
    cv::Point2f axis;
    double length;
    double width;
    float tilt_angle;
};
struct LightParams {
    // width / height
    double min_ratio;
    double max_ratio;
    // vertical angle
    double max_angle;
    // judge color
    int color_diff_thresh;
};
struct ArmorParams {
    double min_light_ratio;
    // light pairs distance
    double min_small_center_distance;
    double max_small_center_distance;
    double min_large_center_distance;
    double max_large_center_distance;
    // horizontal angle
    double max_angle;
};
struct GridAndStride {
    int grid0;
    int grid1;
    int stride;
};

enum class ArmorColor { BLUE = 0, RED, NONE, PURPLE };

enum class ArmorNumber { SENTRY = 0, NO1, NO2, NO3, NO4, NO5, OUTPOST, BASE, UNKNOWN };
inline int formArmorColor(ArmorColor color) {
    switch (color) {
        case ArmorColor::RED:
            return 0;
        case ArmorColor::BLUE:
            return 1;
        case ArmorColor::NONE:
            return 2;
        case ArmorColor::PURPLE:
            return 3;
    }
}
inline int formArmorNumber(ArmorNumber number) {
    switch (number) {
        case ArmorNumber::SENTRY:
            return 0;
        case ArmorNumber::NO1:
            return 1;
        case ArmorNumber::NO2:
            return 2;
        case ArmorNumber::NO3:
            return 3;
        case ArmorNumber::NO4:
            return 4;
        case ArmorNumber::NO5:
            return 5;
        case ArmorNumber::OUTPOST:
            return 6;
        case ArmorNumber::BASE:
            return 7;
        case ArmorNumber::UNKNOWN:
            return 8;
    }
}
enum class AttackMode {
    ARMOR = 0,
    SMALL_RUNE,
    BIG_RUNE,
    UNKNOWN

};
enum class EnemyColor {
    RED = 0,
    BLUE = 1,
    WHITE = 2,
};
inline std::string enemyColorToString(EnemyColor color) {
    switch (color) {
        case EnemyColor::RED:
            return "RED";
        case EnemyColor::BLUE:
            return "BLUE";
        case EnemyColor::WHITE:
            return "WHITE";
        default:
            return "UNKNOWN";
    }
}
inline AttackMode toAttackMode(int value) {
    switch (value) {
        case 0:
            return AttackMode::ARMOR;
        case 1:
            return AttackMode::SMALL_RUNE;
        case 2:
            return AttackMode::BIG_RUNE;
        default:
            return AttackMode::UNKNOWN;
    }
}

inline int retypetotracker(ArmorNumber a) {
    static const std::unordered_map<ArmorNumber, int> map = {
        { ArmorNumber::SENTRY, 0 },  { ArmorNumber::NO1, 1 },  { ArmorNumber::NO2, 0 },
        { ArmorNumber::NO3, 0 },     { ArmorNumber::NO4, 0 },  { ArmorNumber::NO5, 0 },
        { ArmorNumber::OUTPOST, 2 }, { ArmorNumber::BASE, 1 }, { ArmorNumber::UNKNOWN, -1 }
    };

    auto it = map.find(a);
    if (it != map.end())
        return it->second;

    std::cerr << "[retypetotracker] Invalid ArmorNumber: " << static_cast<int>(a) << std::endl;
    return -1;
}
inline bool isSameTarget(ArmorNumber a, ArmorNumber b) {
    return retypetotracker(a) == retypetotracker(b);
}

template<>
struct fmt::formatter<ArmorNumber> {
    constexpr auto parse(fmt::format_parse_context& ctx) -> decltype(ctx.begin()) {
        return ctx.begin();
    }

    template<typename FormatContext>
    auto format(ArmorNumber num, FormatContext& ctx) -> decltype(ctx.out()) {
        const char* names[] = { "SENTRY", "NO1", "NO2", "NO3", "NO4", "NO5", "OUTPOST", "BASE" };
        int index = static_cast<int>(num);
        if (index >= 0 && index < sizeof(names) / sizeof(names[0])) {
            return fmt::format_to(ctx.out(), "{}", names[index]);
        } else {
            return fmt::format_to(ctx.out(), "UNKNOWN");
        }
    }
};

typedef struct ArmorObject {
    ArmorColor color;
    ArmorNumber number;
    float prob;
    std::vector<cv::Point2f> pts;
    std::vector<cv::Point2f> pts_binary;
    cv::Rect box;

    cv::Mat number_img;

    double confidence;

    cv::Mat whole_binary_img;
    cv::Mat whole_rgb_img;
    cv::Mat whole_gray_img;

    std::vector<Light> lights;

    cv::Point2f center;
    double new_x = 0;
    double new_y = 0;
    bool is_ok = false;
    bool is_ok_yaw = false;
    static constexpr const int N_LANDMARKS = 6;
    static constexpr const int N_LANDMARKS_2 = N_LANDMARKS * 2;

    // std::unique_ptr<LightCornerCorrector> corner_corrector;
    template<typename PointType>
    static inline std::vector<PointType>
    buildObjectPoints(const double& w, const double& h) noexcept {
        if constexpr (N_LANDMARKS == 4) {
            return { PointType(0, w / 2, -h / 2),
                     PointType(0, w / 2, h / 2),
                     PointType(0, -w / 2, h / 2),
                     PointType(0, -w / 2, -h / 2) };
        } else {
            return { PointType(0, w / 2, -h / 2), PointType(0, w / 2, 0),
                     PointType(0, w / 2, h / 2),  PointType(0, -w / 2, h / 2),
                     PointType(0, -w / 2, 0),     PointType(0, -w / 2, -h / 2) };
        }
    }

    // Landmarks start from bottom left in clockwise order
    std::vector<cv::Point2f> landmarks() const {
        if constexpr (N_LANDMARKS == 4) {
            return { lights[0].bottom, lights[0].top, lights[1].top, lights[1].bottom };
        } else {
            return { lights[0].bottom, lights[0].center, lights[0].top,
                     lights[1].top,    lights[1].center, lights[1].bottom };
        }
    }
    ArmorObject(const Light& l1, const Light& l2) {
        pts.resize(4);
        pts_binary.resize(4);
        if (l1.center.x < l2.center.x) {
            lights.push_back(l1);
            lights.push_back(l2);
            pts[0] = l1.top;
            pts[1] = l1.bottom;
            pts[2] = l2.bottom;
            pts[3] = l2.top;
            pts_binary[0] = l1.top;
            pts_binary[1] = l1.bottom;
            pts_binary[2] = l2.bottom;
            pts_binary[3] = l2.top;
        } else {
            lights.push_back(l2);
            lights.push_back(l1);
            pts[0] = l2.top;
            pts[1] = l2.bottom;
            pts[2] = l1.bottom;
            pts[3] = l1.top;
            pts_binary[0] = l2.top;
            pts_binary[1] = l2.bottom;
            pts_binary[2] = l1.bottom;
            pts_binary[3] = l1.top;
        }
        is_ok = true;
    }
    ArmorObject():
        box(),
        center(),
        color(),
        confidence(),
        is_ok(),
        is_ok_yaw(),
        number(),
        new_x(),
        new_y(),
        prob(),
        pts() {}
} ArmorObject;

constexpr const char* K_ARMOR_NAMES[] = { "sentry", "1", "2", "3", "4", "5", "outpost", "base" };

struct Armor {
    ArmorNumber number;
    std::string type;

    tf::Position pos;
    tf::Quaternion ori;
    tf::Position target_pos;
    tf::Quaternion target_ori;
    float distance_to_image_center;
    float yaw;
    std::chrono::steady_clock::time_point timestamp;
    bool is_ok;
};
struct Armors {
    std::vector<Armor> armors;
    std::chrono::steady_clock::time_point timestamp;
    std::string frame_id;
};
struct Target {
    std::chrono::steady_clock::time_point timestamp;
    std::string frame_id;
    std::string type;
    bool tracking = false;
    ArmorNumber id = ArmorNumber::UNKNOWN;
    int armors_num = 0;

    tf::Position position_ = tf::Position();
    tf::Position velocity_ = tf::Position();
    tf::Position acceleration_ = tf::Position();
    float yaw = 0;
    float v_yaw = 0;
    float radius_1 = 0.24;
    float radius_2 = 0.24;
    float d_za = 0;
    float d_zc = 0;
    float yaw_diff;
    float position_diff;
    int count = 0;

    void clear() {
        id = ArmorNumber::UNKNOWN;
        tracking = false;
        armors_num = 0;
        type = "";
        position_ = tf::Position();
        velocity_ = tf::Position();
        yaw = 0.0;
        v_yaw = 0.0;
        radius_1 = 0.0;
        radius_2 = 0.0;
        d_zc = 0.0;
        d_za = 0.0;
        yaw_diff = 0.0;
        position_diff = 0.0;
    }
};
struct OneTarget {
    std::chrono::steady_clock::time_point timestamp;
    std::string frame_id;
    std::string type;
    bool tracking = false;
    ArmorNumber id = ArmorNumber::UNKNOWN;
    tf::Position position_ = tf::Position();
    tf::Position velocity_ = tf::Position();
    tf::Position acceleration_ = tf::Position();
    float yaw = 0;
    float v_yaw = 0;
    float distance_to_image_center = 0;
    int count = 0;
    bool is_omni = false;

    void clear() {
        id = ArmorNumber::UNKNOWN;
        tracking = false;

        type = "";
        position_ = tf::Position();
        velocity_ = tf::Position();
        yaw = 0.0;
        v_yaw = 0.0;
    }
};
struct imgframe {
    cv::Mat img;
    std::chrono::steady_clock::time_point timestamp;
};
struct SyncedData {
    Target target;
    imgframe image;
};
struct GimbalCmd {
    std::chrono::steady_clock::time_point timestamp;
    float pitch = 0;
    float yaw = 0;
    float yaw_diff = 0;
    float pitch_diff = 0;
    float v_yaw = 0;
    float v_pitch = 0;
    float distance = -1;
    bool fire_advice = false;
    int select_id = -1;
};
enum class RuneType { INACTIVATED = 0, ACTIVATED };

struct FeaturePoints {
    FeaturePoints() {
        r_center = cv::Point2f(-1, -1);
        bottom_right = cv::Point2f(-1, -1);
        top_right = cv::Point2f(-1, -1);
        top_left = cv::Point2f(-1, -1);
        bottom_left = cv::Point2f(-1, -1);
    }

    void reset() {
        r_center = cv::Point2f(-1, -1);
        bottom_right = cv::Point2f(-1, -1);
        top_right = cv::Point2f(-1, -1);
        top_left = cv::Point2f(-1, -1);
        bottom_left = cv::Point2f(-1, -1);
    }

    FeaturePoints operator+(const FeaturePoints& other) {
        FeaturePoints res;
        res.r_center = r_center + other.r_center;
        res.bottom_right = bottom_right + other.bottom_right;
        res.top_right = top_right + other.top_right;
        res.top_left = top_left + other.top_left;
        res.bottom_left = bottom_left + other.bottom_left;
        return res;
    }

    FeaturePoints operator/(const float& other) {
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
    cv::Mat M;
    cv::Mat target_img;
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
};

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

enum class ArmorsNum { NORMAL_4 = 4, OUTPOST_3 = 3 };

enum class ArmorType { SMALL, LARGE, INVALID };
inline std::string armorTypeToString(const ArmorType& type) {
    switch (type) {
        case ArmorType::SMALL:
            return "small";
        case ArmorType::LARGE:
            return "large";
        default:
            return "invalid";
    }
}
