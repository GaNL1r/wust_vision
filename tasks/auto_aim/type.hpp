#pragma once
#include "tasks/type_common.hpp"
#include <numeric>
#include <opencv2/opencv.hpp>
constexpr double SMALL_ARMOR_WIDTH = 133.0 / 1000.0; // 135
constexpr double SMALL_ARMOR_HEIGHT = 50.0 / 1000.0; // 55
constexpr double LARGE_ARMOR_WIDTH = 225.0 / 1000.0;
constexpr double LARGE_ARMOR_HEIGHT = 50.0 / 1000.0; // 55

constexpr double SMALL_ARMOR_WIDTH_NET = 135.0 / 1000.0; // 135
constexpr double SMALL_ARMOR_HEIGHT_NET = 55.0 / 1000.0; // 55
constexpr double LARGE_ARMOR_WIDTH_NET = 225.0 / 1000.0;
constexpr double LARGE_ARMOR_HEIGHT_NET = 55.0 / 1000.0; // 55
constexpr double FIFTTEN_DEGREE_RAD = 15 * CV_PI / 180;
namespace armor {
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

enum class ArmorColor { BLUE = 0, RED, NONE, PURPLE };

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
enum class ArmorNumber { SENTRY = 0, NO1, NO2, NO3, NO4, NO5, OUTPOST, BASE, UNKNOWN };
inline std::ostream& operator<<(std::ostream& os, ArmorNumber number) {
    switch (number) {
        case ArmorNumber::SENTRY:
            return os << "SENTRY";
        case ArmorNumber::NO1:
            return os << "NO1";
        case ArmorNumber::NO2:
            return os << "NO2";
        case ArmorNumber::NO3:
            return os << "NO3";
        case ArmorNumber::NO4:
            return os << "NO4";
        case ArmorNumber::NO5:
            return os << "NO5";
        case ArmorNumber::OUTPOST:
            return os << "OUTPOST";
        case ArmorNumber::BASE:
            return os << "BASE";
        case ArmorNumber::UNKNOWN:
            return os << "UNKNOWN";
        default:
            return os << "InvalidArmorNumber(" << static_cast<int>(number) << ")";
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
inline int retypetotracker(ArmorNumber a) {
    static const std::unordered_map<ArmorNumber, int> map = {
        { ArmorNumber::SENTRY, 0 },  { ArmorNumber::NO1, 1 },  { ArmorNumber::NO2, 0 },
        { ArmorNumber::NO3, 0 },     { ArmorNumber::NO4, 0 },  { ArmorNumber::NO5, 0 },
        { ArmorNumber::OUTPOST, 6 }, { ArmorNumber::BASE, 7 }, { ArmorNumber::UNKNOWN, -1 }
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

inline std::string armorNumberToString(ArmorNumber num) {
    switch (num) {
        case ArmorNumber::SENTRY:
            return "SENTRY";
        case ArmorNumber::BASE:
            return "BASE";
        case ArmorNumber::OUTPOST:
            return "OUTPOST";
        case ArmorNumber::NO1:
            return "NO1";
        case ArmorNumber::NO2:
            return "NO2";
        case ArmorNumber::NO3:
            return "NO3";
        case ArmorNumber::NO4:
            return "NO4";
        case ArmorNumber::NO5:
            return "NO5";
        default:
            return "UNKNOWN";
    }
}

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

struct ArmorObject {
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
};

constexpr const char* K_ARMOR_NAMES[] = { "sentry", "1", "2", "3", "4", "5", "outpost", "base" };

struct Armor {
    ArmorNumber number;
    std::string type;

    Eigen::Vector3d pos;
    Eigen::Quaterniond ori;
    Eigen::Vector3d target_pos;
    Eigen::Quaterniond target_ori;
    float distance_to_image_center;
    float yaw;
    std::chrono::steady_clock::time_point timestamp;
    bool is_ok = false;
    bool is_none_purple = false;
};
struct Armors {
    std::vector<Armor> armors;
    std::chrono::steady_clock::time_point timestamp;
    std::string frame_id;
    int id;
    Eigen::Matrix3d R_gimbal2odom;
    Eigen::Vector3d v;
};
struct Target {
    std::chrono::steady_clock::time_point timestamp;
    std::string frame_id;
    std::string type;
    bool tracking = false;
    ArmorNumber id = ArmorNumber::UNKNOWN;
    int armors_num = 0;

    Eigen::Vector3d position_ = Eigen::Vector3d();
    Eigen::Vector3d velocity_ = Eigen::Vector3d();
    Eigen::Vector3d acceleration_ = Eigen::Vector3d();
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
        position_ = Eigen::Vector3d();
        velocity_ = Eigen::Vector3d();
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
    Eigen::Vector3d position_ = Eigen::Vector3d();
    Eigen::Vector3d velocity_ = Eigen::Vector3d();
    Eigen::Vector3d acceleration_ = Eigen::Vector3d();
    float yaw = 0;
    float v_yaw = 0;
    float distance_to_image_center = 0;
    int count = 0;
    bool is_omni = false;

    void clear() {
        id = ArmorNumber::UNKNOWN;
        tracking = false;
        type = "";
        position_ = Eigen::Vector3d();
        velocity_ = Eigen::Vector3d();
        yaw = 0.0;
        v_yaw = 0.0;
    }
};

} // namespace armor
struct ArmorSolverTarget {
    armor::Target target;
    std::vector<armor::OneTarget> one_targets;
};
