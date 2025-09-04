#pragma once
#include "3rdparty/angles.h"
#include "tasks/type_common.hpp"
#include <numeric>
#include <opencv2/opencv.hpp>
#include <yaml-cpp/yaml.h>
constexpr double SMALL_ARMOR_WIDTH = 133.0 / 1000.0; // 135
constexpr double SMALL_ARMOR_HEIGHT = 50.0 / 1000.0; // 55
constexpr double LARGE_ARMOR_WIDTH = 225.0 / 1000.0;
constexpr double LARGE_ARMOR_HEIGHT = 50.0 / 1000.0; // 55

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
    armor::ArmorType type;
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
    std::vector<cv::Point2f> toPts() const {
        if (is_ok) {
            return { pts_binary[0], pts_binary[1], pts_binary[2], pts_binary[3] };
        } else {
            return { pts[0], pts[1], pts[2], pts[3] };
        }
    }
    std::array<cv::Point2f, 4> sortCorners(const std::vector<cv::Point2f>& pts) const {
        std::array<cv::Point2f, 4> ordered;

        // 先按 x 坐标分成左右两组
        std::vector<cv::Point2f> left, right;
        std::vector<cv::Point2f> sorted = pts;

        std::sort(sorted.begin(), sorted.end(), [](const cv::Point2f& a, const cv::Point2f& b) {
            return a.x < b.x;
        });

        left.push_back(sorted[0]);
        left.push_back(sorted[1]);
        right.push_back(sorted[2]);
        right.push_back(sorted[3]);

        // 左边两个点，按 y 分为上/下
        std::sort(left.begin(), left.end(), [](const cv::Point2f& a, const cv::Point2f& b) {
            return a.y < b.y;
        });
        ordered[1] = left[0]; // 左上
        ordered[0] = left[1]; // 左下

        // 右边两个点，按 y 分为上/下
        std::sort(right.begin(), right.end(), [](const cv::Point2f& a, const cv::Point2f& b) {
            return a.y < b.y;
        });
        ordered[2] = right[0]; // 右上
        ordered[3] = right[1]; // 右下

        return ordered; // 顺序: 左下, 左上, 右上, 右下
    }

    // Landmarks start from bottom left in clockwise order
    std::vector<cv::Point2f> landmarks() const {
        if constexpr (N_LANDMARKS == 4) {
            if (is_ok) {
                return { lights[0].bottom, lights[0].top, lights[1].top, lights[1].bottom };
            } else {
                auto ordered = sortCorners(pts);
                return { ordered[0], ordered[1], ordered[2], ordered[3] };
            }

        } else {
            if (is_ok) {
                return { lights[0].bottom, lights[0].center, lights[0].top,
                         lights[1].top,    lights[1].center, lights[1].bottom };
            } else {
                auto ordered = sortCorners(pts);
                return { ordered[0], (ordered[0] + ordered[1]) / 2.0, ordered[1],
                         ordered[2], (ordered[2] + ordered[3]) / 2.0, ordered[3] };
            }
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

    std::vector<cv::Point2f>
    toPtsDebug(const cv::Mat& camera_intrinsic, const cv::Mat& camera_distortion) {
        std::vector<cv::Point2f> image_points;
        const std::vector<cv::Point3f>* model_points;
        static std::vector<cv::Point3f> SMALL_ARMOR_3D_POINTS_BLOCK = {
            { 0, 0.025, -0.066 }, // 左上前
            { 0, -0.025, -0.066 }, // 左下前
            { 0, -0.025, 0.066 }, // 右下前
            { 0, 0.025, 0.066 }, // 右上前
            { 0.015, 0.025, -0.066 }, // 左上后
            { 0.015, -0.025, -0.066 }, // 左下后
            { 0.015, -0.025, 0.066 }, // 右下后
            { 0.015, 0.025, 0.066 }, // 右上后
        };

        static std::vector<cv::Point3f> BIG_ARMOR_3D_POINTS_BLOCK = {
            { 0, 0.025, -0.1125 },     { 0, -0.025, -0.1125 },    { 0, -0.025, 0.1125 },
            { 0, 0.025, 0.1125 },      { 0.015, 0.025, -0.1125 }, { 0.015, -0.025, -0.1125 },
            { 0.015, -0.025, 0.1125 }, { 0.015, 0.025, 0.1125 },
        };

        if (type == "large") {
            model_points = &BIG_ARMOR_3D_POINTS_BLOCK;
        } else if (type == "small") {
            model_points = &SMALL_ARMOR_3D_POINTS_BLOCK;
        }
        Eigen::Matrix3d tf_rot = target_ori.toRotationMatrix();
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

        // 平移向量
        cv::Mat tvec = (cv::Mat_<double>(3, 1) << target_pos.x(), target_pos.y(), target_pos.z());

        // 反投影
        cv::projectPoints(
            *model_points,
            rvec,
            tvec,
            camera_intrinsic,
            camera_distortion,
            image_points
        );
        return image_points;
    }
};
struct Armors {
    std::vector<Armor> armors;
    std::chrono::steady_clock::time_point timestamp;
    std::string frame_id;
    int id;
    Eigen::Vector3d v;
};

} // namespace armor
enum class AutoAimFsm { AIM_WHOLE_CAR_ARMOR, AIM_WHOLE_CAR_CENTER, AIM_SINGLE_ARMOR };
inline std::string auto_aim_fsm_to_string(AutoAimFsm state) {
    switch (state) {
        case AutoAimFsm::AIM_WHOLE_CAR_ARMOR:
            return "AIM_WHOLE_CAR_ARMOR";
        case AutoAimFsm::AIM_WHOLE_CAR_CENTER:
            return "AIM_WHOLE_CAR_CENTER";
        case AutoAimFsm::AIM_SINGLE_ARMOR:
            return "AIM_SINGLE_ARMOR";
        default:
            return "UNKNOWN";
    }
}
class AutoAimFsmController {
public:
    AutoAimFsm fsm_state_ { AutoAimFsm::AIM_SINGLE_ARMOR };
    int overflow_count_ = 0;
    int transfer_thresh_ = 5; // 防抖计数阈值

    // 上下阈值
    double thres_up_1 = 1.0; // SINGLE -> WHOLE_ARMOR
    double thres_down_1 = 0.5; // WHOLE_ARMOR -> SINGLE

    double thres_up_2 = 6.0; // WHOLE_ARMOR -> CENTER
    double thres_down_2 = 5.0; // CENTER -> WHOLE_ARMOR
    void load(const YAML::Node& config) {
        thres_up_1 = config["auto_aim_fsm"]["thres_up_1"].as<double>(1.0);
        thres_down_1 = config["auto_aim_fsm"]["thres_down_1"].as<double>(0.5);
        thres_up_2 = config["auto_aim_fsm"]["thres_up_2"].as<double>(6.0);
        thres_down_2 = config["auto_aim_fsm"]["thres_down_2"].as<double>(5.0);
        transfer_thresh_ = config["auto_aim_fsm"]["transfer_thresh"].as<int>(5.0);
    }
    AutoAimFsmController(
        double up1 = 1.0,
        double down1 = 0.5,
        double up2 = 2.0,
        double down2 = 1.5,
        int transfer = 3
    ):
        thres_up_1(up1),
        thres_down_1(down1),
        thres_up_2(up2),
        thres_down_2(down2),
        transfer_thresh_(transfer) {}
    void update(double v_yaw, bool target_jumped) {
        if (!target_jumped) {
            fsm_state_ = AutoAimFsm::AIM_SINGLE_ARMOR;
            return;
        }
        switch (fsm_state_) {
            case AutoAimFsm::AIM_SINGLE_ARMOR:
                if (std::abs(v_yaw) > thres_up_1)
                    ++overflow_count_;
                else
                    overflow_count_ = 0;

                if (overflow_count_ > transfer_thresh_) {
                    fsm_state_ = AutoAimFsm::AIM_WHOLE_CAR_ARMOR;
                    overflow_count_ = 0;
                }
                break;

            case AutoAimFsm::AIM_WHOLE_CAR_ARMOR:
                if (std::abs(v_yaw) > thres_up_2)
                    ++overflow_count_;
                else if (std::abs(v_yaw) < thres_down_1)
                    ++overflow_count_;
                else
                    overflow_count_ = 0;

                if (overflow_count_ > transfer_thresh_) {
                    if (std::abs(v_yaw) > thres_up_2) {
                        fsm_state_ = AutoAimFsm::AIM_WHOLE_CAR_CENTER;
                    } else {
                        fsm_state_ = AutoAimFsm::AIM_SINGLE_ARMOR;
                    }
                    overflow_count_ = 0;
                }
                break;

            case AutoAimFsm::AIM_WHOLE_CAR_CENTER:
                if (std::abs(v_yaw) < thres_down_2)
                    ++overflow_count_;
                else
                    overflow_count_ = 0;

                if (overflow_count_ > transfer_thresh_) {
                    fsm_state_ = AutoAimFsm::AIM_WHOLE_CAR_ARMOR;
                    overflow_count_ = 0;
                }
                break;

            default:
                fsm_state_ = AutoAimFsm::AIM_SINGLE_ARMOR;
                overflow_count_ = 0;
                break;
        }
    }
};
