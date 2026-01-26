#pragma once
#include "tasks/type_common.hpp"
#include "tasks/utils.hpp"
namespace wust_vision {

namespace auto_aim {
    constexpr double SMALL_ARMOR_WIDTH = 133.0 / 1000.0; // 135
    constexpr double SMALL_ARMOR_HEIGHT = 50.0 / 1000.0; // 55
    constexpr double LARGE_ARMOR_WIDTH = 225.0 / 1000.0;
    constexpr double LARGE_ARMOR_HEIGHT = 50.0 / 1000.0; // 55

    constexpr double FIFTTEN_DEGREE_RAD = 15 * CV_PI / 180;
    struct Light: public cv::RotatedRect {
        Light() = default;

        explicit Light(const std::vector<cv::Point>& contour);

        void addOffset(const cv::Point2f& offset) noexcept;
        void transform(const Eigen::Matrix<float, 3, 3>& transform_matrix) noexcept;

        cv::Point2f top, bottom;
        int color = 0;
        cv::Point2f axis;
        double length = 0;
        double width = 0;
        float tilt_angle = 0;
    };

    enum class ArmorColor { BLUE = 0, RED, NONE, PURPLE };

    int formArmorColor(const ArmorColor& color) noexcept;
    enum class ArmorNumber { SENTRY = 0, NO1, NO2, NO3, NO4, NO5, OUTPOST, BASE, UNKNOWN };
    std::ostream& operator<<(std::ostream& os, const ArmorNumber& number) noexcept;

    int formArmorNumber(const ArmorNumber& number) noexcept;
    std::string armorNumberToString(const ArmorNumber& num) noexcept;
    ArmorNumber armorNumberFromString(const std::string& s) noexcept;
    int retypetotracker(const ArmorNumber& a) noexcept;
    bool isSameTarget(const ArmorNumber& a, const ArmorNumber& b) noexcept;
    enum class ArmorsNum { NORMAL_4 = 4, OUTPOST_3 = 3 };

    enum class ArmorType { SMALL, LARGE, INVALID };
    std::string armorTypeToString(const ArmorType& type) noexcept;

    struct ArmorObject {
        ArmorColor color;
        ArmorNumber number;
        float prob;
        std::vector<cv::Point2f> pts;
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
        ArmorType type;
        static constexpr const int N_LANDMARKS = 6;
        static constexpr const int N_LANDMARKS_2 = N_LANDMARKS * 2;

        template<typename PointType>
        std::vector<PointType> static buildObjectPoints(const double& w, const double& h) noexcept;
        template<typename IDType>
        std::vector<std::pair<IDType, IDType>> static buildSymPairs() noexcept;
        std::vector<cv::Point2f> toPts() const noexcept;
        bool checkOkptsRight(double max_error) const noexcept;
        std::array<cv::Point2f, 4> sortCorners(const std::vector<cv::Point2f>& pts) const noexcept;

        // Landmarks start from bottom left in clockwise order
        std::vector<cv::Point2f> landmarks() const noexcept;
        void addOffset(const cv::Point2f& offset) noexcept {
            for (auto& pt: pts) {
                pt += offset;
            }
            for (auto& l: lights) {
                l.addOffset(offset);
            }
        }
        void transform(const Eigen::Matrix<float, 3, 3>& transform_matrix) noexcept {
            for (auto& l: lights) {
                l.transform(transform_matrix);
            }
            auto map_point = [&](float x, float y) -> cv::Point2f {
                Eigen::Vector3f pt(x, y, 1.f);
                Eigen::Vector3f tr = transform_matrix * pt;
                return { tr(0), tr(1) };
            };
            for (auto& pt: pts) {
                pt = map_point(pt.x, pt.y);
            }
        }
        ArmorObject(const Light& l1, const Light& l2);
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

    struct Armor {
    public:
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
        int id = -1;
        std::vector<cv::Point2f> toPtsDebug(
            const cv::Mat& camera_intrinsic,
            const cv::Mat& camera_distortion
        ) const noexcept;
    };
    struct Armors {
    public:
        std::vector<Armor> armors;
        std::chrono::steady_clock::time_point timestamp;
        int id;
        Eigen::Vector3d v;
    };
    static constexpr double outpost_v_yaw = 0.8 * M_PI;
    static constexpr double DZ_1 = 0.1;
    static constexpr double DZ_2 = -0.1;
    static constexpr double DZ_3 = 0.2;
    static constexpr double DZ_4 = -0.2;
    static constexpr std::array<double, 4> outpostDZ = { DZ_1, DZ_2, DZ_3, DZ_4 };
    inline double outpost_diff_from_id(int id) noexcept {
        switch (id) {
            case 1:
                return DZ_1;
            case 2:
                return DZ_2;
            case 3:
                return DZ_3;
            case 4:
                return DZ_4;
            default:
                return 0.0;
        }
    }

    inline int quantize_outpost_diff(double dz) noexcept {
        static constexpr double candidates[] = { DZ_1, DZ_2, DZ_3, DZ_4 };
        int best_id = 1;
        double min_diff = std::abs(dz - candidates[0]);
        for (int i = 1; i < 4; ++i) {
            double diff = std::abs(dz - candidates[i]);
            if (diff < min_diff) {
                min_diff = diff;
                best_id = i + 1; // ID 从 1 开始
            }
        }
        return best_id;
    }

    void transformArmorData(Armors& armors, Eigen::Matrix4d T_camera_to_odom) noexcept;
    void transformArmorData(Armor& armor, const Eigen::Matrix4d& T_camera_to_odom) noexcept;

} // namespace auto_aim
} // namespace wust_vision