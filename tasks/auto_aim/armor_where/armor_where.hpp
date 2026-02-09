
#pragma once
#include "tasks/auto_aim/type.hpp"
namespace wust_vision {
namespace auto_aim {
    class ArmorWhere {
    public:
        using Ptr = std::unique_ptr<ArmorWhere>;
        ArmorWhere(const YAML::Node& config, const std::pair<cv::Mat, cv::Mat>& camera_info);
        static Ptr
        create(const YAML::Node& config, const std::pair<cv::Mat, cv::Mat>& camera_info) {
            return std::make_unique<ArmorWhere>(config, camera_info);
        }
        ~ArmorWhere();
        std::vector<Armor> where(
            const std::vector<ArmorObject>& armors,
            Eigen::Matrix4d T_camera_to_odom
        ) const noexcept;

    private:
        struct Impl;
        std::unique_ptr<Impl> _impl;
    };

} // namespace auto_aim
} // namespace wust_vision