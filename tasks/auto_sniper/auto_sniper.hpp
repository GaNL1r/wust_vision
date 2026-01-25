#pragma once
#include "Eigen/Dense"
#include "tasks/type_common.hpp"
#include <yaml-cpp/yaml.h>
namespace wust_vision {
namespace auto_sniper {
    class AutoSniper {
    public:
        AutoSniper();
        ~AutoSniper();
        bool init(const YAML::Node& config);
        void updatePosBulletSpeed(const Eigen::Vector3d& pos, double bullet_speed);
        GimbalCmd solve(double dt_ms);
        struct Impl;
        std::unique_ptr<Impl> _impl;
    };
} // namespace auto_sniper
} // namespace wust_vision