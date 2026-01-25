#pragma once
#include "Eigen/Dense"
#include "tasks/type_common.hpp"
#include <yaml-cpp/yaml.h>
namespace wust_vision {
namespace auto_offset {
    class AutoOffset {
    public:
        AutoOffset();
        ~AutoOffset();
        bool init(const YAML::Node& config);
        struct Impl;
        std::unique_ptr<Impl> _impl;
    };
} // namespace auto_offset
} // namespace wust_vision