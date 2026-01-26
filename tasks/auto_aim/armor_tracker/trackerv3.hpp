#pragma once

#include "target.hpp"
namespace wust_vision {
namespace auto_aim {
    class Tracker {
    public:
        using Ptr = std::unique_ptr<Tracker>;
        Tracker(wust_vl::common::utils::Parameter::Ptr auto_aim_config_parameter);
        static Ptr create(wust_vl::common::utils::Parameter::Ptr auto_aim_config_parameter) {
            return std::make_unique<Tracker>(auto_aim_config_parameter);
        }
        ~Tracker();
        Target track(const Armors& armors) noexcept;
        int getFoundCount() const noexcept;
        void setFoundCount(int count) noexcept;
        std::chrono::steady_clock::time_point getLastTime() const noexcept;
        struct Impl;
        std::unique_ptr<Impl> _impl;
    };
} // namespace auto_aim
} // namespace wust_vision