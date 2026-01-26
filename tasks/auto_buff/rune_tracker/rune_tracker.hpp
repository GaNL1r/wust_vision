#pragma once
#include "rune_target.hpp"
namespace wust_vision {
namespace auto_buff {
    class RuneTracker {
    public:
        using Ptr = std::unique_ptr<RuneTracker>;
        RuneTracker(wust_vl::common::utils::Parameter::Ptr auto_buff_config_parameter);
        static Ptr create(wust_vl::common::utils::Parameter::Ptr auto_buff_config_parameter) {
            return std::make_unique<RuneTracker>(auto_buff_config_parameter);
        }
        ~RuneTracker();
        auto_buff::RuneTarget track(const auto_buff::RuneFan& fan);

    private:
        struct Impl;
        std::unique_ptr<Impl> _impl;
    };
} // namespace auto_buff
} // namespace wust_vision