#pragma once
#include "tasks/auto_buff/rune_tracker/rune_target.hpp"
#include "tasks/type_common.hpp"
namespace wust_vision {
namespace auto_buff {
    class Aimer {
    public:
        using Ptr = std::unique_ptr<Aimer>;
        Aimer(wust_vl::common::utils::Parameter::Ptr auto_buff_config_parameter);
        static Ptr create(wust_vl::common::utils::Parameter::Ptr auto_buff_config_parameter) {
            return std::make_unique<Aimer>(auto_buff_config_parameter);
        }
        ~Aimer();
        GimbalCmd aim(const auto_buff::RuneTarget& target, double bullet_speed);

    private:
        struct Impl;
        std::unique_ptr<Impl> _impl;
    };

} // namespace auto_buff
} // namespace wust_vision