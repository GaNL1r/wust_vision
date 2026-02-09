#pragma once
#include "tasks/auto_aim/armor_tracker/target.hpp"
#include "tasks/auto_aim/auto_aim_fsm.hpp"
#include <memory>
namespace wust_vision::auto_aim {
class VeryAimer {
public:
    using Ptr = std::unique_ptr<VeryAimer>;
    VeryAimer(wust_vl::common::utils::Parameter::Ptr auto_aim_config_parameter);
    static Ptr create(wust_vl::common::utils::Parameter::Ptr auto_aim_config_parameter) {
        return std::make_unique<VeryAimer>(auto_aim_config_parameter);
    };
    ~VeryAimer();
    [[nodiscard]] GimbalCmd
    veryAim(Target target, double bullet_speed, const AutoAimFsm& auto_aim_fsm);
    struct Impl;
    std::unique_ptr<Impl> _impl;
};
} // namespace wust_vision::auto_aim