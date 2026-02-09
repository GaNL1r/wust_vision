#pragma once
#include <memory>
namespace wust_vl::common::utils {
class Parameter;
}
using wust_vlParamterPtr = std::shared_ptr<wust_vl::common::utils::Parameter>;
namespace wust_vision {
struct GimbalCmd;
}
namespace wust_vision::auto_aim {
enum class AutoAimFsm;

class Target;
class VeryAimer {
public:
    using Ptr = std::unique_ptr<VeryAimer>;
    VeryAimer(wust_vlParamterPtr auto_aim_config_parameter);
    static Ptr create(wust_vlParamterPtr auto_aim_config_parameter) {
        return std::make_unique<VeryAimer>(auto_aim_config_parameter);
    };
    ~VeryAimer();
    [[nodiscard]] GimbalCmd
    veryAim(Target target, double bullet_speed, const AutoAimFsm& auto_aim_fsm);
    struct Impl;
    std::unique_ptr<Impl> _impl;
};
} // namespace wust_vision::auto_aim