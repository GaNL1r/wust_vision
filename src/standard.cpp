#include "tasks/imodule.hpp"
#include "tasks/utils/config.hpp"
#include "tasks/utils/main_base.hpp"
#include "tasks/vision_base.hpp"
ENABLE_BACKWARD()
namespace wust_vision {
class vision: public VisionBase<InfantryMode> {
public:
    vision(): VisionBase(COMMON_CONFIG, CAMERA_CONFIG, AUTO_AIM_CONFIG, AUTO_BUFF_CONFIG) {}
    bool init(bool debug_mode) {
        if (!VisionBase::init(debug_mode)) {
            return false;
        }
        auto auto_aim =
            auto_aim::AutoAim::create(auto_aim_config_, tf_config_, camera_info_, debug_mode);
        modules_.emplace(InfantryMode::AttackMode::ARMOR, auto_aim);
        modules_.emplace(InfantryMode::AttackMode::UNKNOWN, auto_aim);
        auto auto_buff =
            auto_buff::AutoBuff::create(auto_buff_config_, tf_config_, camera_info_, debug_mode);
        modules_.emplace(InfantryMode::AttackMode::BIG_RUNE, auto_buff);
        modules_.emplace(InfantryMode::AttackMode::SMALL_RUNE, auto_buff);
        return true;
    }
};
} // namespace wust_vision

VISION_MAIN(wust_vision::vision)