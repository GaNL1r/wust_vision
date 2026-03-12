#include "ros2/ros2.hpp"
#include "tasks/auto_sniper/auto_sniper.hpp"
#include "tasks/type_common.hpp"
#include "tasks/utils/config.hpp"
#include "tasks/utils/main_base.hpp"
#include "tasks/vision_base.hpp"
ENABLE_BACKWARD()

namespace wust_vision {
class vision: public VisionBase<HeroMode> {
public:
    vision(): VisionBase(COMMON_CONFIG, CAMERA_CONFIG, AUTO_AIM_CONFIG, AUTO_BUFF_CONFIG) {}

    bool init(bool debug_mode) {
        VisionBase::init(debug_mode);
        auto auto_aim =
            auto_aim::AutoAim::create(auto_aim_config_, tf_config_, camera_info_, debug_mode);
        modules_.emplace(HeroMode::AttackMode::ARMOR, auto_aim);
        modules_.emplace(HeroMode::AttackMode::UNKNOWN, auto_aim);

        camera_->setFrameCallback(std::bind(&vision::frameCallback, this, std::placeholders::_1));
        rclcpp::init(0, nullptr);
        ros2_ = std::make_shared<Ros2Node>("vison_node");
        auto auto_sniper = auto_sniper::AutoSniper::create(*ros2_);
        modules_.emplace(HeroMode::AttackMode::SNIPER, auto_sniper);
        return true;
    }

    void start() {
        VisionBase::start();
        if (timer_) {
            auto timercallback = std::bind(&vision::timerCallback, this, std::placeholders::_1);
            double rate_hz = control_config_->control_rate_param.get();
            timer_->start(rate_hz, timercallback);
        }
        ros2_->start();
    }

    std::shared_ptr<Ros2Node> ros2_;
};
} // namespace wust_vision
VISION_MAIN(wust_vision::vision)