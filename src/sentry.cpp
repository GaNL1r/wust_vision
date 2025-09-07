#include "3rdparty/backward-cpp/backward.hpp"
#include "geometry_msgs/msg/twist.hpp"
#include "ros2/ros2.hpp"
#include "tasks/vision_base.hpp"
#define COMMON_CONFIG "config/common.yaml"
#define CAMERA_CONFIG "config/camera.yaml"
#define AUTO_AIM_CONFIG "config/auto_aim.yaml"
#define AUTO_BUFF_CONFIG "config/auto_buff.yaml"

namespace backward {
backward::SignalHandling sh;
}
class vision: public VisionBase {
public:
    vision(): VisionBase(COMMON_CONFIG, CAMERA_CONFIG, AUTO_AIM_CONFIG, AUTO_BUFF_CONFIG) {}
    void init() {
        rclcpp::init(0, nullptr);
        VisionBase::init();
        ros2_ = std::make_shared<Ros2Node>("vison_node");
        ros2_->add_subscription<geometry_msgs::msg::Twist>(
            "cmd_vel",
            std::bind(&vision::TwistCb, this, std::placeholders::_1),
            rclcpp::QoS(10)
        );
    }
    void start() {
        VisionBase::start();
        ros2_->start();
    }

    void TwistCb(const geometry_msgs::msg::Twist::SharedPtr msg) {
        NavRobotCmdData send_data;
        send_data.cmd_ID = ID_NAV_CMD;
        send_data.check = true;
        send_data.time_stamp =
            static_cast<uint32_t>(std::chrono::duration_cast<std::chrono::milliseconds>(
                                      time_utils::now().time_since_epoch()
            )
                                      .count());
        send_data.vx = msg->linear.x;
        send_data.vy = msg->linear.y;
        send_data.wz = msg->angular.z;
        if (serial_) {
            serial_->write(std::move(toVector(send_data)));
        }
    }
    std::shared_ptr<Ros2Node> ros2_;
};
VISION_MAIN(vision)