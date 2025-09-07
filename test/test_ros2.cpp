#include "geometry_msgs/msg/twist.hpp"
#include "rclcpp/rclcpp.hpp"
#include "ros2/ros2.hpp"
#include <wust_vl/common/utils/signal.hpp>
void TwistCb(const geometry_msgs::msg::Twist::SharedPtr msg) {
    std::cout << msg->linear.x << std::endl;
}
int main() {
    Ros2Node ros2;
    ros2.add_subscription<geometry_msgs::msg::Twist>(
        "cmd_vel",
        std::bind(TwistCb, std::placeholders::_1),
        rclcpp::QoS(10)
    );
    SignalHandler sig;
    sig.start([&] { return 0; });
    return 0;
}