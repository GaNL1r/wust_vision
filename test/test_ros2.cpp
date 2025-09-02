#include "rclcpp/rclcpp.hpp"
#include "ros2/ros2.hpp"
#include <wust_vl/common/utils/signal.hpp>
void TwistCb(const geometry_msgs::msg::Twist::SharedPtr msg) {
    std::cout << msg->linear.x << std::endl;
}
int main() {
    Ros2 ros2(std::bind(TwistCb, std::placeholders::_1));
    SignalHandler sig;
    sig.start([&] { return 0; });
    return 0;
}