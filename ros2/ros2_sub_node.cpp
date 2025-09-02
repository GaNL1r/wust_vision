#include "ros2_sub_node.hpp"
Ros2SubNode::Ros2SubNode(std::function<void(const geometry_msgs::msg::Twist::SharedPtr)> twist_cb):
    Node("toRos2SubNode") {
    twist_sub_ = this->create_subscription<geometry_msgs::msg::Twist>("cmd_vel", 10, twist_cb);
}
void Ros2SubNode::start() {
    RCLCPP_INFO(this->get_logger(), "nav_subscriber node Starting to spin...");
    rclcpp::spin(this->shared_from_this());
}
