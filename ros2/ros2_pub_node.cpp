#include "ros2_pub_node.hpp"
Ros2PubNode::Ros2PubNode(): Node("toRos2PubNode") {}
void Ros2PubNode::start() {
    RCLCPP_INFO(this->get_logger(), "auto_aim_target_pos_publisher node starting to spin...");
    rclcpp::spin(this->shared_from_this());
}