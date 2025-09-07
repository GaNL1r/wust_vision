#include "ros2.hpp"

Ros2Node::Ros2Node(const std::string& node_name) : Node(node_name) {
    
    RCLCPP_INFO(this->get_logger(), "Your env must has been export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp");
}

Ros2Node::~Ros2Node() {
    if (spin_thread_ && spin_thread_->joinable()) {
        rclcpp::shutdown();
        spin_thread_->join();
    }
}

void Ros2Node::start() {
    spin_thread_ = std::make_unique<std::thread>([this]() {
        RCLCPP_INFO(this->get_logger(), "Node [%s] starting to spin...", this->get_name());
        rclcpp::spin(this->shared_from_this());
    });
}
