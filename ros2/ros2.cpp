#include "ros2.hpp"

Ros2::Ros2(std::function<void(const geometry_msgs::msg::Twist::SharedPtr)> twist_cb) {
    rclcpp::init(0, nullptr);

    publish2nav_ = std::make_shared<Ros2PubNode>();

    subscribe2nav_ = std::make_shared<Ros2SubNode>(twist_cb);

    publish_spin_thread_ = std::make_unique<std::thread>([this]() { publish2nav_->start(); });

    subscribe_spin_thread_ = std::make_unique<std::thread>([this]() { subscribe2nav_->start(); });
}
Ros2::~Ros2() {
    rclcpp::shutdown();
    publish_spin_thread_->join();
    subscribe_spin_thread_->join();
}
