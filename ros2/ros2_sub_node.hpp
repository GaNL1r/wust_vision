#pragma once
#include "geometry_msgs/msg/twist.hpp"
#include "rclcpp/rclcpp.hpp"
class Ros2SubNode: public rclcpp::Node {
public:
    Ros2SubNode(std::function<void(const geometry_msgs::msg::Twist::SharedPtr)> twist_cb);
    void start();

private:
    rclcpp::Subscription<geometry_msgs::msg::Twist>::SharedPtr twist_sub_;
};