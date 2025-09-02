#pragma once
#include "geometry_msgs/msg/twist.hpp"
#include "rclcpp/rclcpp.hpp"
class Ros2PubNode: public rclcpp::Node {
public:
    Ros2PubNode();
    void start();

private:
};