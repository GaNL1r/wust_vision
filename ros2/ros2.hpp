#pragma once
#include "ros2_pub_node.hpp"
#include "ros2_sub_node.hpp"
#include <memory>
#include <thread>
class Ros2 {
public:
    Ros2(std::function<void(const geometry_msgs::msg::Twist::SharedPtr)> twist_cb);
    ~Ros2();

private:
    std::shared_ptr<Ros2PubNode> publish2nav_;
    std::shared_ptr<Ros2SubNode> subscribe2nav_;

    std::unique_ptr<std::thread> publish_spin_thread_;
    std::unique_ptr<std::thread> subscribe_spin_thread_;
};