#include "ros2/ros2.hpp"

int main(int argc, char** argv) {
    rclcpp::init(argc, argv);

    {
        auto node = rclcpp::Node::make_shared("wust_vision_global_node");
        rclcpp::spin(node);
        // rclcpp::shutdown();
    }

    return 0;
}
