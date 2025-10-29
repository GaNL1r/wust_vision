#include "ros2.hpp"

void Ros2Node::start() {
    auto self = this->shared_from_this();
    spin_thread_ = std::thread([self]() {
        RCLCPP_INFO(self->get_logger(), "Node [%s] starting to spin...", self->get_name());
        rclcpp::spin(self);
    });
}
bool Ros2Node::lookup_transform(
    const std::string& target_frame,
    const std::string& source_frame,
    geometry_msgs::msg::TransformStamped& out_transform,
    const rclcpp::Time& time,
    double timeout_sec
) {
    try {
        out_transform = tf_buffer_.lookupTransform(
            target_frame,
            source_frame,
            time,
            std::chrono::duration<double>(timeout_sec)
        );
        return true;
    } catch (const tf2::TransformException& ex) {
        RCLCPP_WARN(this->get_logger(), "TF lookup failed: %s", ex.what());
        return false;
    }
}