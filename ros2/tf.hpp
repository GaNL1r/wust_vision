#pragma once
#include "rclcpp/rclcpp.hpp"
#include <Eigen/Dense>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>
#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_broadcaster.h>
#include <tf2_ros/transform_listener.h>
template<typename PublisherT>
inline bool publisherSubscribed(const PublisherT& publisher) noexcept {
    return publisher && publisher->get_subscription_count() > 0;
}
inline Eigen::Isometry3f tf2ToEigen(const geometry_msgs::msg::TransformStamped& tf) noexcept {
    Eigen::Isometry3f T = Eigen::Isometry3f::Identity();

    const auto& t = tf.transform.translation;
    T.translation() =
        Eigen::Vector3f(static_cast<float>(t.x), static_cast<float>(t.y), static_cast<float>(t.z));

    const auto& q = tf.transform.rotation;

    Eigen::Quaternionf Q(
        static_cast<float>(q.w),
        static_cast<float>(q.x),
        static_cast<float>(q.y),
        static_cast<float>(q.z)
    );

    Q.normalize();

    T.linear() = Q.toRotationMatrix();

    return T;
}
class TF {
public:
    using Ptr = std::shared_ptr<TF>;
    TF(rclcpp::Node& n) {
        tf_buffer_ = std::make_unique<tf2_ros::Buffer>(n.get_clock());
        tf_listener_ = std::make_unique<tf2_ros::TransformListener>(*tf_buffer_);
        tf_broadcaster_ = std::make_unique<tf2_ros::TransformBroadcaster>(n);
        node_ = &n;
    }
    static Ptr create(rclcpp::Node& n) {
        return std::make_shared<TF>(n);
    }
    std::optional<Eigen::Isometry3f>
    getTransform(const std::string& target, const std::string& source, rclcpp::Time t)
        const noexcept {
        Eigen::Isometry3f T_out = Eigen::Isometry3f::Identity();
        try {
            geometry_msgs::msg::TransformStamped tf_msg =
                tf_buffer_->lookupTransform(target, source, t, rclcpp::Duration::from_seconds(0.1));

            T_out = tf2ToEigen(tf_msg);
            return T_out;
        } catch (tf2::TransformException& ex) {
            RCLCPP_WARN(rclcpp::get_logger("tf"), "TF lookup failed: %s", ex.what());
            return std::nullopt;
        }
        return T_out;
    }

    void publishTransform(
        const Eigen::Isometry3d& transform,
        const std::string& parent_frame,
        const std::string& child_frame,
        const rclcpp::Time& stamp
    ) const noexcept {
        geometry_msgs::msg::TransformStamped tmsg;
        tmsg.header.stamp = stamp;
        tmsg.header.frame_id = parent_frame;
        tmsg.child_frame_id = child_frame;
        const Eigen::Vector3d tr = transform.translation();
        const Eigen::Quaterniond q(transform.rotation());
        tmsg.transform.translation.x = tr.x();
        tmsg.transform.translation.y = tr.y();
        tmsg.transform.translation.z = tr.z();
        tmsg.transform.rotation.x = q.x();
        tmsg.transform.rotation.y = q.y();
        tmsg.transform.rotation.z = q.z();
        tmsg.transform.rotation.w = q.w();
        tf_broadcaster_->sendTransform(tmsg);
    }
    rclcpp::Node* node_;
    std::unique_ptr<tf2_ros::Buffer> tf_buffer_;
    std::unique_ptr<tf2_ros::TransformListener> tf_listener_;
    std::unique_ptr<tf2_ros::TransformBroadcaster> tf_broadcaster_;
};
