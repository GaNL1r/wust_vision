#pragma once

#include "geometry_msgs/msg/transform_stamped.hpp"
#include "rclcpp/rclcpp.hpp"
#include "tf2_ros/buffer.h"
#include "tf2_ros/transform_listener.h"

#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <typeindex>
#include <unordered_map>

class Ros2Node: public rclcpp::Node {
public:
    explicit Ros2Node(const std::string& node_name = "ros2_node"):
        rclcpp::Node(node_name),
        tf_buffer_(this->get_clock()),
        tf_listener_(tf_buffer_) {}

    ~Ros2Node() {
        stop();
    }

    // ------------------- Subscription -------------------
    template<typename MsgT>
    void add_subscription(
        const std::string& topic_name,
        std::function<void(const typename MsgT::SharedPtr)> callback,
        const rclcpp::QoS& qos = rclcpp::QoS(rclcpp::KeepLast(10))
    ) {
        auto sub = this->create_subscription<MsgT>(topic_name, qos, callback);
        std::lock_guard<std::mutex> lg(map_mutex_);
        subscriptions_[topic_name] = sub;
    }

    // ------------------- Publisher -------------------
    template<typename MsgT>
    void add_publisher(
        const std::string& topic_name,
        const rclcpp::QoS& qos = rclcpp::QoS(rclcpp::KeepLast(10))
    ) {
        auto pub = this->create_publisher<MsgT>(topic_name, qos);
        rclcpp::PublisherBase::SharedPtr base_pub =
            std::static_pointer_cast<rclcpp::PublisherBase>(pub);
        std::lock_guard<std::mutex> lg(map_mutex_);
        publishers_[topic_name] = base_pub;
    }

    template<typename MsgT>
    void publish(const std::string& topic_name, const MsgT& msg) {
        std::lock_guard<std::mutex> lg(map_mutex_);
        auto it = publishers_.find(topic_name);
        if (it != publishers_.end()) {
            auto typed_pub = std::static_pointer_cast<rclcpp::Publisher<MsgT>>(it->second);
            if (typed_pub)
                typed_pub->publish(msg);
        }
    }

    // ------------------- TF Lookup -------------------
    bool lookup_transform(
        const std::string& target_frame,
        const std::string& source_frame,
        geometry_msgs::msg::TransformStamped& out_tf,
        std::chrono::milliseconds timeout = std::chrono::milliseconds(200)
    ) {
        try {
            out_tf =
                tf_buffer_.lookupTransform(target_frame, source_frame, tf2::TimePointZero, timeout);
            return true;
        } catch (const tf2::TransformException& ex) {
            RCLCPP_WARN(
                this->get_logger(),
                "lookup_transform failed %s -> %s: %s",
                source_frame.c_str(),
                target_frame.c_str(),
                ex.what()
            );
            return false;
        }
    }

    // ------------------- Spin -------------------
    void start() {
        auto node = shared_from_this();
        spin_thread_ = std::thread([this, node]() { rclcpp::spin(node); });
    }

    void stop() {
        if (spin_thread_.joinable()) {
            spin_thread_.join();
        }
    }

private:
    std::unordered_map<std::string, rclcpp::SubscriptionBase::SharedPtr> subscriptions_;
    std::unordered_map<std::string, rclcpp::PublisherBase::SharedPtr> publishers_;

    std::thread spin_thread_;
    std::mutex map_mutex_;

    // ----- TF Buffer / Listener -----
    tf2_ros::Buffer tf_buffer_;
    tf2_ros::TransformListener tf_listener_;
};
