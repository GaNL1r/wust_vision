#pragma once

#include "geometry_msgs/msg/transform_stamped.hpp"
#include "rclcpp/rclcpp.hpp"
#include "tf2_ros/buffer.h"
#include "tf2_ros/transform_broadcaster.h"
#include "tf2_ros/transform_listener.h"

#include <chrono>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <typeindex>
#include <unordered_map>
#include <vector>

class Ros2Node: public rclcpp::Node {
public:
    explicit Ros2Node(const std::string& node_name = "vision"):
        Node(node_name),
        tf_buffer_(this->get_clock()),
        tf_listener_(tf_buffer_),
        tf_broadcaster_(std::make_shared<tf2_ros::TransformBroadcaster>(this)) {}
    using Ptr = std::shared_ptr<Ros2Node>;

    static Ptr instance() {
        static Ptr inst = std::make_shared<Ros2Node>();
        return inst;
    }
    ~Ros2Node() {
        stop();
    }

    template<typename MsgT>
    void add_subscription(
        const std::string& topic_name,
        std::function<void(const typename MsgT::SharedPtr)> callback,
        const rclcpp::QoS& qos = rclcpp::QoS(rclcpp::KeepLast(10))
    ) {
        auto sub = this->create_subscription<MsgT>(topic_name, qos, callback);
        std::lock_guard<std::mutex> lock(map_mutex_);
        subscriptions_[topic_name] = sub;
    }

    template<typename MsgT>
    void add_publisher(
        const std::string& topic_name,
        const rclcpp::QoS& qos = rclcpp::QoS(rclcpp::KeepLast(10))
    ) {
        auto pub = this->create_publisher<MsgT>(topic_name, qos);
        std::lock_guard<std::mutex> lock(map_mutex_);
        publishers_[topic_name] = pub;
    }

    template<typename MsgT>
    void publish(const std::string& topic_name, const MsgT& msg) {
        std::lock_guard<std::mutex> lock(map_mutex_);
        auto it = publishers_.find(topic_name);
        if (it != publishers_.end()) {
            auto typed_pub = std::dynamic_pointer_cast<rclcpp::Publisher<MsgT>>(it->second);
            if (typed_pub)
                typed_pub->publish(msg);
        }
    }

    template<typename Rep, typename Period>
    void
    add_timer(const std::chrono::duration<Rep, Period>& interval, std::function<void()> callback) {
        auto timer = this->create_wall_timer(interval, callback);
        std::lock_guard<std::mutex> lock(map_mutex_);
        timers_.push_back(timer);
    }

    bool lookup_transform(
        const std::string& target_frame,
        const std::string& source_frame,
        geometry_msgs::msg::TransformStamped& out_tf,
        std::chrono::milliseconds timeout = std::chrono::milliseconds(200)
    ) const {
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

    void broadcast_tf(const geometry_msgs::msg::TransformStamped& tf_msg) {
        tf_broadcaster_->sendTransform(tf_msg);
    }

    template<typename Rep, typename Period>
    void broadcast_tf_periodic(
        const geometry_msgs::msg::TransformStamped& tf_msg,
        const std::chrono::duration<Rep, Period>& interval
    ) {
        add_timer(interval, [this, tf_msg]() { broadcast_tf(tf_msg); });
    }

    void start() {
        if (!spin_thread_.joinable()) {
            auto node = shared_from_this();
            spin_thread_ = std::thread([node]() { rclcpp::spin(node); });
        }
    }

    void stop() {
        rclcpp::shutdown();
        if (spin_thread_.joinable()) {
            spin_thread_.join();
        }
    }

private:
    mutable std::mutex map_mutex_;

    std::unordered_map<std::string, rclcpp::SubscriptionBase::SharedPtr> subscriptions_;
    std::unordered_map<std::string, rclcpp::PublisherBase::SharedPtr> publishers_;
    std::vector<rclcpp::TimerBase::SharedPtr> timers_;

    std::thread spin_thread_;

    tf2_ros::Buffer tf_buffer_;
    tf2_ros::TransformListener tf_listener_;
    std::shared_ptr<tf2_ros::TransformBroadcaster> tf_broadcaster_;
};