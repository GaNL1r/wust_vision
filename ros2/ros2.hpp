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
#include <unordered_map>

class Ros2Node: public rclcpp::Node {
public:
    explicit Ros2Node(const std::string& node_name = "ros2_node"):
        Node(node_name),
        tf_buffer_(this->get_clock()),
        tf_listener_(tf_buffer_) {}

    ~Ros2Node() {
        rclcpp::shutdown();
        if (spin_thread_.joinable()) {
            spin_thread_.join();
        }
    }
    void shutdown() {
        if (spin_thread_.joinable()) {
            spin_thread_.join();
        }
        subscriptions_.clear();
        publishers_.clear();
    }

    // ------------------- 订阅 -------------------
    template<typename MsgT>
    void add_subscription(
        const std::string& topic_name,
        std::function<void(const typename MsgT::SharedPtr)> callback,
        const rclcpp::QoS& qos = rclcpp::QoS(rclcpp::KeepLast(10))
    ) {
        auto sub = this->create_subscription<MsgT>(topic_name, qos, callback);
        subscriptions_[topic_name] = sub;
    }

    // ------------------- 发布 -------------------
    template<typename MsgT>
    void add_publisher(
        const std::string& topic_name,
        const rclcpp::QoS& qos = rclcpp::QoS(rclcpp::KeepLast(10))
    ) {
        auto pub = this->create_publisher<MsgT>(topic_name, qos);
        publishers_[topic_name] = pub;
    }

    template<typename MsgT>
    void publish(const std::string& topic_name, const MsgT& msg) {
        auto it = publishers_.find(topic_name);
        if (it != publishers_.end()) {
            auto pub = std::static_pointer_cast<rclcpp::Publisher<MsgT>>(it->second);
            pub->publish(msg);
        } else {
            RCLCPP_WARN(
                this->get_logger(),
                "Publisher for topic [%s] not found",
                topic_name.c_str()
            );
        }
    }

    bool lookup_transform(
        const std::string& target_frame,
        const std::string& source_frame,
        geometry_msgs::msg::TransformStamped& out_transform,
        const rclcpp::Time& time = rclcpp::Time(0),
        double timeout_sec = 0.1
    );

    // ------------------- Spin -------------------
    void start();

private:
    bool runuing = false;
    std::unordered_map<std::string, rclcpp::SubscriptionBase::SharedPtr> subscriptions_;
    std::unordered_map<std::string, std::shared_ptr<void>> publishers_;
    std::thread spin_thread_;

    // TF
    tf2_ros::Buffer tf_buffer_;
    tf2_ros::TransformListener tf_listener_;
};

// ------------------- 全局上下文 -------------------
class Ros2Context {
public:
    static std::shared_ptr<Ros2Node> get_node(const std::string& node_name = "ros2_global_node") {
        std::lock_guard<std::mutex> lock(mutex_);
        if (!instance_) {
            if (!rclcpp::ok()) {
                int argc = 0;
                char** argv = nullptr;
                rclcpp::init(argc, argv);
            }
            instance_ = std::make_shared<Ros2Node>(node_name);
        }
        return instance_;
    }

    static void shutdown() {
        std::lock_guard<std::mutex> lock(mutex_);
        if (instance_) {
            instance_.reset();
            if (rclcpp::ok()) {
                rclcpp::shutdown();
            }
        }
    }

private:
    inline static std::shared_ptr<Ros2Node> instance_ = nullptr;
    inline static std::mutex mutex_;
};
