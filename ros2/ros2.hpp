#pragma once

#include "rclcpp/rclcpp.hpp"
#include <functional>
#include <memory>
#include <string>
#include <typeindex>
#include <unordered_map>

class Ros2Node: public rclcpp::Node {
public:
    explicit Ros2Node(const std::string& node_name = "ros2_node");
    ~Ros2Node();
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

    // ------------------- Spin -------------------
    void start();

private:
    std::unordered_map<std::string, rclcpp::SubscriptionBase::SharedPtr> subscriptions_;
    std::unordered_map<std::string, std::shared_ptr<void>> publishers_;
    std::unique_ptr<std::thread> spin_thread_;
};
