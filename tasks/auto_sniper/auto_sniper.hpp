#pragma once
#include "tasks/imodule.hpp"
#include "tasks/type_common.hpp"
#include <memory>
#include <rclcpp/node.hpp>
namespace wust_vision {
namespace auto_sniper {
    class AutoSniper: public IModule {
    public:
        using Ptr = std::shared_ptr<AutoSniper>;
        AutoSniper(
            rclcpp::Node& node,
            std::shared_ptr<wust_vl::common::utils::MotionBufferGeneric<Motion, 1024>> motion_buffer
        );
        static Ptr create(
            rclcpp::Node& node,
            std::shared_ptr<wust_vl::common::utils::MotionBufferGeneric<Motion, 1024>> motion_buffer
        ) {
            return std::make_shared<AutoSniper>(node, motion_buffer);
        }
        ~AutoSniper();
        void start() override;
        void doDebug() override;
        void pushInput(CommonFrame& frame) override;
        GimbalCmd solve(double bullet_speed) override;
        wust_vl::common::concurrency::MonitoredThread::Ptr getThread() override;
        struct Impl;
        std::unique_ptr<Impl> _impl;
    };
} // namespace auto_sniper

} // namespace wust_vision