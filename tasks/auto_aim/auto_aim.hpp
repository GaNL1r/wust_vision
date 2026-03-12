#pragma once

#include "tasks/auto_aim/armor_tracker/target.hpp"
#include "tasks/imodule.hpp"
#include "tasks/type_common.hpp"
#include "wust_vl/video/camera.hpp"
#include <memory>
namespace wust_vision {
namespace auto_aim {

    class AutoAim: public IModule {
    public:
        using Ptr = std::shared_ptr<AutoAim>;
        AutoAim(
            const std::string& config_path,
            TFConfig::Ptr tf_config,
            const std::pair<cv::Mat, cv::Mat>& camera_info,
            bool debug
        );
        static Ptr create(
            const std::string& config_path,
            TFConfig::Ptr tf_config,
            const std::pair<cv::Mat, cv::Mat>& camera_info,
            bool debug
        ) {
            return std::make_shared<AutoAim>(config_path, tf_config, camera_info, debug);
        }
        ~AutoAim();

        void start() override;
        void doDebug() override;
        void pushInput(CommonFrame& frame) override;
        Target getTarget();
        GimbalCmd solve(double bullet_speed) override;
        wust_vl::common::concurrency::MonitoredThread::Ptr getThread() override;
        struct Impl;
        std::unique_ptr<Impl> _impl;
    };
    inline AutoAim::Ptr toAutoAim(IModule::Ptr module) {
        return std::dynamic_pointer_cast<AutoAim>(module);
    }
} // namespace auto_aim
} // namespace wust_vision