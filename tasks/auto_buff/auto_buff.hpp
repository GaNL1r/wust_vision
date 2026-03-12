#pragma once
#include "tasks/imodule.hpp"
#include "tasks/type_common.hpp"
#include "wust_vl/video/camera.hpp"
namespace wust_vision {
namespace auto_buff {

    class AutoBuff: public IModule {
    public:
        using Ptr = std::shared_ptr<AutoBuff>;
        AutoBuff(
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
            return std::make_shared<AutoBuff>(config_path, tf_config, camera_info, debug);
        }
        ~AutoBuff();
        void start() override;
        void doDebug() override;
        void pushInput(CommonFrame& frame) override;
        GimbalCmd solve(double bullet_speed) override;
        wust_vl::common::concurrency::MonitoredThread::Ptr getThread() override;
        struct Impl;
        std::unique_ptr<Impl> _impl;
    };
    inline AutoBuff::Ptr toAutoBuff(IModule::Ptr module) {
        return std::dynamic_pointer_cast<AutoBuff>(module);
    }
} // namespace auto_buff
} // namespace wust_vision