#pragma once
#include "tasks/debug.hpp"
#include "tasks/type_common.hpp"
#include "wust_vl/video/camera.hpp"
namespace wust_vision {
namespace auto_buff {
    struct AutoBuffShared {
        std::shared_ptr<wust_vl::common::utils::MotionBufferGeneric<Motion, 1024>> motion_buffer;
        double bullet_speed;
        bool is_rune_big;
        double communication_delay_μs;
        AutoBuffShared(
            std::shared_ptr<wust_vl::common::utils::MotionBufferGeneric<Motion, 1024>> mb,
            double bullet_speed,
            double communication_delay_μs
        ) {
            motion_buffer = mb;
            this->bullet_speed = bullet_speed;
            this->communication_delay_μs = communication_delay_μs;
        }
    };
    class AutoBuff {
    public:
        AutoBuff();
        ~AutoBuff();
        bool init(
            const std::string& config_path,
            int& use_detect_ncnn_count,
            TFConfig::Ptr tf_config,
            const std::pair<cv::Mat, cv::Mat>& camera_info
        );
        void start();
        void pushInput(CommonFrame& frame, bool is_big);
        void setDebug(bool debug);
        DebugRune getDebugFrame();
        GimbalCmd solve();
        void setShared(std::shared_ptr<AutoBuffShared> shared);
        bool isActive();
        void processingWait();
        void processingUp();
        void
        autoExposureControl(const cv::Mat& frame, std::shared_ptr<wust_vl::video::Camera> camera);
        struct Impl;
        std::unique_ptr<Impl> _impl;
    };
} // namespace auto_buff
} // namespace wust_vision