#pragma once
#include "tasks/auto_aim/armor_control/very_aimer.hpp"
#include "tasks/type_common.hpp"
#include <memory>
#include <wust_vl/common/utils/motion_buffer.hpp>
#include <yaml-cpp/node/node.h>
namespace wust_vision {
namespace auto_aim {
    class ArmorOmni {
    public:
        struct Ctx {
            std::shared_ptr<wust_vl::common::utils::MotionBufferGeneric<CarMotion, 1024>>
                car_motion_buffer;
            std::shared_ptr<wust_vl::common::utils::MotionBufferGeneric<BigYaw, 1024>>
                big_yaw_motion_buffer;
            VeryAimer::Ptr very_aimer;
        };
        static constexpr const char* OMNI_CONFIG = "config/omni/omni.yaml";
        using Ptr = std::unique_ptr<ArmorOmni>;
        ArmorOmni(bool detect_color_init, const Ctx& ctx);
        static Ptr create(bool detect_color_init, const Ctx& ctx) {
            return std::make_unique<ArmorOmni>(detect_color_init, ctx);
        }
        ~ArmorOmni();
        void start();
        void setDetectColor(bool flag);
        void updateMainTracking(bool flag);
        int getBestTarget();
        GimbalCmd solve(double bullet_speed);

    private:
        struct Impl;
        std::unique_ptr<Impl> _impl;
    };
} // namespace auto_aim
} // namespace wust_vision
