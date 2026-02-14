#pragma once
#include <memory>
#include <yaml-cpp/node/node.h>
namespace wust_vision {
namespace auto_aim {
    class ArmorOmni {
    public:
        using Ptr = std::unique_ptr<ArmorOmni>;
        ArmorOmni(bool detect_color_init);
        static Ptr create(bool detect_color_init) {
            return std::make_unique<ArmorOmni>(detect_color_init);
        }
        ~ArmorOmni();
        void start();
        void setDetectColor(bool flag);
        void updateMainTracking(bool flag);
        int getBestTarget();

    private:
        struct Impl;
        std::unique_ptr<Impl> _impl;
    };
} // namespace auto_aim
} // namespace wust_vision
