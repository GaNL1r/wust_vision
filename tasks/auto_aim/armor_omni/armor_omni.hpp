#pragma once
#include <memory>
#include <yaml-cpp/node/node.h>
namespace wust_vision {
namespace auto_aim {
    class ArmorOmni {
    public:
        using Ptr = std::unique_ptr<ArmorOmni>;
        ArmorOmni(const YAML::Node& config, bool detect_color_init);
        static Ptr create(const YAML::Node& config, bool detect_color_init) {
            return std::make_unique<ArmorOmni>(config, detect_color_init);
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
