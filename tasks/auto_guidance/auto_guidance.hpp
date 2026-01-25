#pragma once
#include "debug.hpp"
#include "tasks/type_common.hpp"
namespace wust_vision {
namespace auto_guidance {
    class AutoGuidance {
    public:
        static inline std::unique_ptr<AutoGuidance> create() {
            return std::make_unique<AutoGuidance>();
        }
        AutoGuidance();
        ~AutoGuidance();
        void init(const YAML::Node& config, const std::pair<cv::Mat, cv::Mat>& camera_info);
        void start();
        void pushInput(CommonFrame& frame);
        void setDebug(bool debug);
        GuidanceTarget getTarget();
        AutoGuidanceDebug getDebug();
        struct Impl;
        std::unique_ptr<Impl> _impl;
    };

} // namespace auto_guidance
} // namespace wust_vision