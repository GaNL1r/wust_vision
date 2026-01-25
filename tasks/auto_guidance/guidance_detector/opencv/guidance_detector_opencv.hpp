#pragma once
#include "tasks/auto_guidance/guidance_detector/detector_base.hpp"
namespace wust_vision {
namespace auto_guidance {
    class GuidanceDetectorOpenCV: public detector_base {
    public:
        GuidanceDetectorOpenCV(const YAML::Node& config, bool debug);
        ~GuidanceDetectorOpenCV();
        void pushInput(CommonFrame& frame) override;

        void setCallback(DetectorCallback callback) override;

    private:
        struct Impl;
        std::unique_ptr<Impl> _impl;
    };
} // namespace auto_guidance
} // namespace wust_vision