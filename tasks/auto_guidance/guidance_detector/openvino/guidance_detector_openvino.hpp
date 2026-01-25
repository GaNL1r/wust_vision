#pragma once
#include "tasks/auto_guidance/guidance_detector/detector_base.hpp"
#include "tasks/auto_guidance/guidance_detector/green_light_infer.hpp"
#include "wust_vl/ml_net/openvino/openvino_net.hpp"
namespace wust_vision {
namespace auto_guidance {
    class GuidanceDetectorOpenVino: public detector_base {
    public:
        GuidanceDetectorOpenVino(const YAML::Node& config);
        ~GuidanceDetectorOpenVino();
        void pushInput(CommonFrame& frame) override;

        void setCallback(DetectorCallback callback) override;

    private:
        struct Impl;
        std::unique_ptr<Impl> _impl;
    };
} // namespace auto_guidance
} // namespace wust_vision