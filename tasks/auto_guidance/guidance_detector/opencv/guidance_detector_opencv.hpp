#pragma once
#include "tasks/auto_guidance/guidance_detector/detector_base.hpp"

namespace auto_guidance {
class GuidanceDetectorOpenCV: public detector_base {
public:
    GuidanceDetectorOpenCV(const YAML::Node& config, bool debug);
    ~GuidanceDetectorOpenCV();
    void pushInput(CommonFrame& frame) override;

    void setCallback(DetectorCallback callback) override;
    void processCallback(const CommonFrame& frame);
    DetectorCallback infer_callback_;
    int current_id_ = 0;
    double min_area_ = 100;
    double max_area_ = 10000;
    double min_fill_ratio_ = 0.5;
    double min_aspect_ratio = 0.7;
    bool debug_ = false;
    bool use_gui_ = false;
};
} // namespace auto_guidance
