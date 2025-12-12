#pragma once
#include "tasks/auto_guidance/guidance_detector/detector_base.hpp"
#include "tasks/auto_guidance/guidance_detector/green_light_infer.hpp"
#include "wust_vl/ml_net/openvino/openvino_net.hpp"
namespace auto_guidance {
class GuidanceDetectorOpenVino: public detector_base {
public:
    GuidanceDetectorOpenVino(const YAML::Node& config);
    ~GuidanceDetectorOpenVino();
    void pushInput(CommonFrame& frame) override;

    void setCallback(DetectorCallback callback) override;
    void processCallback(const CommonFrame& frame);
    std::string model_path_;
    std::string device_name_;
    float conf_threshold_;
    int top_k_;
    float nms_threshold_;
    std::unique_ptr<ml_net::OpenvinoNet> openvino_net_;
    std::unique_ptr<GreenLightInfer> green_light_infer_;
    DetectorCallback infer_callback_;
    int current_id_ = 0;
};
} // namespace auto_guidance
