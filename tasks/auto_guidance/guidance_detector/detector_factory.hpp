#pragma once

#include "tasks/auto_guidance/guidance_detector/opencv/guidance_detector_opencv.hpp"
#ifdef USE_OPENVINO
    #include "openvino/guidance_detector_openvino.hpp"

#endif
namespace auto_guidance {
class DetectorFactory {
public:
    static std::unique_ptr<detector_base>
    createDetector(const std::string& backend, const YAML::Node& config, bool debug) {
#if defined(USE_OPENVINO)
        if (backend == "openvino") {
            return std::make_unique<GuidanceDetectorOpenVino>(config);
        }
#endif

        if (backend == "opencv") {
            return std::make_unique<GuidanceDetectorOpenCV>(config, debug);
        }

        throw std::runtime_error("Unsupported  detector backend (or not compiled): " + backend);
    }
};
} // namespace auto_guidance
