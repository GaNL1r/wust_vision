#pragma once
#include "detect/armor_detector_base.hpp"
#include "detect/rune_detector_base.hpp"
#include <memory>
#include <string>

#ifdef USE_OPENVINO
    #include "detect/armor_detector_openvino_wrapper.hpp"
    #include "detect/rune_detector_openvino_wrapper.hpp"
#endif

#ifdef USE_TRT
    #include "detect/armor_detector_trt_wrapper.hpp"
    #include "detect/rune_detector_trt_wrapper.hpp"
#endif
#include "detect/armor_detector_opencv_wrapper.hpp"
class DetectorFactory {
public:
    static std::unique_ptr<ArmorDetectorBase>
    createArmorDetector(const std::string& backend, const YAML::Node& config) {
#if defined(USE_OPENVINO)
        if (backend == "openvino") {
            return std::make_unique<ArmorDetectorOpenvinoWrapper>(config);
        }
        if (backend == "opencv") {
            return std::make_unique<ArmorDetectorOpencvWrapper>(config);
        }
#endif
#if defined(USE_TRT)
        if (backend == "tensorrt") {
            return std::make_unique<ArmorDetectorTrtWrapper>(config);
        }
        if (backend == "opencv") {
            return std::make_unique<ArmorDetectorOpencvWrapper>(config);
        }
#endif
        throw std::runtime_error(
            "Unsupported armor detector backend (or not compiled): " + backend
        );
    }

    static std::unique_ptr<RuneDetectorBase>
    createRuneDetector(const std::string& backend, const YAML::Node& config) {
#if defined(USE_OPENVINO)
        if (backend == "openvino") {
            return std::make_unique<RuneDetectorOpenvinoWrapper>(config);
        }
#endif
#if defined(USE_TRT)
        if (backend == "tensorrt") {
            return std::make_unique<RuneDetectorTrtWrapper>(config);
        }
#endif
        throw std::runtime_error("Unsupported rune detector backend (or not compiled): " + backend);
    }
};
