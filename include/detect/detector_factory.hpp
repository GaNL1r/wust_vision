// Copyright 2025 XiaoJian Wu
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
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
#ifdef USE_NCNN
    #include "detect/armor_detector_ncnn_wrapper.hpp"
    #include "detect/rune_detector_ncnn_wrapper.hpp"
#endif
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
#ifdef USE_NCNN
        if (backend == "ncnn") {
            return std::make_unique<ArmorDetectorNCNNWrapper>(config);
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
#ifdef USE_NCNN
        if (backend == "ncnn") {
            return std::make_unique<RuneDetectorNCNNWrapper>(config);
        }
#endif
        throw std::runtime_error("Unsupported rune detector backend (or not compiled): " + backend);
    }
};
