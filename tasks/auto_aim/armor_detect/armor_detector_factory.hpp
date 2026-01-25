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
#include "tasks/auto_aim/armor_detect/armor_detector_base.hpp"
#include <string>

#ifdef USE_OPENVINO
    #include "tasks/auto_aim/armor_detect/openvino/armor_detector_openvino.hpp"
#endif

#ifdef USE_TRT
    #include "tasks/auto_aim/armor_detect/tensorrt/armor_detector_tensorrt.hpp"
#endif

#ifdef USE_NCNN
    #include "tasks/auto_aim/armor_detect/ncnn/armor_detector_ncnn.hpp"
#endif

#ifdef USE_ORT
    #include "tasks/auto_aim/armor_detect/onnxruntime/armor_detector_onnxruntime.hpp"
#endif
#include "tasks/auto_aim/armor_detect/opencv/armor_detector_opencv.hpp"
namespace auto_aim {
class DetectorFactory {
public:
    static ArmorDetectorBase::Ptr createArmorDetector(
        const std::string& backend,
        const YAML::Node& config,
        bool use_armor_detect_common
    ) {
#if defined(USE_OPENVINO)
        if (backend == "openvino") {
            return ArmorDetectorOpenVino::create(config["armor_detector"], use_armor_detect_common);
        }
#endif
#if defined(USE_TRT)
        if (backend == "tensorrt") {
            return ArmorDetectorTrt::create(config["armor_detector"], use_armor_detect_common);
        }
#endif
#if defined(USE_NCNN)
        if (backend == "ncnn") {
            return ArmorDetectorNCNN::create(config["armor_detector"], use_armor_detect_common);
        }
#endif
#if defined(USE_ORT)
        if (backend == "onnxruntime") {
            return ArmorDetectorOnnxRuntime::create(
                config["armor_detector"],
                use_armor_detect_common
            );
        }
#endif

        if (backend == "opencv") {
            return ArmorDetectorOpenCV::create(config["armor_detector"]);
        }

        throw std::runtime_error(
            "Unsupported armor detector backend (or not compiled): " + backend
        );
    }
};

} // namespace auto_aim
