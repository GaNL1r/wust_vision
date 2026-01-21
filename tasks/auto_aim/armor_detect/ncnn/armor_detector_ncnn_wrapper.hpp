// Copyright 2025 Xiaojian Wu
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
#include "tasks/auto_aim/armor_detect/ncnn/armor_detector_ncnn.hpp"
#include <yaml-cpp/yaml.h>

class ArmorDetectorNCNNWrapper: public ArmorDetectorBase {
public:
    ArmorDetectorNCNNWrapper(const YAML::Node& config, bool use_armor_detect_common = true);
    ~ArmorDetectorNCNNWrapper() override;

    void
    pushInput(CommonFrame& frame, const std::optional<armor::ArmorNumber>& target_number) override;

    void setCallback(DetectorCallback cb) override;

private:
    std::unique_ptr<ArmorDetectNCNN> detector_;
};
