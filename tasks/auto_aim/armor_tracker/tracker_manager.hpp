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
#include "3rdparty/angles.h"
#include "tasks/auto_aim/armor_tracker/motion_models/acc_model.hpp"
#include "tasks/auto_aim/armor_tracker/one_ca_tracker.hpp"
#include "tasks/auto_aim/armor_tracker/one_tracker.hpp"
#include "tasks/auto_aim/armor_tracker/tracker.hpp"
#include "tasks/auto_aim/armor_tracker/trackerv3.hpp"
#include "tasks/auto_aim/type.hpp"
#include "trackerv2.hpp"
#include "wust_vl/common/utils/config_binder.hpp"
#include "yaml-cpp/yaml.h"
class TrackerManager {
public:
    explicit TrackerManager(const YAML::Node& config, std::shared_ptr<ConfigBinder> config_binder);
    Target update(const armor::Armors& armors, AutoAimFsmController& auto_aim_fsm_cl);
    Target updateTracker(const armor::Armors& armors);
    std::unique_ptr<TrackerV3> tracker_v3_;
    int track_one_num_;
    std::vector<std::unique_ptr<OneTracker>> one_trackers_;

    std::chrono::steady_clock::time_point last_time_;
    std::shared_ptr<ConfigBinder> config_binder_;
    AutoAimFsm auto_aim_fsm_ = AutoAimFsm::AIM_WHOLE_CAR_ARMOR;
};
