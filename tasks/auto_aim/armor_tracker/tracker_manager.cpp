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
#include "tracker_manager.hpp"

TrackerManager::TrackerManager(
    const YAML::Node& config_,
    std::shared_ptr<wust_vl_utils::ConfigBinder> config_binder
) {
    tracker_v3_ = std::make_unique<TrackerV3>(config_);
}

Target TrackerManager::update(const armor::Armors& armors, AutoAimFsmController& auto_aim_fsm_cl) {
    Target target = updateTracker(armors);
    auto_aim_fsm_cl.update(std::abs(target.v_yaw()), target.jumped);
    auto_aim_fsm_ = auto_aim_fsm_cl.fsm_state_;
    last_time_ = armors.timestamp;
    return target;
}
Target TrackerManager::updateTracker(const armor::Armors& armors) {
    return this->tracker_v3_->track(armors);
}