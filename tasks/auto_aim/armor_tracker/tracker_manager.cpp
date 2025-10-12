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
#include "tasks/auto_aim/type.hpp"
#include "tasks/utils.hpp"
TrackerManager::TrackerManager(
    const YAML::Node& config_,
    std::shared_ptr<wust_vl_utils::ConfigBinder> config_binder
) {
    double tracking_thres = config_["armor_tracker"]["tracking_thres"].as<int>(5);
    double lost_time_thres = config_["armor_tracker"]["lost_time_thres"].as<double>();
    double max_yaw_diff_deg = config_["armor_tracker"]["max_yaw_diff_deg"].as<double>(80.0);
    double max_dis_diff = config_["armor_tracker"]["max_dis_diff"].as<double>(0.5);
    TargetConfig target_config;
    target_config.qxyz_common = config_["armor_tracker"]["qxyz_common"].as<double>();
    target_config.qyaw_common = config_["armor_tracker"]["qyaw_common"].as<double>();
    target_config.qxyz_output = config_["armor_tracker"]["qxyz_output"].as<double>();
    target_config.qyaw_output = config_["armor_tracker"]["qyaw_output"].as<double>();
    target_config.q_r = config_["armor_tracker"]["q_r"].as<double>();
    target_config.q_l = config_["armor_tracker"]["q_l"].as<double>();
    target_config.q_h = config_["armor_tracker"]["q_h"].as<double>();
    target_config.yp_r = config_["armor_tracker"]["yp_r"].as<double>();
    target_config.dis_r_front = config_["armor_tracker"]["dis_r_front"].as<double>();
    target_config.dis_r_side = config_["armor_tracker"]["dis_r_side"].as<double>();
    target_config.dis2_r_ratio = config_["armor_tracker"]["dis2_r_ratio"].as<double>();
    target_config.yaw_r_base_front = config_["armor_tracker"]["yaw_r_base_front"].as<double>();
    target_config.yaw_r_base_side = config_["armor_tracker"]["yaw_r_base_side"].as<double>();
    target_config.yaw_r_log_ratio = config_["armor_tracker"]["yaw_r_log_ratio"].as<double>();
    target_config.esekf_iter_num = config_["armor_tracker"]["esekf_iter_num"].as<int>(2);
    tracker_v3_ = std::make_unique<TrackerV3>(
        tracking_thres,
        lost_time_thres,
        max_yaw_diff_deg,
        max_dis_diff,
        target_config
    );
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