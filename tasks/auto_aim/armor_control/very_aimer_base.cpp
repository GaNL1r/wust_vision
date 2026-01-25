#pragma once
#include "very_aimer_base.hpp"
namespace auto_aim {
void VeryAimerBase::reset() {
    shooting_range_w_ = config_["shooting_range_w"].as<double>(0.12);
    shooting_range_h_ = config_["shooting_range_h"].as<double>(0.12);
    const double yaw_limit_deg = config_["yaw_limit"].as<double>(0.0);
    yaw_limit = yaw_limit_deg / 180.0 * M_PI;
    min_enable_pitch_deg_ = config_["min_enable_pitch_deg"].as<double>(0.0);
    min_enable_yaw_deg_ = config_["min_enable_yaw_deg"].as<double>(0.0);
    manual_compensator_ = std::make_unique<ManualCompensator>();
    std::vector<OffsetEntry> entries;

    if (config_["trajectory_offset"]) {
        for (const auto& node: config_["trajectory_offset"]) {
            OffsetEntry e;
            e.d_min = node["d_min"].as<double>();
            e.d_max = node["d_max"].as<double>();
            e.h_min = node["h_min"].as<double>();
            e.h_max = node["h_max"].as<double>();
            e.pitch_off = node["pitch_off"].as<double>();
            e.yaw_off = node["yaw_off"].as<double>();
            entries.push_back(e);
        }
        manual_compensator_->setBasePitch(config_["base_offset"]["pitch"].as<double>());
        manual_compensator_->setBaseYaw(config_["base_offset"]["yaw"].as<double>());
    }
    if (!manual_compensator_->updateMapFlow(entries) || entries.size() < 1) {
        std::cout << "Trajectory compensator init failed" << std::endl;
    }
    prediction_delay_ = config_["prediction_delay"].as<double>(0.0);
    comming_angle_ = config_["comming_angle"].as<double>(5.0);
    leaving_angle_ = config_["leaving_angle"].as<double>(5.0);
    control_delay_ = config_["control_delay"].as<double>();
    delay_enable_fire_error_ = config_["delay_enable_fire_error"].as<double>(0.0);
    max_pitch_acc_ = config_["max_pitch_acc"].as<double>();
    max_yaw_acc_ = config_["max_yaw_acc"].as<double>();
}
int VeryAimerBase::selectArmor(const Target& target, const AutoAimFsm& auto_aim_fsm)
    const noexcept {
    static int lock_id = -1;
    const auto [aim_first, aim_center, aim_pair] = getAimStatus(auto_aim_fsm);
    const auto armor_list = target.getArmorPosAndYaw();
    const int armor_num = static_cast<int>(armor_list.size());
    int i_chosen = 0;

    const double center_yaw = std::atan2(target.position().y(), target.position().x());

    std::vector<double> delta_angles;
    delta_angles.reserve(armor_num);
    for (int i = 0; i < armor_num; ++i) {
        delta_angles.push_back(angles::normalize_angle(armor_list[i][3] - center_yaw));
    }

    const auto pick_best_by_min_abs = [&](const std::vector<int>& idxs) -> int {
        int best = -1;
        double best_val = std::numeric_limits<double>::infinity();
        for (int i: idxs) {
            const double val = std::abs(delta_angles[i]);
            if (val < best_val) {
                best_val = val;
                best = i;
            }
        }
        return best;
    };

    if (aim_first && target.tracked_id_ != ArmorNumber::OUTPOST && armor_num > 0) {
        std::vector<int> candidates;
        for (int i = 0; i < armor_num; ++i)
            if (std::abs(delta_angles[i]) <= 60.0 / 57.3)
                candidates.push_back(i);

        if (!candidates.empty()) {
            if (candidates.size() > 1) {
                int a = candidates[0], b = candidates[1];
                if (lock_id != a && lock_id != b) {
                    lock_id = (std::abs(delta_angles[a]) < std::abs(delta_angles[b])) ? a : b;
                }
                int pick = (lock_id >= 0 && lock_id < armor_num) ? lock_id
                                                                 : pick_best_by_min_abs(candidates);
                if (pick >= 0) {
                    i_chosen = pick;
                }
            } else {
                lock_id = -1;
                int pick = candidates[0];
                i_chosen = pick;
            }
        }

        return i_chosen;
    }
    if (armor_num > 0) {
        int best_idx = -1;

        if (target.tracked_id_ != ArmorNumber::OUTPOST) {
            const double coming_angle = comming_angle_ * M_PI / 180.0;
            const double leaving_angle = leaving_angle_ * M_PI / 180.0;

            for (int i = 0; i < armor_num; ++i) {
                if (std::abs(delta_angles[i]) > coming_angle)
                    continue;

                if (target.v_yaw() > 0 && delta_angles[i] < leaving_angle)
                    best_idx = i;
                if (target.v_yaw() < 0 && delta_angles[i] > -leaving_angle)
                    best_idx = i;
            }
        }

        if (best_idx < 0) {
            std::vector<int> all(armor_num);
            std::iota(all.begin(), all.end(), 0);
            best_idx = pick_best_by_min_abs(all);
        }
        if (aim_pair) {
            std::vector<int> all;
            all.push_back(0);
            all.push_back(2);
            best_idx = pick_best_by_min_abs(all);
        }

        i_chosen = best_idx;
    }

    return i_chosen;
}
ControlPoint
VeryAimerBase::getControlPoint(Eigen::Vector3d aim_target_pos, double diff_yaw, double bullet_speed)
    const noexcept {
    ControlPoint cp;
    double control_yaw = std::atan2(aim_target_pos.y(), aim_target_pos.x());
    double raw_pitch = std::atan2(
        aim_target_pos.z(),
        std::sqrt(aim_target_pos.x() * aim_target_pos.x() + aim_target_pos.y() * aim_target_pos.y())
    );
    try {
        trajectory_compensator_->compensate(aim_target_pos, raw_pitch, bullet_speed);
    } catch (std::exception& e) {
        std::cout << "compensate error: " << e.what() << std::endl;
    }

    double control_pitch = raw_pitch;
    const auto offs =
        manual_compensator_->angleHardCorrect(aim_target_pos.head(2).norm(), aim_target_pos.z());
    control_yaw = angles::normalize_angle((control_yaw + offs[1] * M_PI / 180.0));
    control_pitch = (control_pitch + offs[0] * M_PI / 180.0);
    cp.pitch = control_pitch;
    cp.yaw = control_yaw;
    cp.xyza.head<3>() = aim_target_pos;
    cp.xyza[3] = diff_yaw;
    return cp;
}
std::tuple<double, double> VeryAimerBase::calEnableDiff(
    Eigen::Vector3d aim_target_pos,
    double diff_yaw,
    const AutoAimFsm& auto_aim_fsm
) const noexcept {
    const double distance = aim_target_pos.norm();
    double shooting_range_yaw = std::abs(atan2(shooting_range_w_ / 2, distance));
    double shooting_range_pitch = std::abs(atan2(shooting_range_h_ / 2, distance));
    double yaw_factor = 0.0;
    const double yaw_rad = diff_yaw;
    if (auto_aim_fsm != AutoAimFsm::AIM_SINGLE_ARMOR) {
        if (std::abs(yaw_rad) <= yaw_limit) {
            yaw_factor = std::cos(yaw_rad);
        }
    } else {
        yaw_factor = std::cos(yaw_rad);
    }

    const double pitch_factor = std::cos(15.0 * M_PI / 180);

    shooting_range_yaw = std::max(shooting_range_yaw, min_enable_yaw_deg_ * M_PI / 180);
    shooting_range_pitch = std::max(shooting_range_pitch, min_enable_pitch_deg_ * M_PI / 180);
    shooting_range_yaw *= yaw_factor;
    shooting_range_pitch *= pitch_factor;

    return std::make_tuple(std::abs(shooting_range_yaw), std::abs(shooting_range_pitch));
}
std::tuple<bool, bool, bool> VeryAimerBase::getAimStatus(const AutoAimFsm& auto_aim_fsm
) const noexcept {
    const bool aim_first = (auto_aim_fsm == AutoAimFsm::AIM_SINGLE_ARMOR);
    const bool aim_center = (auto_aim_fsm == AutoAimFsm::AIM_WHOLE_CAR_CENTER);
    const bool aim_pair = (auto_aim_fsm == AutoAimFsm::AIM_WHOLE_CAR_PAIR);
    return std::make_tuple(aim_first, aim_center, aim_pair);
}

ControlPoint VeryAimerBase::choseAndGetControlPoint(
    const Target& target,
    double bullet_speed,
    const AutoAimFsm& auto_aim_fsm
) const noexcept {
    const auto [aim_first, aim_center, aim_pair] = getAimStatus(auto_aim_fsm);
    const int target_select = selectArmor(target, auto_aim_fsm);
    const auto armors_xyza = target.getArmorPosAndYaw();
    Eigen::Vector3d aim_pos = armors_xyza[target_select].head<3>();

    if (aim_center) {
        const double raw_z = aim_pos.z();
        double c_xy_dis = std::sqrt(
            target.position().x() * target.position().x()
            + target.position().y() * target.position().y()
        );
        const double c_yaw = std::atan2(target.position().y(), target.position().x());
        c_xy_dis -= target.getArmor2CenterXYDis(target_select);
        aim_pos.x() = c_xy_dis * std::cos(c_yaw);
        aim_pos.y() = c_xy_dis * std::sin(c_yaw);
        aim_pos.z() = raw_z;
    }
    const double center_yaw = std::atan2(target.position().y(), target.position().x());
    const double d_angle =
        angles::shortest_angular_distance(center_yaw, armors_xyza[target_select][3]);
    ControlPoint cp = getControlPoint(aim_pos, d_angle, bullet_speed);
    cp.id_in_target = target_select;
    return cp;
}
inline VeryAimerBase::FireResult
VeryAimerBase::canFireAtTime(const VeryAimerTrajBase::Ptr& traj, double t) const noexcept {
    const auto target_delay = traj->getTargetState(t + control_delay_);
    const auto control_delay = traj->getControlState(t + control_delay_);

    if (std::hypot(
            angles::normalize_angle(target_delay.yaw_state.p + traj->cp0.yaw)
                - angles::normalize_angle(control_delay.yaw_state.p + traj->cp0.yaw),
            angles::normalize_angle(target_delay.pitch_state.p)
                - angles::normalize_angle(control_delay.pitch_state.p)
        )
        >= delay_enable_fire_error_)
    {
        return { false, 0, 0 };
    }

    const auto target = traj->getTargetState(t);
    const auto control = traj->getControlState(t);
    const auto aim = traj->aim_traj.getStateAtTime(t);

    const double target_yaw = angles::normalize_angle(target.yaw_state.p + traj->cp0.yaw);
    const double control_yaw = angles::normalize_angle(control.yaw_state.p + traj->cp0.yaw);

    const auto [enable_yaw, enable_pitch] = calEnableDiff(aim.pos, aim.d_angle, traj->fsm);

    return {
        std::abs(angles::shortest_angular_distance(target_yaw, control_yaw)) <= enable_yaw
            && std::abs(
                   angles::shortest_angular_distance(target.pitch_state.p, control.pitch_state.p)
               ) <= enable_pitch,
        enable_yaw,
        enable_pitch
    };
}
std::pair<LimitTrajectory, Trajectory<AimPoint>> VeryAimerBase::getTrajectory(
    Target& target,
    const ControlPoint& cp0,
    double bullet_speed,
    const AutoAimFsm& auto_aim_fsm
) const {
    LimitTrajectory traj;
    Trajectory<AimPoint> aim_traj;
    traj.reserve(HORIZON);
    aim_traj.reserve(HORIZON);

    // prepare: roll the target back so we start from same relative time as original impl
    target.predictSimple(-DT * (HALF_HORIZON + 1));

    // compute first two cps (target state is mutated between calls but choseAndGetControlPoint takes const&)
    auto cp_last = choseAndGetControlPoint(target, bullet_speed, auto_aim_fsm);
    target.predictSimple(DT);
    auto cp = choseAndGetControlPoint(target, bullet_speed, auto_aim_fsm);

    for (int i = 0; i < HORIZON; ++i) {
        target.predictSimple(DT);
        const auto cp_next = choseAndGetControlPoint(target, bullet_speed, auto_aim_fsm);
        const double yaw_vel = angles::normalize_angle(cp_next.yaw - cp_last.yaw) / (2.0 * DT);
        const double pitch_vel = (cp_next.pitch - cp_last.pitch) / (2.0 * DT);
        GimbalState pt;
        pt.yaw_state.p = angles::normalize_angle(cp.yaw - cp0.yaw);
        pt.pitch_state.p = cp.pitch;
        pt.yaw_state.v = yaw_vel;
        pt.pitch_state.v = pitch_vel;
        pt.aim_id = cp.id_in_target;
        traj.push_back(pt, DT);
        AimPoint aim_pt;
        aim_pt.d_angle = cp.xyza[3];
        aim_pt.pos = cp.xyza.head<3>();
        aim_traj.push_back(aim_pt, DT);
        cp_last = cp;
        cp = cp_next;
    }

    return { std::move(traj), std::move(aim_traj) };
}

} // namespace auto_aim