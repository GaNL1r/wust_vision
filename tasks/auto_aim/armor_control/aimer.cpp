#include "aimer.hpp"
#include "tasks/utils.hpp"
#include <wust_vl/common/utils/timer.hpp>
struct Aimer::Impl {
    void
    init(const YAML::Node& config, std::shared_ptr<TrajectoryCompensator> trajectory_compensator) {
        prediction_delay_ = config["aimer"]["prediction_delay"].as<double>(0.0);
        comming_angle_ = config["aimer"]["comming_angle"].as<double>(5.0);
        leaving_angle_ = config["aimer"]["leaving_angle"].as<double>(5.0);
        trajectory_compensator_ = trajectory_compensator;
    }
    GimbalCmd
    aim(const Target& target,
        std::chrono::steady_clock::time_point time,
        const double bullet_speed,
        const AutoAimFsm& auto_aim_fsm,
        const double self_v_yaw) {
        auto aim_target = aimTarget(target, time, bullet_speed, auto_aim_fsm);
        double control_yaw = 0.0, control_pitch = 0.0;
        double v_yaw = 0.0, v_pitch = 0.0;
        GimbalCmd cmd;
        if (!aim_target.valid) {
            cmd.appera = false;
            return cmd;
        }

        v_pitch = aim_target.calVPitch();
        cmd.timestamp = std::chrono::steady_clock::now();
        cmd.distance = aim_target.distance();
        if (auto_aim_fsm == AutoAimFsm::AIM_WHOLE_CAR_CENTER && aim_target.have_host) {
            control_yaw = std::atan2(aim_target.host_pos.y(), aim_target.host_pos.x())
                - self_v_yaw * aim_target.dt;
            v_yaw = aim_target.calHostVYaw();
        } else {
            control_yaw = aim_target.calYaw();
            v_yaw = aim_target.calVYaw();
        }
        control_pitch = aim_target.shoot_pitch;
        cmd.yaw = control_yaw / M_PI * 180.0;
        cmd.pitch = control_pitch / M_PI * 180.0;
        cmd.v_pitch = v_pitch / M_PI * 180.0;
        cmd.v_yaw = v_yaw / M_PI * 180.0;
        cmd.aim_target = aim_target;
        cmd.appera = true;
        cmd.armor_posandyaw = aim_target.armor_posandyaw;
        return cmd;
    }
    AimTarget aimTarget(
        const Target& target,
        std::chrono::steady_clock::time_point time,
        const double bullet_speed,
        const AutoAimFsm& auto_aim_fsm
    ) {
        bool aim_first = false;
        if (auto_aim_fsm == AutoAimFsm::AIM_SINGLE_ARMOR) {
            aim_first = true;
        }
        return aimTargetWholeCar(target, time, bullet_speed, aim_first);
    }
    AimTarget aimTargetNoPre(const Target& target, const double bullet_speed, bool aim_first) {
        Target tmp_target = target;

        auto p_armors = tmp_target.getArmorPositions();
        auto v_armors = tmp_target.getArmorVelocities();
        int idx = selectBestArmor(tmp_target, aim_first);
        if (idx < 0 || idx >= (int)p_armors.size() || p_armors[idx].norm() < 0.1) {
            return returnDefaultAimTarget();
        }

        AimTarget aim_target;
        aim_target.idx = idx;
        aim_target.have_host = true;
        aim_target.host_pos = tmp_target.position();
        aim_target.host_vel = tmp_target.velocity();

        aim_target.pos = p_armors[idx];
        aim_target.vel = v_armors[idx];

        aim_target.host_v_yaw = target.v_yaw();
        double raw_pitch = aim_target.calRawPitch();
        trajectory_compensator_->compensate(aim_target.pos, raw_pitch, bullet_speed);
        aim_target.shoot_pitch = raw_pitch;
        if (target.tracked_id_ == armor::ArmorNumber::BASE
            || target.tracked_id_ == armor::ArmorNumber::NO1) {
            aim_target.is_big_armor = true;
        }
        double tmp_yaw = target.getArmorYaws()[idx];
        Eigen::Vector3d euler;
        euler.x() = M_PI / 2;
        euler.y() = target.tracked_id_ == armor::ArmorNumber::OUTPOST ? -0.2618 : 0.2618,
        euler.z() = tmp_yaw;
        Eigen::Quaterniond ori = utils::eulerToQuat(euler, utils::EulerOrder::ZYX);
        aim_target.ori = ori;
        aim_target.armor_posandyaw = target.getArmorPosAndYaw();
        aim_target.valid = true;
        last_aim_target_ = aim_target;
        return aim_target;
    }
    AimTarget aimTargetNoPreWithIdx(const Target& target, const double bullet_speed, int idx) {
        Target tmp_target = target;

        auto p_armors = tmp_target.getArmorPositions();
        auto v_armors = tmp_target.getArmorVelocities();

        AimTarget aim_target;
        aim_target.idx = idx;
        aim_target.have_host = true;
        aim_target.host_pos = tmp_target.position();
        aim_target.host_vel = tmp_target.velocity();

        aim_target.pos = p_armors[idx];
        aim_target.vel = v_armors[idx];

        aim_target.host_v_yaw = target.v_yaw();
        double raw_pitch = aim_target.calRawPitch();
        trajectory_compensator_->compensate(aim_target.pos, raw_pitch, bullet_speed);
        aim_target.shoot_pitch = raw_pitch;
        if (target.tracked_id_ == armor::ArmorNumber::BASE
            || target.tracked_id_ == armor::ArmorNumber::NO1) {
            aim_target.is_big_armor = true;
        }
        double tmp_yaw = target.getArmorYaws()[idx];
        Eigen::Vector3d euler;
        euler.x() = M_PI / 2;
        euler.y() = target.tracked_id_ == armor::ArmorNumber::OUTPOST ? -0.2618 : 0.2618,
        euler.z() = tmp_yaw;
        Eigen::Quaterniond ori = utils::eulerToQuat(euler, utils::EulerOrder::ZYX);
        aim_target.ori = ori;
        aim_target.armor_posandyaw = target.getArmorPosAndYaw();
        aim_target.valid = true;
        last_aim_target_ = aim_target;
        return aim_target;
    }

    AimTarget aimTargetWholeCar(
        const Target& target,
        std::chrono::steady_clock::time_point time,
        const double bullet_speed,
        bool aim_first
    ) {
        Target tmp_target = target;

        double dt0 = time_utils::durationSec(target.timestamp_, time) + prediction_delay_;
        std::chrono::steady_clock::time_point future =
            time + std::chrono::microseconds(int(dt0 * 1e6));
        tmp_target.predict(future);
        auto p_armors = tmp_target.getArmorPositions();
        auto v_armors = tmp_target.getArmorVelocities();
        int idx = selectBestArmor(tmp_target, aim_first);
        if (idx < 0 || idx >= (int)p_armors.size() || p_armors[idx].norm() < 0.1) {
            return returnDefaultAimTarget();
        }
        bool converged = false;
        double prev_fly_time = trajectory_compensator_->getFlyingTime(p_armors[idx], bullet_speed);
        auto pre_time0 = future + std::chrono::microseconds(static_cast<int>(prev_fly_time * 1e6));
        tmp_target.predict(pre_time0);
        auto p_armors_fallback = tmp_target.getArmorPositions();
        auto v_armors_fallback = tmp_target.getArmorVelocities();
        int idx_fallback = selectBestArmor(tmp_target, aim_first);
        if (idx_fallback < 0 || idx_fallback >= (int)p_armors_fallback.size()
            || p_armors_fallback[idx_fallback].norm() < 0.1)
        {
            return returnDefaultAimTarget();
        }
        std::vector<Target> iteration_target(10, tmp_target);
        int best_iter = -1;
        Eigen::Vector3d best_pos, best_vel;
        for (int iter = 0; iter < 10; ++iter) {
            auto predict_time =
                future + std::chrono::microseconds(static_cast<int>(prev_fly_time * 1e6));
            iteration_target[iter].predict(predict_time);
            int iter_idx = selectBestArmor(iteration_target[iter], aim_first);
            auto iter_p_armors = iteration_target[iter].getArmorPositions();
            auto iter_v_armors = iteration_target[iter].getArmorVelocities();
            if (idx < 0 || idx >= (int)iter_p_armors.size() || iter_p_armors[iter_idx].norm() < 0.1)
            {
                continue;
            }
            best_iter = iter;
            best_pos = iter_p_armors[iter_idx];
            best_vel = iter_v_armors[iter_idx];
            double iter_fly_time =
                trajectory_compensator_->getFlyingTime(iter_p_armors[iter_idx], bullet_speed);
            if (std::abs(iter_fly_time - prev_fly_time) < 0.001) {
                converged = true;
                break;
            }
            prev_fly_time = iter_fly_time;
        }
        AimTarget aim_target;
        aim_target.dt = prev_fly_time;
        aim_target.idx = idx;
        aim_target.have_host = true;
        aim_target.host_pos = tmp_target.position();
        aim_target.host_vel = tmp_target.velocity();
        if (best_iter > 0) {
            aim_target.pos = best_pos;
            aim_target.vel = best_vel;
        } else {
            aim_target.pos = p_armors_fallback[idx_fallback];
            aim_target.vel = v_armors_fallback[idx_fallback];
        }
        aim_target.host_v_yaw = target.v_yaw();
        double raw_pitch = aim_target.calRawPitch();
        trajectory_compensator_->compensate(aim_target.pos, raw_pitch, bullet_speed);
        aim_target.shoot_pitch = raw_pitch;
        if (target.tracked_id_ == armor::ArmorNumber::BASE
            || target.tracked_id_ == armor::ArmorNumber::NO1) {
            aim_target.is_big_armor = true;
        }
        double tmp_yaw = target.getArmorYaws()[idx];
        Eigen::Vector3d euler;
        euler.x() = M_PI / 2;
        euler.y() = target.tracked_id_ == armor::ArmorNumber::OUTPOST ? -0.2618 : 0.2618,
        euler.z() = tmp_yaw;
        Eigen::Quaterniond ori = utils::eulerToQuat(euler, utils::EulerOrder::ZYX);
        aim_target.ori = ori;
        if (best_iter > 0) {
            aim_target.armor_posandyaw = tmp_target.getArmorPosAndYaw();
        } else {
            aim_target.armor_posandyaw = tmp_target.getArmorPosAndYaw();
        }
        aim_target.valid = true;
        last_aim_target_ = aim_target;
        return aim_target;
    }
    AimTarget returnDefaultAimTarget() {
        last_aim_target_.is_old = true;
        last_aim_target_.valid = false;
        return last_aim_target_;
    }

    int selectBestArmor(const Target& target, bool aim_first) noexcept {
        std::vector<Eigen::Vector4d> armor_xyza_list = target.getArmorPosAndYaw();
        auto armor_num = armor_xyza_list.size();
        double center_yaw = std::atan2(target.position().y(), target.position().x());
        std::vector<double> delta_angle_list;
        for (int i = 0; i < armor_num; i++) {
            auto delta_angle = angles::normalize_angle(armor_xyza_list[i][3] - center_yaw);
            delta_angle_list.emplace_back(delta_angle);
        }

        if (aim_first && target.tracked_id_ != armor::ArmorNumber::OUTPOST) {
            std::vector<int> id_list;
            for (int i = 0; i < armor_num; i++) {
                if (std::abs(delta_angle_list[i]) > 60 / 57.3)
                    continue;
                id_list.push_back(i);
            }
            if (id_list.empty()) {
                return -1;
            }

            if (id_list.size() > 1) {
                int id0 = id_list[0], id1 = id_list[1];
                if (lock_id_ != id0 && lock_id_ != id1)
                    lock_id_ = (std::abs(delta_angle_list[id0]) < std::abs(delta_angle_list[id1]))
                        ? id0
                        : id1;
                return lock_id_;
            }
            lock_id_ = -1;
            return id_list[0];
        }
        double coming_angle = comming_angle_ / 180.0 * M_PI;
        double leaving_angle = leaving_angle_ / 180.0 * M_PI;
        for (size_t i = 0; i < target.armor_num_; ++i) {
            if (std::abs(delta_angle_list[i]) > coming_angle)
                continue;
            if (target.v_yaw() > 0 && delta_angle_list[i] < leaving_angle)
                return static_cast<int>(i);
            if (target.v_yaw() < 0 && delta_angle_list[i] > -leaving_angle)
                return static_cast<int>(i);
        }

        double min_angle = std::numeric_limits<double>::max();
        int best_idx = -1;
        for (int i = 0; i < armor_num; ++i) {
            double abs_angle = std::abs(delta_angle_list[i]);
            if (abs_angle < min_angle) {
                min_angle = abs_angle;
                best_idx = i;
            }
        }
        return best_idx;

        return -1;
    }

    std::shared_ptr<TrajectoryCompensator> trajectory_compensator_;
    double prediction_delay_;
    double comming_angle_;
    double leaving_angle_;
    AimTarget last_aim_target_;
    int lock_id_ = -1;
};
Aimer::Aimer(): _impl(std::make_unique<Impl>()) {}
Aimer::~Aimer() {
    _impl.reset();
}
void Aimer::init(
    const YAML::Node& config,
    std::shared_ptr<TrajectoryCompensator> trajectory_compensator
) {
    _impl->init(config, trajectory_compensator);
}
AimTarget Aimer::aimTarget(
    const Target& armor_slover_target,
    std::chrono::steady_clock::time_point time,
    const double bullet_speed,
    const AutoAimFsm& auto_aim_fsm
) {
    return _impl->aimTarget(armor_slover_target, time, bullet_speed, auto_aim_fsm);
}
AimTarget Aimer::aimTargetNoPre(const Target& target, const double bullet_speed, bool aim_first) {
    return _impl->aimTargetNoPre(target, bullet_speed, aim_first);
}
int Aimer::selectBestArmor(const Target& target, bool aim_first) {
    return _impl->selectBestArmor(target, aim_first);
}
double Aimer::getFlyingTime(const Eigen::Vector3d& target_position, const double bullet_speed) {
    return _impl->trajectory_compensator_->getFlyingTime(target_position, bullet_speed);
}
AimTarget Aimer::aimTargetNoPreWithIdx(const Target& target, const double bullet_speed, int idx) {
    return _impl->aimTargetNoPreWithIdx(target, bullet_speed, idx);
}
bool Aimer::compensate(
    const Eigen::Vector3d& target_position,
    double& pitch,
    const double bullet_speed
) {
    return _impl->trajectory_compensator_->compensate(target_position, pitch, bullet_speed);
}
GimbalCmd Aimer::aim(
    const Target& target,
    std::chrono::steady_clock::time_point time,
    const double bullet_speed,
    const AutoAimFsm& auto_aim_fsm,
    const double self_v_yaw
) {
    return _impl->aim(target, time, bullet_speed, auto_aim_fsm, self_v_yaw);
}
std::pair<double, double> Aimer::getCommingLeaving() {
    return { _impl->comming_angle_, _impl->leaving_angle_ };
}
double Aimer::getPredelay() {
    return _impl->prediction_delay_;
}