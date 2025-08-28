#include "aimer.hpp"
#include "tasks/utils.hpp"
#include <wust_vl/common/utils/timer.hpp>
#include <wust_vl/common/utils/trajectory_compensator.hpp>
struct Aimer::Impl {
    void init(const YAML::Node& config) {
        std::string comp_type = config["aimer"]["compenstator_type"].as<std::string>("ideal");
        double gravity_ = config["aimer"]["gravity"].as<double>(10.0);
        double resistance_ = config["aimer"]["resistance"].as<double>(0.092);
        int iteration_times_ = config["aimer"]["iteration_times"].as<int>(20);
        prediction_delay_ = config["aimer"]["prediction_delay"].as<double>(0.0);
        comming_angle_ = config["aimer"]["comming_angle"].as<double>(5.0);
        leaving_angle_ = config["aimer"]["leaving_angle"].as<double>(5.0);
        trajectory_compensator_ = CompensatorFactory::createCompensator(comp_type);
        trajectory_compensator_->iteration_times_ = iteration_times_;
        trajectory_compensator_->gravity_ = gravity_;
        trajectory_compensator_->resistance_ = resistance_;
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
        return aim_target;
    }
    AimTarget returnDefaultAimTarget() {
        last_aim_target_.is_old = true;
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
        return -1;
    }

    std::unique_ptr<TrajectoryCompensator> trajectory_compensator_;
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
void Aimer::init(const YAML::Node& config) {
    _impl->init(config);
}
AimTarget Aimer::aimTarget(
    const Target& armor_slover_target,
    std::chrono::steady_clock::time_point time,
    const double bullet_speed,
    const AutoAimFsm& auto_aim_fsm
) {
    return _impl->aimTarget(armor_slover_target, time, bullet_speed, auto_aim_fsm);
}