#include "aimer.hpp"
#include "tasks/utils.hpp"
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
        const armor::Target& target,
        std::chrono::steady_clock::time_point time,
        const double bullet_speed,
        const AutoAimFsm& auto_aim_fsm
    ) {
        AimTarget aim_target;
        if (target.one_targets_is_valid) {
            aim_target = aimTargetSingleArmor(target, time, bullet_speed);
        } else {
            aim_target = aimTargetWholeCar(target, time, bullet_speed);
        }
        return aim_target;
    }
    AimTarget aimTargetSingleArmor(
        const armor::Target& target,
        std::chrono::steady_clock::time_point time,
        const double bullet_speed
    ) {
        using namespace std::chrono;

        double fly_t = trajectory_compensator_->getFlyingTime(target.position_, bullet_speed);
        double dt_sec =
            fly_t + prediction_delay_ + duration<double>(time - target.timestamp).count();
        armor::Target tmp_target = target;
        tmp_target.Predict(dt_sec);
        int idx = selectBestArmor(target);
        if (idx < 0 || idx >= (int)target.one_targets.size()
            || target.one_targets[idx].position_.norm() < 0.1)
        {
            return returnDefaultAimTarget();
        }
        AimTarget aim_target;
        aim_target.have_host = true;
        aim_target.host_pos = target.position_;
        aim_target.host_vel = target.velocity_;
        aim_target.pos = target.one_targets[idx].position_;
        aim_target.vel = target.one_targets[idx].velocity_;

        aim_target.host_v_yaw = target.v_yaw;
        double raw_pitch = aim_target.calRawPitch();
        trajectory_compensator_->compensate(aim_target.pos, raw_pitch, bullet_speed);
        aim_target.shoot_pitch = raw_pitch;
        if (target.id == armor::ArmorNumber::BASE || target.id == armor::ArmorNumber::NO1) {
            aim_target.is_big_armor = true;
        }
        double tmp_yaw = target.one_targets[idx].yaw;
        Eigen::Vector3d euler;
        euler.x() = M_PI / 2;
        euler.y() = target.id == armor::ArmorNumber::OUTPOST ? -0.2618 : 0.2618,
        euler.z() = tmp_yaw;
        Eigen::Quaterniond ori = utils::eulerToQuat(euler, utils::EulerOrder::ZYX);
        aim_target.ori = ori;
        return aim_target;
    }
    AimTarget aimTargetWholeCar(
        const armor::Target& target,
        std::chrono::steady_clock::time_point time,
        const double bullet_speed
    ) {
        using namespace std::chrono;

        double fly_t = trajectory_compensator_->getFlyingTime(target.position_, bullet_speed);
        double dt_sec =
            fly_t + prediction_delay_ + duration<double>(time - target.timestamp).count();
        armor::Target tmp_target = target;
        tmp_target.Predict(dt_sec);
        auto p_armors = tmp_target.getArmorPositions();
        auto v_armors = tmp_target.getArmorVelocities();
        int idx = selectBestArmor(p_armors, target);
        if (idx < 0 || idx >= (int)p_armors.size() || p_armors[idx].norm() < 0.1) {
            return returnDefaultAimTarget();
        }
        AimTarget aim_target;
        aim_target.have_host = true;
        aim_target.host_pos = target.position_;
        aim_target.host_vel = target.velocity_;
        aim_target.pos = p_armors[idx];
        aim_target.vel = v_armors[idx];

        aim_target.host_v_yaw = target.v_yaw;
        double raw_pitch = aim_target.calRawPitch();
        trajectory_compensator_->compensate(aim_target.pos, raw_pitch, bullet_speed);
        aim_target.shoot_pitch = raw_pitch;
        if (target.id == armor::ArmorNumber::BASE || target.id == armor::ArmorNumber::NO1) {
            aim_target.is_big_armor = true;
        }
        double tmp_yaw = target.getArmorYaw()[idx];
        Eigen::Vector3d euler;
        euler.x() = M_PI / 2;
        euler.y() = target.id == armor::ArmorNumber::OUTPOST ? -0.2618 : 0.2618,
        euler.z() = tmp_yaw;
        Eigen::Quaterniond ori = utils::eulerToQuat(euler, utils::EulerOrder::ZYX);
        aim_target.ori = ori;
        return aim_target;
    }
    AimTarget returnDefaultAimTarget() {
        last_aim_target_.is_old = true;
        return last_aim_target_;
    }
    int selectBestArmor(const armor::Target& target) {
        static Eigen::Vector3d last_pos = Eigen::Vector3d::Zero();
        static constexpr double switch_threshold = 0.2; // 防抖阈值

        if (!target.one_targets_is_valid || target.one_targets.empty()) {
            last_pos = Eigen::Vector3d::Zero();
            return -1;
        }

        int best_idx = 0;
        double best_score = target.one_targets[0].position_.norm();

        for (size_t i = 0; i < target.one_targets.size(); ++i) {
            double dist = target.one_targets[i].position_.norm();

            // 防抖：如果距离上次选择的位置较近，优先锁定
            double delta_last = (target.one_targets[i].position_ - last_pos).norm();
            if (delta_last < best_score) {
                best_idx = static_cast<int>(i);
                best_score = delta_last;
            }
        }

        // 更新 last_pos 为当前选择的目标位置
        last_pos = target.one_targets[best_idx].position_;
        return best_idx;
    }

    int selectBestArmor(
        const std::vector<Eigen::Vector3d>& armor_positions,
        const armor::Target& target
    ) const noexcept {
        if (armor_positions.empty())
            return -1;

        double center_yaw = std::atan2(target.position_.y(), target.position_.x());
        auto angles = target.getArmorYaw();
        std::vector<double> delta_angle_list;
        for (int i = 0; i < target.armors_num; i++) {
            auto delta_angle = angles::normalize_angle(angles[i] - center_yaw);
            delta_angle_list.emplace_back(delta_angle);
        }

        double coming_angle = comming_angle_ / 180.0 * M_PI;
        double leaving_angle = leaving_angle_ / 180.0 * M_PI;
        for (size_t i = 0; i < target.armors_num; ++i) {
            if (std::abs(delta_angle_list[i]) > coming_angle)
                continue;
            if (target.v_yaw > 0 && delta_angle_list[i] < leaving_angle)
                return static_cast<int>(i);
            if (target.v_yaw < 0 && delta_angle_list[i] > -leaving_angle)
                return static_cast<int>(i);
        }
        return -1;
    }

    std::unique_ptr<TrajectoryCompensator> trajectory_compensator_;
    double prediction_delay_;
    double comming_angle_;
    double leaving_angle_;
    AimTarget last_aim_target_;
};
Aimer::Aimer(): _impl(std::make_unique<Impl>()) {}
Aimer::~Aimer() {
    _impl.reset();
}
void Aimer::init(const YAML::Node& config) {
    _impl->init(config);
}
AimTarget Aimer::aimTarget(
    const armor::Target& armor_slover_target,
    std::chrono::steady_clock::time_point time,
    const double bullet_speed,
    const AutoAimFsm& auto_aim_fsm
) {
    return _impl->aimTarget(armor_slover_target, time, bullet_speed, auto_aim_fsm);
}