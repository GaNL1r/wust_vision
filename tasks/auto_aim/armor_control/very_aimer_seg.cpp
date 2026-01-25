#include "very_aimer_seg.hpp"

namespace auto_aim {
VeryAimerTrajSeg::Ptr VeryAimerSeg::buildVerAimerTraj(
    Target& target,
    int fin_target_select,
    double bullet_speed,
    const AutoAimFsm& auto_aim_fsm
) const {
    VeryAimerTrajSeg::Ptr res;
    res = VeryAimerTrajSeg::create();

    const auto fin_armors_xyza = target.getArmorPosAndYaw();
    res->fin_aim_pos = fin_armors_xyza[fin_target_select].head<3>();

    const auto [aim_first, aim_center, aim_pair] = getAimStatus(auto_aim_fsm);
    if (aim_center) {
        const double raw_z = res->fin_aim_pos.z();
        double c_xy_dis = std::hypot(target.position().x(), target.position().y());
        const double c_yaw = std::atan2(target.position().y(), target.position().x());

        c_xy_dis -= target.getArmor2CenterXYDis(fin_target_select);

        res->fin_aim_pos.x() = c_xy_dis * std::cos(c_yaw);
        res->fin_aim_pos.y() = c_xy_dis * std::sin(c_yaw);
        res->fin_aim_pos.z() = raw_z;
    }

    {
        AimTarget at;
        at.pos = res->fin_aim_pos;
        at.valid = true;

        Eigen::Vector3d euler;
        euler.x() = M_PI / 2.0;
        euler.y() = (target.tracked_id_ == ArmorNumber::OUTPOST) ? -0.2618 : 0.2618;
        euler.z() = target.yaw();

        at.ori = utils::eulerToQuat(euler, utils::EulerOrder::ZYX);
        res->aim_target = at;
    }

    const double center_yaw = std::atan2(target.position().y(), target.position().x());

    const double d_angle =
        angles::shortest_angular_distance(center_yaw, fin_armors_xyza[fin_target_select][3]);

    res->cp0 = getControlPoint(res->fin_aim_pos, d_angle, bullet_speed);
    res->cp0.id_in_target = fin_target_select;
    res->cp0.xyza = fin_armors_xyza[fin_target_select];
    auto traj = getTrajectory(target, res->cp0, bullet_speed, auto_aim_fsm);
    traj.first.buildLimit(max_yaw_acc_, max_pitch_acc_);
    res->target_traj = traj.first;
    res->aim_traj = traj.second;
    return res;
}

GimbalCmd
VeryAimerSeg::veryAim(Target target, double bullet_speed, const AutoAimFsm& auto_aim_fsm) {
    GimbalCmd cmd;

    const auto [aim_first, aim_center, aim_pair] = getAimStatus(auto_aim_fsm);
    const int roughly_select = selectArmor(target, auto_aim_fsm);

    const auto now = time_utils::now();
    const double dt0 = time_utils::durationSec(target.timestamp_, now);
    target.predictSimple(now + std::chrono::microseconds(int(dt0 * 1e6)));

    const auto ap = target.getArmorPositions();
    const double fly_time =
        trajectory_compensator_->getFlyingTime(ap[roughly_select].head<3>(), bullet_speed);

    double prev_fly_time = fly_time;
    std::vector<Target> iteration_target(10, target);

    for (int iter = 0; iter < 10; ++iter) {
        iteration_target[iter].predictSimple(prev_fly_time);

        const int iter_select = selectArmor(iteration_target[iter], auto_aim_fsm);

        const auto iter_poss = iteration_target[iter].getArmorPositions();
        const double iter_fly_time =
            trajectory_compensator_->getFlyingTime(iter_poss[iter_select], bullet_speed);

        if (std::abs(iter_fly_time - prev_fly_time) < 0.001)
            break;

        prev_fly_time = iter_fly_time;
    }

    const double predict_time = prev_fly_time + prediction_delay_;
    target.predictSimple(predict_time);
    const auto fin_armors_xyza = target.getArmorPosAndYaw();

    const int fin_target_select = selectArmor(target, auto_aim_fsm);

    VeryAimerTrajSeg::Ptr build;
    try {
        build = buildVerAimerTraj(target, fin_target_select, bullet_speed, auto_aim_fsm);
    } catch (const std::exception& e) {
        WUST_WARN("very_aimer") << "mpc solver error: " << e.what();
        cmd.appera = false;
        return cmd;
    } catch (...) {
        WUST_WARN("very_aimer") << "mpc solver unknown error";
        cmd.appera = false;
        return cmd;
    }

    cmd.aim_target = build->aim_target;
    double target_yaw_rad = build->cp0.yaw;
    double target_pitch_rad = build->cp0.pitch;

    if (aim_center) {
        const double center_yaw = std::atan2(target.position().y(), target.position().x());

        const double d_angle =
            angles::shortest_angular_distance(center_yaw, fin_armors_xyza[fin_target_select][3]);
        auto cp_center =
            getControlPoint(fin_armors_xyza[fin_target_select].head<3>(), d_angle, bullet_speed);
        target_yaw_rad = cp_center.yaw;
        target_pitch_rad = cp_center.pitch;
    }

    const double half_t = build->target_traj.getPrefixTimeAtIdx(HALF_HORIZON);

    const auto control_state = build->getControlState(half_t);

    const double control_yaw_rad =
        angles::normalize_angle(control_state.yaw_state.p + build->cp0.yaw);

    cmd.yaw = rad2deg(control_yaw_rad);
    cmd.v_yaw = rad2deg(control_state.yaw_state.v);
    cmd.pitch = rad2deg(control_state.pitch_state.p);
    cmd.v_pitch = rad2deg(control_state.pitch_state.v);
    cmd.a_yaw = rad2deg(control_state.yaw_state.a);
    cmd.a_pitch = rad2deg(control_state.pitch_state.a);
    cmd.target_yaw = rad2deg(target_yaw_rad);
    cmd.target_pitch = rad2deg(target_pitch_rad);
    cmd.raw_yaw = cmd.target_yaw;
    cmd.raw_pitch = cmd.target_pitch;
    cmd.distance = build->fin_aim_pos.norm();
    cmd.fly_time = prev_fly_time;

    const auto fire_now = VeryAimerBase::canFireAtTime(build, half_t);
    cmd.enable_yaw_diff = fire_now.enable_yaw_diff;
    cmd.enable_pitch_diff = fire_now.enable_pitch_diff;
    cmd.fire_advice = fire_now.fire;

    cmd.appera = cmd.isValid();
    if (!cmd.appera) {
        reset();
        WUST_WARN("very_aimer") << "very_aimer nan!";
    }

    return cmd;
}
} // namespace auto_aim