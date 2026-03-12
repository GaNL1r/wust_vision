#include "debug.hpp"
namespace wust_vision::auto_aim {
struct DebugLogs {
#define DEBUG_LOG_LIST(X) \
    X(double, 100, time) \
    X(double, 100, raw_yaw) \
    X(double, 100, raw_pitch) \
    X(double, 100, yaw) \
    X(double, 100, pitch) \
    X(double, 100, armor_dis) \
    X(double, 100, armor_x) \
    X(double, 100, armor_y) \
    X(double, 100, armor_z) \
    X(double, 100, armor_yaw) \
    X(double, 100, ypd_y) \
    X(double, 100, ypd_p) \
    X(double, 100, gimbal_yaw) \
    X(double, 100, gimbal_pitch) \
    X(double, 100, target_v_yaw) \
    X(double, 100, control_v_yaw) \
    X(double, 100, control_v_pitch) \
    X(double, 100, yaw_diff) \
    X(double, 100, fire) \
    X(double, 100, rune_dis) \
    X(double, 100, fly_time) \
    X(double, 100, control_a_yaw) \
    X(double, 100, control_a_pitch)
#define GEN_LOG(TYPE, SIZE, NAME) LogsStream<TYPE, SIZE> NAME##_log { #NAME };

#define X(TYPE, SIZE, NAME) GEN_LOG(TYPE, SIZE, NAME)
    DEBUG_LOG_LIST(X)
#undef X

    void clear() {
#define X(TYPE, SIZE, NAME) NAME##_log.clear();
        DEBUG_LOG_LIST(X)
#undef X
    }
};

void debuglog(const AutoAimDebug& dbg_armor) {
    static bool first_log = true;
    static std::chrono::steady_clock::time_point start_time;

    static auto_aim::Armor last_armor_;
    static double last_armor_yaw_ = 0.0;
    static double last_ypd_y_ = 0.0;
    static double last_ypd_p_ = 0.0;
    static double last_distance_ = 0.0;
    static DebugLogs log;
    static GimbalCmd last_cmd_;
    static double rune_dis = 0.0;
    if (first_log) {
        start_time = std::chrono::steady_clock::now();
        first_log = false;
    }
    const auto now = std::chrono::steady_clock::now();
    const auto_aim::Armors& armors = dbg_armor.armors;
    const double t = std::chrono::duration<double>(now - start_time).count();
    const auto_aim::Target& target = dbg_armor.target;
    writeTargetLogToJson(target);

    double armor_yaw = 0.0, ypd_y = 0.0, ypd_p = 0.0, armor_distance = 0.0;

    if (!armors.armors.empty()) {
        std::vector<auto_aim::Armor> ok_armors;
        for (const auto& armor: armors.armors) {
            if (armor.number != auto_aim::ArmorNumber::OUTPOST)
                ok_armors.push_back(armor);
        }

        if (!ok_armors.empty()) {
            const auto_aim::Armor& min_armor = *std::min_element(
                ok_armors.begin(),
                ok_armors.end(),
                [](const auto_aim::Armor& a, const auto_aim::Armor& b) {
                    return a.distance_to_image_center < b.distance_to_image_center;
                }
            );

            last_armor_ = min_armor;

            armor_distance = std::hypot(
                min_armor.target_pos.x(),
                min_armor.target_pos.y(),
                min_armor.target_pos.z()
            );
            auto orientationToYaw = [](const Eigen::Quaterniond& q) noexcept -> double {
                Eigen::Vector3d euler = utils::quatToEuler(q, utils::EulerOrder::ZYX, false);
                double yaw = euler[0];
                yaw = last_armor_yaw_ + angles::shortest_angular_distance(last_armor_yaw_, yaw);
                last_armor_yaw_ = yaw;
                return yaw;
            };

            armor_yaw = orientationToYaw(min_armor.target_ori);

            ypd_y = std::atan2(min_armor.target_pos.y(), min_armor.target_pos.x());
            ypd_y = last_ypd_y_ + angles::shortest_angular_distance(last_ypd_y_, ypd_y);
            last_ypd_y_ = ypd_y;

            ypd_p = std::atan2(
                min_armor.target_pos.z(),
                std::hypot(min_armor.target_pos.x(), min_armor.target_pos.y())
            );
            last_ypd_p_ = ypd_p;

            last_distance_ = armor_distance;
        }
    }
    GimbalCmd i_use;
    if (dbg_armor.gimbal_cmd.appera) {
        i_use = dbg_armor.gimbal_cmd;
    } else {
        i_use = last_cmd_;
    }
    last_cmd_ = i_use;
    nlohmann::json j;
    log.time_log.handleOnce(t, j);
    log.raw_yaw_log.handleOnce(i_use.raw_yaw, j);
    log.raw_pitch_log.handleOnce(i_use.raw_pitch, j);
    log.yaw_log.handleOnce(i_use.yaw, j);
    log.pitch_log.handleOnce(i_use.pitch, j);
    log.armor_yaw_log.handleOnce(armor_yaw * 180.0 / M_PI, j);
    log.armor_x_log.handleOnce(last_armor_.target_pos.x(), j);
    log.armor_y_log.handleOnce(last_armor_.target_pos.y(), j);
    log.armor_z_log.handleOnce(last_armor_.target_pos.z(), j);
    log.ypd_y_log.handleOnce(last_ypd_y_ * 180.0 / M_PI, j);
    log.ypd_p_log.handleOnce(last_ypd_p_ * 180.0 / M_PI, j);
    log.armor_dis_log.handleOnce(last_distance_, j);
    log.gimbal_pitch_log.handleOnce(dbg_armor.gimbal_py.first * 180.0 / M_PI, j);
    log.gimbal_yaw_log.handleOnce(dbg_armor.gimbal_py.second * 180.0 / M_PI, j);
    log.target_v_yaw_log.handleOnce(target.target_state_.vyaw(), j);
    log.control_v_pitch_log.handleOnce(i_use.v_pitch, j);
    log.control_v_yaw_log.handleOnce(i_use.v_yaw, j);
    log.fire_log.handleOnce(i_use.fire_advice, j);
    log.rune_dis_log.handleOnce(rune_dis, j);
    log.fly_time_log.handleOnce(i_use.fly_time, j);
    log.control_a_yaw_log.handleOnce(i_use.a_yaw / 180.0 * M_PI, j);
    log.control_a_pitch_log.handleOnce(i_use.a_pitch / 180.0 * M_PI, j);

    std::ofstream file("/dev/shm/cmd_log.json");
    if (file.is_open()) {
        file << j.dump();
    }
}
} // namespace wust_vision::auto_aim