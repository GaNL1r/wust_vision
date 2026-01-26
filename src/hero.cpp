#include "3rdparty/backward-cpp/backward.hpp"
#include "ros2/ros2.hpp"
#include "tasks/auto_sniper/auto_sniper.hpp"
#include "tasks/vision_base.hpp"
ENABLE_BACKWARD()
#define AUTO_SNIPER_CONFIG "config/auto_sniper.yaml"
namespace wust_vision {
class vision: public VisionBase {
public:
    vision(): VisionBase(COMMON_CONFIG, CAMERA_CONFIG, AUTO_AIM_CONFIG, AUTO_BUFF_CONFIG) {}
    ~vision() {
        rclcpp::shutdown();
    }
    bool init(bool debug_mode) {
        VisionBase::init(debug_mode);
        rclcpp::init(0, nullptr);
        ros2_ = std::make_shared<Ros2Node>("vison_node");
        auto_sniper_ = std::make_unique<auto_sniper::AutoSniper>();
        auto auto_sniper_config = YAML::LoadFile(AUTO_SNIPER_CONFIG);
        auto_sniper_->init(auto_sniper_config);
        shoot_frame_ = auto_sniper_config["shoot_frame"].as<std::string>("base_footprint");
        return true;
    }
    void start() {
        VisionBase::start();
        if (timer_) {
            auto timercallback = std::bind(&vision::timerCallback, this, std::placeholders::_1);
            double rate_hz = control_config_->control_rate_param.get();
            timer_->start(rate_hz, timercallback);
        }
    }
    inline Eigen::Vector3d getPointFromMap() {
        geometry_msgs::msg::TransformStamped tf;

        if (!ros2_->lookup_transform("map", shoot_frame_, tf)) {
            RCLCPP_WARN(ros2_->get_logger(), "TF lookup failed, not publishing goal");
            return Eigen::Vector3d();
        }

        Eigen::Vector3d robot_pos(
            tf.transform.translation.x,
            tf.transform.translation.y,
            tf.transform.translation.z
        );
        return robot_pos;
    }
    void timerCallback(double dt_ms) {
        if (!run_flag_) {
            return;
        }
        AttackMode mode = toAttackMode(attack_mode_);
        if (mode == AttackMode::ARMOR || mode == AttackMode::UNKNOWN) {
            VisionBase::timerCallback(dt_ms);
            return;
        }
        Eigen::Vector3d self_pos = getPointFromMap();
        double bullet_speed = bullet_speed_;
        auto_sniper_->updatePosBulletSpeed(self_pos, bullet_speed);
        GimbalCmd cmd = auto_sniper_->solve(dt_ms);
        SendRobotCmdData send_data;
        send_data.cmd_ID = ID_ROBOT_CMD;

        send_data.appear = cmd.appera;

        send_data.appear = false;

        send_data.detect_color = detect_color_;
        send_data.pitch = cmd.pitch;
        send_data.yaw = cmd.yaw;
        send_data.v_pitch = cmd.v_pitch;
        send_data.v_yaw = cmd.v_yaw;
        send_data.target_yaw = cmd.target_yaw;
        send_data.target_pitch = cmd.target_pitch;
        send_data.enable_pitch_diff = cmd.enable_pitch_diff;
        send_data.enable_yaw_diff = cmd.enable_yaw_diff;

        if (serial_) {
            serial_->write(std::move(wust_vl::common::drivers::toVector(send_data)));
        }
    }

    std::unique_ptr<auto_sniper::AutoSniper> auto_sniper_;
    std::string shoot_frame_ = "base_footprint";
    std::shared_ptr<Ros2Node> ros2_;
};
} // namespace wust_vision
VISION_MAIN(wust_vision::vision)