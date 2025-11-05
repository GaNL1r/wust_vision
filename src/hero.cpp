#include "3rdparty/backward-cpp/backward.hpp"
#include "ros2/ros2.hpp"
#include "tasks/auto_sniper/auto_sniper.hpp"
#include "tasks/vision_base.hpp"
ENABLE_BACKWARD()
#define AUTO_SNIPER_CONFIG "config/auto_sniper.yaml"

class vision: public VisionBase {
public:
    vision(): VisionBase(COMMON_CONFIG, CAMERA_CONFIG, AUTO_AIM_CONFIG, AUTO_BUFF_CONFIG) {}
    ~vision() {
        VisionBase::~VisionBase();
        auto node = Ros2Context::get_node();
        rclcpp::shutdown();
        node->shutdown();
        Ros2Context::shutdown();
    }
    bool init(bool debug_mode) {
        VisionBase::init(debug_mode);
        rclcpp::init(0, nullptr);
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
            double rate_hz = static_cast<double>(config_["control"]["control_rate"].as<int>());
            timer_->start(rate_hz, timercallback);
        }
    }
    inline Eigen::Vector3d getPointFromMap() {
        auto ros_node = Ros2Context::get_node();

        geometry_msgs::msg::TransformStamped tf;

        if (!ros_node->lookup_transform("map", shoot_frame_, tf)) {
            RCLCPP_WARN(ros_node->get_logger(), "TF lookup failed, not publishing goal");
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
        double avg_pitch = pitch_avg_->average();
        double avg_yaw = yaw_avg_->average();
        send_data.pitch = avg_pitch;
        send_data.yaw = avg_yaw;
        send_data.v_pitch = cmd.v_pitch;
        send_data.v_yaw = cmd.v_yaw;
        send_data.target_yaw = cmd.target_yaw;
        send_data.target_pitch = cmd.target_pitch;
        send_data.enable_pitch_diff = cmd.enable_pitch_diff;
        send_data.enable_yaw_diff = cmd.enable_yaw_diff;

        if (serial_) {
            serial_->write(std::move(toVector(send_data)));
        }
    }

    std::unique_ptr<auto_sniper::AutoSniper> auto_sniper_;
    std::string shoot_frame_ = "base_footprint";
};
VISION_MAIN(vision)