#include "geometry_msgs/msg/twist.hpp"
#include "ros2/ros2.hpp"
#include "sentry_interfaces/msg/detail/mode__struct.hpp"
#include "sentry_interfaces/msg/detail/robo_state__struct.hpp"
#include "sentry_interfaces/msg/detail/target__struct.hpp"
#include "sentry_interfaces/msg/mode.hpp"
#include "sentry_interfaces/msg/robo_state.hpp"
#include "sentry_interfaces/msg/target.hpp"
#include "tasks/auto_aim/armor_control/very_aimer.hpp"
#include "tasks/auto_aim/armor_omni/armor_omni.hpp"
#include "tasks/auto_aim/auto_aim.hpp"
#include "tasks/packet_typedef.hpp"
#include "tasks/type_common.hpp"
#include "tasks/utils/config.hpp"
#include "tasks/utils/main_base.hpp"
#include "tasks/vision_base.hpp"
#include "visualization_msgs/msg/marker.hpp"
#include <cmath>
#include <memory>
#include <wust_vl/common/utils/timer.hpp>
#include <yaml-cpp/node/parse.h>
ENABLE_BACKWARD()
namespace wust_vision {
class vision: public VisionBase<InfantryMode> {
public:
    static constexpr const char* TARGET_MARKER = "target_marker";

    vision(): VisionBase(COMMON_CONFIG, CAMERA_CONFIG, AUTO_AIM_CONFIG, AUTO_BUFF_CONFIG) {}
    ~vision() {
        armor_omni_.reset();
    }
    bool init(bool debug_mode) {
        VisionBase::init(debug_mode);
        auto auto_aim =
            auto_aim::AutoAim::create(auto_aim_config_, tf_config_, camera_info_, debug_mode);
        modules_.emplace(InfantryMode::AttackMode::ARMOR, auto_aim);
        modules_.emplace(InfantryMode::AttackMode::UNKNOWN, auto_aim);
        auto auto_buff =
            auto_buff::AutoBuff::create(auto_buff_config_, tf_config_, camera_info_, debug_mode);
        modules_.emplace(InfantryMode::AttackMode::BIG_RUNE, auto_buff);
        modules_.emplace(InfantryMode::AttackMode::SMALL_RUNE, auto_buff);

        serial_->set_receive_callback(
            std::bind(&vision::serialCallback, this, std::placeholders::_1, std::placeholders::_2)
        );
        timer_B_ = std::make_unique<wust_vl::common::utils::Timer>("10hz");
        big_yaw_motion_buffer_ =
            std::make_shared<wust_vl::common::utils::MotionBufferGeneric<BigYaw, 1024>>();
        // auto very_aimer_copy = std::make_shared<auto_aim::VeryAimer>(*auto_aim->getVeryAimer());
        auto_aim::ArmorOmni::Ctx omni_ctx = {
            .car_motion_buffer = motion_buffer_,
            .big_yaw_motion_buffer = big_yaw_motion_buffer_,
            .very_aimer = auto_aim->getVeryAimer(),
        };
        armor_omni_ = auto_aim::ArmorOmni::create(detect_color_, omni_ctx);
        rclcpp::init(0, nullptr);
        ros2_ = std::make_shared<Ros2Node>("vison_node");
        ros2_->add_subscription<geometry_msgs::msg::Twist>(
            "cmd_vel",
            std::bind(&vision::twistCb, this, std::placeholders::_1),
            rclcpp::QoS(10)
        );
        ros2_->add_subscription<sentry_interfaces::msg::Mode>(
            MODE_TOPIC,
            std::bind(&vision::modeCb, this, std::placeholders::_1)
        );
        ros2_->add_publisher<visualization_msgs::msg::Marker>(TARGET_MARKER);
        ros2_->add_publisher<sentry_interfaces::msg::Target>(TARGET_TOPIC);
        ros2_->add_publisher<sentry_interfaces::msg::RoboState>(ROBO_STATE_TOPIC);
        return true;
    }

    void start() {
        VisionBase::start();
        if (timer_) {
            const auto timercallback =
                std::bind(&vision::timerCallback, this, std::placeholders::_1);
            const double rate_hz = control_config_->control_rate_param.get();
            timer_->start(rate_hz, timercallback);
        }
        if (timer_B_) {
            const auto timercallback =
                std::bind(&vision::timerBCallback, this, std::placeholders::_1);
            const double rate_hz = 10.0;
            timer_B_->start(rate_hz, timercallback);
        }
        armor_omni_->start();
        ros2_->start();
    }
    void timerCallback(double dt_ms) {
        if (!run_flag_) {
            return;
        }

        GimbalCmd cmd;
        try {
            InfantryMode::AttackMode mode = InfantryMode::toAttackMode(attack_mode_);
            auto module = modules_.at(mode);
            if (!module) {
                return;
            }
            cmd = module->solve(bullet_speed_);
        } catch (const std::exception& e) {
            std::cout << "solve error: " << e.what() << std::endl;
        }
        if (!cmd.isValid()) {
            return;
        }
        if (!cmd.appear && armor_omni_) {
            cmd = armor_omni_->solve(bullet_speed_);
        }
        last_cmd_ = cmd;

        double cmd_pitch = cmd.pitch;
        double cmd_yaw = cmd.yaw;
        SendRobotCmdData send_data;
        send_data.cmd_ID = ID_ROBOT_CMD;
        send_data.time_stamp = std::chrono::duration_cast<std::chrono::milliseconds>(
                                   std::chrono::steady_clock::now().time_since_epoch()
        )
                                   .count();
        if (cmd.distance > 0.5) {
            send_data.appear = cmd.appear;
        } else {
            send_data.appear = false;
        }

        send_data.detect_color = detect_color_;
        send_data.pitch = cmd_pitch + cmd.v_pitch * control_config_->pitch_ramp_param.get();
        send_data.yaw = cmd_yaw + cmd.v_yaw * control_config_->yaw_ramp_param.get();
        send_data.v_pitch = cmd.v_pitch;
        send_data.v_yaw = cmd.v_yaw;
        send_data.a_pitch = cmd.a_pitch;
        send_data.a_yaw = cmd.a_yaw;
        send_data.target_yaw = cmd.target_yaw;
        send_data.target_pitch = cmd.target_pitch;
        send_data.enable_pitch_diff = cmd.enable_pitch_diff;
        send_data.enable_yaw_diff = cmd.enable_yaw_diff;
        send_data.shoot_rate = shoot_config_->rate_param.get();
        if (serial_) {
            serial_->write(std::move(wust_vl::common::drivers::toVector(send_data)));
        }
    }
    void timerBCallback(double dt_ms) {
        InfantryMode::AttackMode mode = InfantryMode::toAttackMode(attack_mode_);
        do {
            if (mode == InfantryMode::AttackMode::ARMOR) {
                auto auto_aim = auto_aim::toAutoAim(modules_.at(mode));
                if (auto_aim) {
                    auto target = auto_aim->getTarget();
                    if (armor_omni_) {
                        armor_omni_->setDetectColor(detect_color_);
                        armor_omni_->updateMainTracking(target.checkTargetAppear());
                    }

                    if (!target.checkTargetAppear()) {
                        break;
                    }
                    auto target_pos = target.target_state_.pos(); // world frame

                    double big_yaw = 0.0;

                    Eigen::Rotation2Dd rot(-big_yaw);

                    Eigen::Vector2d pos_world(target_pos.x(), target_pos.y());
                    Eigen::Vector2d pos_bigyaw = rot * pos_world;
                    publishMarker(pos_bigyaw);
                    sentry_interfaces::msg::Target target_msg;
                    target_msg.pos.x = pos_bigyaw.x();
                    target_msg.pos.y = pos_bigyaw.y();
                    target_msg.pos.z = 0.0;
                    target_msg.color = detect_color_;
                    target_msg.id = 0;
                    target_msg.header.frame_id = "gimbal_yaw";
                    target_msg.header.stamp = ros2_->now();
                    ros2_->publish(TARGET_TOPIC, target_msg);
                }
            }

        } while (0);
    }

    void serialCallback(const uint8_t* data, std::size_t len) {
        if (len < 1)
            return;

        uint8_t cmd = data[0];

        try {
            if (cmd == ID_AIM_INFO) {
                if (len != sizeof(ReceiveAimINFO))
                    return;

                const std::vector<uint8_t> buf(data, data + len);

                auto aim_data = wust_vl::common::drivers::fromVector<ReceiveAimINFO>(buf);

                processAimData(aim_data);
            }

            else if (cmd == ID_REFEREE_INFO)
            {
                if (len != sizeof(ReceiveReferee))
                    return;

                const std::vector<uint8_t> buf(data, data + len);

                auto referee_data = wust_vl::common::drivers::fromVector<ReceiveReferee>(buf);

                processRefereeData(referee_data);
            }

            else
            {
                std::cerr << "Unknown cmd_ID: " << int(cmd) << std::endl;
            }

        } catch (const std::exception& e) {
            std::cerr << "serialCallback exception: " << e.what() << std::endl;
        }
    }
    void processRefereeData(const ReceiveReferee& ref) {
        if (debug_mode_) {
            updateSerialLog(ref);
            flushSerialLog();
        }
        const auto now = std::chrono::steady_clock::now();
        double big_yaw_rad = ref.big_yaw_in_world / 180.0 * M_PI;
        if (big_yaw_motion_buffer_) {
            BigYaw big_yaw { .big_yaw = big_yaw_rad };
            big_yaw_motion_buffer_->push(big_yaw, now);
        }
        sentry_interfaces::msg::RoboState robo_state;
        robo_state.game_time = ref.game_time;
        robo_state.cur_hp = ref.cur_health;
        robo_state.max_hp = ref.max_health;
        robo_state.cur_bullet = ref.cur_bullet;
        robo_state.center_state = ref.center_state;
        ros2_->publish(ROBO_STATE_TOPIC, robo_state);
    }
    void publishMarker(const Eigen::Vector2d& pos) {
        visualization_msgs::msg::Marker marker;

        marker.header.frame_id = "gimbal_yaw";
        marker.header.stamp = ros2_->now();

        marker.ns = "target";
        marker.id = 0;
        marker.type = visualization_msgs::msg::Marker::SPHERE;
        marker.action = visualization_msgs::msg::Marker::ADD;

        marker.pose.position.x = pos.x();
        marker.pose.position.y = pos.y();
        marker.pose.position.z = 0.0;

        marker.pose.orientation.w = 1.0;

        marker.scale.x = 0.15;
        marker.scale.y = 0.15;
        marker.scale.z = 0.15;

        marker.color.a = 1.0;
        marker.color.r = 1.0;
        marker.color.g = 0.0;
        marker.color.b = 0.0;

        marker.lifetime = rclcpp::Duration(0, 0);

        ros2_->publish(TARGET_MARKER, marker);
    }
    void modeCb(const sentry_interfaces::msg::Mode::SharedPtr msg) {
        last_mode_ = msg->mode;
    }
    int last_mode_ = 0;
    void twistCb(const geometry_msgs::msg::Twist::SharedPtr msg) {
        NavRobotCmdData send_data;
        send_data.cmd_ID = ID_NAV_CMD;
        send_data.packet_type = ID_NAV_CONTROL;
        send_data.time_stamp =
            static_cast<uint32_t>(std::chrono::duration_cast<std::chrono::milliseconds>(
                                      wust_vl::common::utils::time_utils::now().time_since_epoch()
            )
                                      .count());
        send_data.vx = -msg->linear.y;
        send_data.vy = msg->linear.x;
        send_data.wz = msg->angular.z;

        if (serial_) {
            serial_->write(std::move(wust_vl::common::drivers::toVector(send_data)));
        }
        last_cmd_time_ = wust_vl::common::utils::time_utils::now();
    }
    std::shared_ptr<Ros2Node> ros2_;
    std::unique_ptr<wust_vl::common::utils::Timer> timer_B_;
    auto_aim::ArmorOmni::Ptr armor_omni_;
    std::chrono::steady_clock::time_point last_cmd_time_;
    std::shared_ptr<wust_vl::common::utils::MotionBufferGeneric<BigYaw, 1024>>
        big_yaw_motion_buffer_;
    bool use_sim_ = false;
};
} // namespace wust_vision
VISION_MAIN(wust_vision::vision)
