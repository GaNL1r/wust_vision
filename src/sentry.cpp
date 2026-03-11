#include "geometry_msgs/msg/twist.hpp"
#include "ros2/ros2.hpp"
#include "sentry_interfase/msg/detail/mode__struct.hpp"
#include "sentry_interfase/msg/detail/robo_state__struct.hpp"
#include "sentry_interfase/msg/detail/target__struct.hpp"
#include "sentry_interfase/msg/mode.hpp"
#include "sentry_interfase/msg/robo_state.hpp"
#include "sentry_interfase/msg/target.hpp"
#include "tasks/config.hpp"
#include "tasks/main_base.hpp"
#include "tasks/packet_typedef.hpp"
#include "tasks/type_common.hpp"
#include "tasks/vision_base.hpp"
#include "visualization_msgs/msg/marker.hpp"
#include <wust_vl/common/utils/timer.hpp>
#include <yaml-cpp/node/parse.h>

ENABLE_BACKWARD()
namespace wust_vision {
class vision: public VisionBase {
public:
    static constexpr const char* TARGET_MARKER = "target_marker";

    vision(): VisionBase(COMMON_CONFIG, CAMERA_CONFIG, AUTO_AIM_CONFIG, AUTO_BUFF_CONFIG) {}
    ~vision() {
        run_flag_ = false;
        rclcpp::shutdown();
    }
    bool init(bool debug_mode) {
        VisionBase::init(debug_mode);
        rclcpp::init(0, nullptr);
        ros2_ = std::make_shared<Ros2Node>("vison_node");
        ros2_->add_subscription<geometry_msgs::msg::Twist>(
            "cmd_vel",
            std::bind(&vision::twistCb, this, std::placeholders::_1),
            rclcpp::QoS(10)
        );
        ros2_->add_subscription<sentry_interfase::msg::Mode>(
            MODE_TOPIC,
            std::bind(&vision::modeCb, this, std::placeholders::_1)
        );
        ros2_->add_publisher<visualization_msgs::msg::Marker>(TARGET_MARKER);
        ros2_->add_publisher<sentry_interfase::msg::Target>(TARGET_TOPIC);
        ros2_->add_publisher<sentry_interfase::msg::RoboState>(ROBO_STATE_TOPIC);
        serial_->set_receive_callback(
            std::bind(&vision::serialCallback, this, std::placeholders::_1, std::placeholders::_2)
        );
        timer_B_ = std::make_unique<wust_vl::common::utils::Timer>();
        return true;
    }

    void start() {
        VisionBase::start();
        ros2_->start();
        if (timer_) {
            const auto timercallback =
                std::bind(&vision::timerBCallback, this, std::placeholders::_1);
            const double rate_hz = 10.0;
            timer_->start(rate_hz, timercallback);
        }
    }
    void timerBCallback(double dt_ms) {
        AttackMode mode = toAttackMode(attack_mode_);
        do {
            if (mode == AttackMode::ARMOR) {
                auto target = auto_aim_->getTarget();
                if (!target.checkTargetAppear()) {
                    break;
                }
                auto target_pos = target.target_state_.pos(); // world frame

                double big_yaw = 0.0;

                Eigen::Rotation2Dd rot(-big_yaw);

                Eigen::Vector2d pos_world(target_pos.x(), target_pos.y());
                Eigen::Vector2d pos_bigyaw = rot * pos_world;
                publishMarker(pos_bigyaw);
                sentry_interfase::msg::Target target_msg;
                target_msg.pos.x = pos_bigyaw.x();
                target_msg.pos.y = pos_bigyaw.y();
                target_msg.pos.z = 0.0;
                target_msg.color = detect_color_;
                target_msg.id = 0;
                target_msg.header.frame_id = "gimbal_yaw";
                target_msg.header.stamp = ros2_->now();
                ros2_->publish(TARGET_TOPIC, target_msg);
            }

        } while (0);
        sentry_interfase::msg::RoboState robo_state_msg;
        auto sim = YAML::LoadFile("/home/hy/wust_vision/sim.yaml");
        robo_state_msg.cur_hp = sim["cur_hp"].as<double>();
        robo_state_msg.max_hp = sim["max_hp"].as<double>();
        ros2_->publish(ROBO_STATE_TOPIC, robo_state_msg);
        auto now = std::chrono::steady_clock::now();
        auto dt = wust_vl::common::utils::time_utils::durationSec(last_cmd_time_, now);
        if (dt > 1) {
            NavRobotCmdData send_data;
            send_data.cmd_ID = ID_NAV_CMD;
            send_data.check = true;
            send_data.time_stamp = static_cast<uint32_t>(
                std::chrono::duration_cast<std::chrono::milliseconds>(
                    wust_vl::common::utils::time_utils::now().time_since_epoch()
                )
                    .count()
            );
            send_data.vx = 0.0;
            send_data.vy = 0.0;
            send_data.wz = 0.0;
            send_data.mode = last_mode_;
            if (serial_) {
                serial_->write(std::move(wust_vl::common::drivers::toVector(send_data)));
            }
        }
    }

    void handleAim(const ReceiveAimINFO& aim_data) {
        VisionBase::processAimData(aim_data);
    }
    void handleReferee(const ReceiveReferee& ref) {}

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

                // processRefereeData(referee_data);
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
        AttackMode mode = toAttackMode(attack_mode_);
        do {
            if (mode == AttackMode::ARMOR) {
                auto target = auto_aim_->getTarget();
                if (!target.checkTargetAppear()) {
                    break;
                }
                auto target_pos = target.target_state_.pos(); // world frame

                double big_yaw = ref.big_yaw_in_world;

                Eigen::Rotation2Dd rot(-big_yaw);

                Eigen::Vector2d pos_world(target_pos.x(), target_pos.y());
                Eigen::Vector2d pos_bigyaw = rot * pos_world;
                publishMarker(pos_bigyaw);
                sentry_interfase::msg::Target target_msg;
                target_msg.pos.x = pos_bigyaw.x();
                target_msg.pos.y = pos_bigyaw.y();
                target_msg.pos.z = 0.0;
                target_msg.color = detect_color_;
                target_msg.id = 0;
                target_msg.header.frame_id = "gimbal_yaw";
                target_msg.header.stamp = ros2_->now();
                ros2_->publish(TARGET_TOPIC, target_msg);
            }

        } while (0);
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
    void modeCb(const sentry_interfase::msg::Mode::SharedPtr msg) {
        last_mode_ = msg->mode;
    }
    int last_mode_ = 0;
    void twistCb(const geometry_msgs::msg::Twist::SharedPtr msg) {
        NavRobotCmdData send_data;
        send_data.cmd_ID = ID_NAV_CMD;
        send_data.check = true;
        send_data.time_stamp =
            static_cast<uint32_t>(std::chrono::duration_cast<std::chrono::milliseconds>(
                                      wust_vl::common::utils::time_utils::now().time_since_epoch()
            )
                                      .count());
        send_data.vx = -msg->linear.y;
        send_data.vy = msg->linear.x;
        send_data.wz = msg->angular.z;
        send_data.mode = last_mode_;
        if (serial_) {
            serial_->write(std::move(wust_vl::common::drivers::toVector(send_data)));
        }
        last_cmd_time_ = wust_vl::common::utils::time_utils::now();
    }
    std::shared_ptr<Ros2Node> ros2_;
    std::unique_ptr<wust_vl::common::utils::Timer> timer_B_;
    std::chrono::steady_clock::time_point last_cmd_time_;
    bool use_sim_ = false;
};
} // namespace wust_vision
VISION_MAIN(wust_vision::vision)
