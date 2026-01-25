#include "3rdparty/backward-cpp/backward.hpp"
#include "geometry_msgs/msg/twist.hpp"
#include "ros2/ros2.hpp"
#include "tasks/vision_base.hpp"

ENABLE_BACKWARD()
class vision: public VisionBase {
public:
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
        serial_->set_receive_callback(
            std::bind(&vision::serialCallback, this, std::placeholders::_1, std::placeholders::_2)
        );
        return true;
    }

    void start() {
        VisionBase::start();
        ros2_->start();
    }

    void handleAim(const ReceiveAimINFO& aim_data) {
        VisionBase::processAimData(aim_data);
    }
    void handleReferee(const ReceiveReferee& ref) {
        // wust_interfaces::msg::Rmul2024 msg;
        // msg.header.stamp = ros2_->now();
        // msg.myteam = ref.my_team;
        // msg.game_time = ref.game_time;
        // msg.max_health = ref.max_health;
        // msg.cur_health = ref.cur_health;
        // msg.r1_health = ref.r1_health;
        // msg.r3_health = ref.r3_health;
        // msg.r7_health = ref.r7_health;
        // msg.b1_health = ref.b1_health;
        // msg.b3_health = ref.b3_health;
        // msg.b7_health = ref.b7_health;
        // ros2_->publish("/rmul2024", msg);
    }

    void serialCallback(const uint8_t* data, std::size_t len) {
        try {
            if (!data || len == 0)
                return;

            uint8_t cmd = data[0];

            if (cmd == ID_AIM_INFO) {
                if (len < sizeof(ReceiveAimINFO)) {
                    std::cerr << "AIM: bad length " << len << " (need >= " << sizeof(ReceiveAimINFO)
                              << ")\n";
                    return;
                }
                ReceiveAimINFO aim;
                std::memcpy(&aim, data, sizeof(aim));
                if (aim.cmd_ID != ID_AIM_INFO) {
                    std::cerr << "AIM: cmd_ID mismatch\n";
                    return;
                }
                handleAim(aim);

            } else if (cmd == ID_REFEREE_INFO) {
                if (len < sizeof(ReceiveReferee)) {
                    std::cerr << "REF: bad length " << len << " (need >= " << sizeof(ReceiveReferee)
                              << ")\n";
                    return;
                }
                ReceiveReferee ref;
                std::memcpy(&ref, data, sizeof(ref));
                if (ref.cmd_ID != ID_REFEREE_INFO) {
                    std::cerr << "REF: cmd_ID mismatch\n";
                    return;
                }
                handleReferee(ref);

            } else {
                std::cerr << "serialCallback: unknown cmd " << static_cast<int>(cmd) << '\n';
            }

        } catch (const std::exception& e) {
            std::cerr << "serialCallback exception: " << e.what() << std::endl;
        } catch (...) {
            std::cerr << "serialCallback unknown exception" << std::endl;
        }
    }

    void twistCb(const geometry_msgs::msg::Twist::SharedPtr msg) {
        NavRobotCmdData send_data;
        send_data.cmd_ID = ID_NAV_CMD;
        send_data.check = true;
        send_data.time_stamp =
            static_cast<uint32_t>(std::chrono::duration_cast<std::chrono::milliseconds>(
                                      wust_vl::common::utils::time_utils::now().time_since_epoch()
            )
                                      .count());
        send_data.vx = msg->linear.x;
        send_data.vy = msg->linear.y;
        send_data.wz = msg->angular.z;
        if (serial_) {
            serial_->write(std::move(wust_vl::common::drivers::toVector(send_data)));
        }
    }
    std::shared_ptr<Ros2Node> ros2_;
    bool use_sim_ = false;
};
VISION_MAIN(vision)
