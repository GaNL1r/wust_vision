#include "3rdparty/backward-cpp/backward.hpp"
#include "geometry_msgs/msg/twist.hpp"
#include "ros2/ros2.hpp"
#include "tasks/vision_base.hpp"
#include "wust_interfaces/msg/rmul2024.hpp"

ENABLE_BACKWARD()
class vision: public VisionBase {
public:
    vision(): VisionBase(COMMON_CONFIG, CAMERA_CONFIG, AUTO_AIM_CONFIG, AUTO_BUFF_CONFIG) {}
    ~vision() {
        run_flag_ = false;
        if (sim_thread_.joinable()) {
            sim_thread_.join();
        }
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
        ros2_->add_publisher<wust_interfaces::msg::Rmul2024>("/rmul2024", rclcpp::QoS(10));
        use_sim_ = config_["use_sim"].as<bool>();
        if (use_sim_) {
            sim_thread_ = std::thread(std::bind(&vision::simThread, this));
        }
        serial_->set_receive_callback(
            std::bind(&vision::serialCallback, this, std::placeholders::_1, std::placeholders::_2)
        );
        return true;
    }

    void start() {
        VisionBase::start();
        ros2_->start();
    }
    std::thread sim_thread_;
    void simThread() {
        const std::string yaml_path = "config/rmul2024_sim.yaml";
        // 默认值
        bool myteam = true;
        int game_time = -1;
        int max_health = 400;
        int cur_health = 400;
        int cur_bullet = 0;
        int r1_health = 500, r3_health = 500, r7_health = 500;
        int b1_health = 500, b3_health = 500, b7_health = 500;

        while (run_flag_) {
            try {
                // 读取（若文件存在则覆盖默认）
                if (std::filesystem::exists(yaml_path)) {
                    YAML::Node root = YAML::LoadFile(yaml_path);
                    myteam = root["myteam"].as<bool>(myteam);
                    game_time = root["game_time"].as<int>(game_time);
                    max_health = root["max_health"].as<int>(max_health);
                    cur_health = root["cur_health"].as<int>(cur_health);
                    cur_bullet = root["cur_bullet"].as<int>(cur_bullet);
                    r1_health = root["r1_health"].as<int>(r1_health);
                    r3_health = root["r3_health"].as<int>(r3_health);
                    r7_health = root["r7_health"].as<int>(r7_health);
                    b1_health = root["b1_health"].as<int>(b1_health);
                    b3_health = root["b3_health"].as<int>(b3_health);
                    b7_health = root["b7_health"].as<int>(b7_health);
                }

                // 构建并发布
                wust_interfaces::msg::Rmul2024 msg;
                msg.header.stamp = ros2_->now();
                msg.myteam = myteam;
                msg.game_time = game_time;
                msg.max_health = max_health;
                msg.cur_health = cur_health;
                msg.cur_bullet = cur_bullet;
                msg.r1_health = r1_health;
                msg.r3_health = r3_health;
                msg.r7_health = r7_health;
                msg.b1_health = b1_health;
                msg.b3_health = b3_health;
                msg.b7_health = b7_health;

                ros2_->publish("/rmul2024", msg);
                WUST_INFO("sim") << "sim publish";
            } catch (const std::exception& e) {
                // 避免线程崩溃，记录后继续
                std::cerr << "simThread exception: " << e.what() << '\n';
            } catch (...) {
                std::cerr << "simThread unknown exception\n";
            }

            std::this_thread::sleep_for(std::chrono::milliseconds(1000));
        }
    }

    void handleAim(const ReceiveAimINFO& aim_data) {
        static Averager<double> vyaw_avg(100);
        double roll = -(aim_data.roll) * M_PI / 180.0;
        double pitch = (aim_data.pitch) * M_PI / 180.0;
        double yaw = (aim_data.yaw) * M_PI / 180.0;
        double v_roll = aim_data.roll_vel * M_PI / 180.0;
        double v_pitch = aim_data.pitch_vel * M_PI / 180.0;
        double v_yaw = aim_data.yaw_vel * M_PI / 180.0;
        vyaw_avg.add(v_yaw);
        double v_x = aim_data.v_x;
        double v_y = aim_data.v_y;
        double v_z = aim_data.v_z;

        auto now = std::chrono::steady_clock::now();
        if (motion_buffer_) {
            Motion motion { yaw, pitch, roll, vyaw_avg.average(), v_pitch, v_roll, v_x, v_y, v_z };
            motion_buffer_->push(motion, now);
        }

        writeSerialLogToJson(aim_data);
    }
    void handleReferee(const ReceiveReferee& ref) {
        wust_interfaces::msg::Rmul2024 msg;
        msg.header.stamp = ros2_->now();
        msg.myteam = ref.my_team;
        msg.game_time = ref.game_time;
        msg.max_health = ref.max_health;
        msg.cur_health = ref.cur_health;
        msg.r1_health = ref.r1_health;
        msg.r3_health = ref.r3_health;
        msg.r7_health = ref.r7_health;
        msg.b1_health = ref.b1_health;
        msg.b3_health = ref.b3_health;
        msg.b7_health = ref.b7_health;
        ros2_->publish("/rmul2024", msg);
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
                                      time_utils::now().time_since_epoch()
            )
                                      .count());
        send_data.vx = msg->linear.x;
        send_data.vy = msg->linear.y;
        send_data.wz = msg->angular.z;
        if (serial_) {
            serial_->write(std::move(toVector(send_data)));
        }
    }
    std::shared_ptr<Ros2Node> ros2_;
    bool use_sim_ = false;
};
VISION_MAIN(vision)
