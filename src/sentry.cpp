#include "geometry_msgs/msg/twist.hpp"
#include "ros2/ros2.hpp"
#include "tasks/config.hpp"
#include "tasks/main_base.hpp"
#include "tasks/vision_base.hpp"
ENABLE_BACKWARD()
namespace wust_vision {
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
    void handleReferee(const ReceiveReferee& ref) {}

    void serialCallback(const uint8_t* data, std::size_t len) {
        if (len != sizeof(ReceiveAimINFO)) {
            return;
        }
        try {
            const std::vector<uint8_t> buf(data, data + len);
            const ReceiveAimINFO aim_data =
                wust_vl::common::drivers::fromVector<ReceiveAimINFO>(buf);
            processAimData(aim_data);

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
        send_data.vx = -msg->linear.y;
        send_data.vy = msg->linear.x;
        send_data.wz = msg->angular.z;
        if (serial_) {
            serial_->write(std::move(wust_vl::common::drivers::toVector(send_data)));
        }
    }
    std::shared_ptr<Ros2Node> ros2_;
    bool use_sim_ = false;
};
} // namespace wust_vision
VISION_MAIN(wust_vision::vision)
