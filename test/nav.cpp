#include "geometry_msgs/msg/twist.hpp"
#include "ros2/ros2.hpp"
#include "tasks/packet_typedef.hpp"
#include <rclcpp/subscription.hpp>
#include <wust_vl/common/drivers/serial_driver.hpp>
#include <wust_vl/common/utils/logger.hpp>
#include <wust_vl/common/utils/timer.hpp>
using namespace wust_vision;
class Nav: public rclcpp::Node {
public:
    Nav(): Node("wust_vision_global_node") {
        cmd_sub_ = this->create_subscription<geometry_msgs::msg::Twist>(
            "cmd_vel",
            10,
            std::bind(&Nav::twistCb, this, std::placeholders::_1)
        );
        serial_ = std::make_shared<wust_vl::common::drivers::SerialDriver>();
        wust_vl::common::drivers::SerialDriver::SerialPortConfig cfg {
            /*baud*/ 115200,
            /*csize*/ 8,
            boost::asio::serial_port_base::parity::none,
            boost::asio::serial_port_base::stop_bits::one,
            boost::asio::serial_port_base::flow_control::none
        };
        serial_->init_port("/dev/ttyACM0", cfg);
        serial_->set_receive_callback(std::bind(
            &Nav::serialCallback,
            this,
            std::placeholders::_1,
            std::placeholders::_2

        ));
        serial_->set_error_callback([&](const boost::system::error_code& ec) {
            WUST_ERROR("serial") << "serial error: " << ec.message();
        });
    }
    void serialCallback(const uint8_t* data, std::size_t len) {}
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
    }
    rclcpp::Subscription<geometry_msgs::msg::Twist>::SharedPtr cmd_sub_;
    std::shared_ptr<wust_vl::common::drivers::SerialDriver> serial_;
};
int main(int argc, char** argv) {
    rclcpp::init(argc, argv);

    {
        auto node = rclcpp::Node::make_shared("wust_vision_global_node");
        rclcpp::spin(node);
        // rclcpp::shutdown();
    }

    return 0;
}
