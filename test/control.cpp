#include "common/utils/config_binder.hpp"
#include "common/utils/logger.hpp"
#include "common/utils/signal.hpp"
#include "wust_vl/algorithm/control/pid.hpp"
#include "wust_vl/algorithm/control/smc.hpp"
#include "wust_vl/common/drivers/serial_driver.hpp"

#include "ros2/ros2.hpp"
#include "std_msgs/msg/float32.hpp"

#include <Eigen/Dense>
#include <atomic>
#include <chrono>
#include <cmath>
#include <fstream>
#include <functional>
#include <limits>
#include <mutex>
#include <nlohmann/json.hpp>
#include <opencv2/opencv.hpp>
#include <thread>

struct ReceiveAimINFO {
    float position; // 角度 (度)
    float speed;    // 转速 (deg/s)
    float torque;   // 扭矩 (Nm)
} __attribute__((packed));

struct SendAimINFO {
    float torque; // 扭矩 (Nm)
} __attribute__((packed));

inline float normalize_angle_deg(float angle) {
    while (angle >= 180.0f) angle -= 360.0f;
    while (angle < -180.0f) angle += 360.0f;
    return angle;
}

class AimController {
public:
    AimController(const std::string& serial_port,
                  const std::string& config_path = "config/smc.yaml",
                  int send_hz = 1000)
        : serial_(),
          config_binder_(config_path),
          send_hz_(send_hz),
          send_interval_(std::chrono::microseconds(1000000 / send_hz)),
          running_(false),
          target_deg_(0.0f),
          target_speed_(0.0f),
          isinited(false)
    {
        // 初始化 SMC 矩阵与边界
        int dim = 1;
        B_ = Eigen::MatrixXd::Identity(dim, dim);
        Lambda_ = Eigen::MatrixXd::Identity(dim, dim) * 2.0;
        K_ = Eigen::MatrixXd::Identity(dim, dim) * 5.0;
        phi_vec_ = Eigen::VectorXd::Constant(dim, 0.5);

        // 配置热绑定
        config_binder_.bind({ "lambda" }, &Lambda_(0,0));
        config_binder_.bind({ "k" }, &K_(0,0));
        config_binder_.bind({ "phi" }, &phi_vec_(0));
        config_binder_.bind({"alpha"},&alpha_);

        smc_ = std::make_unique<wust_vl_algorithm::SMC_MIMO>(
            dim, B_, Lambda_, K_, phi_vec_, 0.05
        );

        // 串口初始化
        SerialDriver::SerialPortConfig cfg {
            115200, 8,
            boost::asio::serial_port_base::parity::none,
            boost::asio::serial_port_base::stop_bits::one,
            boost::asio::serial_port_base::flow_control::none
        };
        serial_.init_port(serial_port, cfg);
        serial_.set_error_callback([&](const boost::system::error_code& ec) {
            WUST_ERROR("serial") << "serial error: " << ec.message();
        });
        serial_.set_receive_callback([this](const uint8_t* data, std::size_t len) {
            this->onSerialData(data, len);
        });
        serial_.start();

        // 初始化 ROS2
        ros_node_ = std::make_shared<Ros2Node>("aim_controller_node");
        ros_node_->add_publisher<std_msgs::msg::Float32>("aim/position");
        ros_node_->add_publisher<std_msgs::msg::Float32>("aim/speed");
        ros_node_->add_publisher<std_msgs::msg::Float32>("aim/target");
        ros_node_->add_publisher<std_msgs::msg::Float32>("aim/target_speed");
        ros_node_->add_publisher<std_msgs::msg::Float32>("aim/torque");
        ros_node_->add_publisher<std_msgs::msg::Float32>("aim/s");
        ros_node_->add_publisher<std_msgs::msg::Float32>("aim/error");
        ros_node_->start();
    }

    ~AimController() {
        stop();
        SendAimINFO zero{ 0.0f };
        serial_.write(toVector(zero));
        serial_.stop();
    }

    void start() {
        if (running_) return;
        running_ = true;
        worker_thread_ = std::thread([this] { this->loopThread(); });
    }

    void stop() {
        if (!running_) return;
        running_ = false;
        if (worker_thread_.joinable())
            worker_thread_.join();
    }

    void setTarget(float deg, float speed = 0.0f) {
        isinited = true;
        target_deg_.store(deg, std::memory_order_relaxed);
        target_speed_.store(speed, std::memory_order_relaxed);
    }

    float getCurPos() {
        std::lock_guard<std::mutex> lk(data_mutex_);
        return latest_data_.position;
    }

    ReceiveAimINFO getLatestMeasurement() const {
        std::lock_guard<std::mutex> lk(data_mutex_);
        return latest_data_;
    }

private:
    void onSerialData(const uint8_t* data, std::size_t len) {
        try {
            std::vector<uint8_t> buf(data, data + len);
            ReceiveAimINFO aim_data = fromVector<ReceiveAimINFO>(buf);
            std::lock_guard<std::mutex> lk(data_mutex_);
            latest_data_ = aim_data;
        } catch (...) {}
    }

    void loopThread() {
        auto last_loop = std::chrono::steady_clock::now();
        auto last_print = last_loop;

        while (running_) {
            auto loop_start = std::chrono::steady_clock::now();
            float dt = std::chrono::duration<float>(loop_start - last_loop).count();
            last_loop = loop_start;
            if (dt <= 0) dt = 0.001f;

            ReceiveAimINFO cur;
            {
                std::lock_guard<std::mutex> lk(data_mutex_);
                cur = latest_data_;
            }

            // 热加载配置
            //config_binder_.reload("config/smc.yaml");
            smc_->updateParameters(B_, Lambda_, K_, phi_vec_, alpha_);

            float target = target_deg_.load(std::memory_order_relaxed);
            float target_speed = target_speed_.load(std::memory_order_relaxed);

            Eigen::VectorXd x(1), x_dot(1), x_ref(1);
            x << cur.position;
            x_dot << cur.speed;
            x_ref << target;

            Eigen::VectorXd s = x_dot - x_ref + Lambda_ * (x - x_ref);
            Eigen::VectorXd torque_vec = smc_->computeControl(x, x_ref, x_dot);
            float torque = torque_vec(0);

            if (isinited) {
                SendAimINFO send{ torque };
                serial_.write(toVector(send));
            }

            // ROS2 发布
            std_msgs::msg::Float32 msg;
            msg.data = cur.position; ros_node_->publish("aim/position", msg);
            msg.data = cur.speed; ros_node_->publish("aim/speed", msg);
            msg.data = target; ros_node_->publish("aim/target", msg);
            msg.data = target_speed; ros_node_->publish("aim/target_speed", msg);
            msg.data = torque; ros_node_->publish("aim/torque", msg);
            msg.data = s(0); ros_node_->publish("aim/s", msg);
            msg.data = cur.position - target; ros_node_->publish("aim/error", msg);

            // 日志
            auto now = std::chrono::steady_clock::now();
            if (now - last_print >= std::chrono::seconds(1)) {
                WUST_INFO("aim_ctrl") << "[SMC Log] target=" << target
                                      << "°, cur=" << cur.position
                                      << "°, pos_err=" << normalize_angle_deg(cur.position-target)
                                      << "°, vel_err=" << (cur.speed-target_speed)
                                      << " deg/s, torque=" << torque
                                      << " Nm, s=" << s(0);
                last_print = now;
            }

            std::this_thread::sleep_until(loop_start + send_interval_);
        }

        // 停止时发送零扭矩
        SendAimINFO zero{ 0.0f };
        serial_.write(toVector(zero));
    }

private:
    SerialDriver serial_;
    mutable std::mutex data_mutex_;
    ReceiveAimINFO latest_data_{0,0,0};
    wust_vl_utils::ConfigBinder config_binder_;

    int send_hz_;
    std::chrono::microseconds send_interval_;
    std::atomic<bool> running_;
    std::thread worker_thread_;

    std::atomic<float> target_deg_;
    std::atomic<float> target_speed_;
    bool isinited;

    // SMC 参数
    Eigen::MatrixXd B_, Lambda_, K_;
    double alpha_ ;
    Eigen::VectorXd phi_vec_;
    std::unique_ptr<wust_vl_algorithm::SMC_MIMO> smc_;

    // ROS2
    std::shared_ptr<Ros2Node> ros_node_;
};

int main(int argc, char** argv) {
    rclcpp::init(argc, argv);

    const std::string serial_port = "/dev/ttyACM_RMc";
    AimController controller(serial_port, "config/smc.yaml", 3000);
    controller.start();

    SignalHandler sig;
    std::atomic<bool> stop{false};
    sig.start([&]{ stop.store(true); controller.stop(); });

    const std::string win_name = "Aim Target Control";
    cv::namedWindow(win_name, cv::WINDOW_AUTOSIZE);

    float cur_pos = controller.getCurPos();
    float target_deg = cur_pos;
    int slider_val = static_cast<int>(cur_pos + 180.0f);
    cv::createTrackbar("Target", win_name, &slider_val, 360);

    std::cout << "使用滑动条控制目标角度（float），按 ESC 退出。\n";

    while (!sig.shouldExit() && !stop.load()) {
        cur_pos = controller.getCurPos();
        target_deg = static_cast<float>(slider_val) - 180.0f;
        controller.setTarget(target_deg, 0.0f);

        cv::Mat frame(100, 1000, CV_8UC3, cv::Scalar(50,50,50));
        std::string text = "CurPos: " + std::to_string(cur_pos)
                         + " deg, Target: " + std::to_string(target_deg) + " deg";
        cv::putText(frame, text, cv::Point(10,60),
                    cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0,255,0), 2);
        cv::imshow(win_name, frame);

        int key = cv::waitKey(50);
        if (key == 27) break;
    }

    controller.stop();
    cv::destroyAllWindows();
    rclcpp::shutdown();
    return 0;
}
