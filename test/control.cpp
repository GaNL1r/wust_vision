#include "common/utils/config_binder.hpp"
#include "common/utils/logger.hpp"
#include "common/utils/signal.hpp"
#include "wust_vl/algorithm/control/pid.hpp"
#include "wust_vl/algorithm/control/smc.hpp"
#include "wust_vl/common/drivers/serial_driver.hpp"

#include <Eigen/Dense>
#include <atomic>
#include <chrono>
#include <cmath>
#include <fstream>
#include <functional>
#include <limits>
#include <mutex>
#include <nlohmann/json.hpp>
#include <thread>

using json = nlohmann::json;

struct ReceiveAimINFO {
    float position; // 角度 (度)
    float speed;    // 转速 (deg/s)
    float torque;   // 扭矩 (Nm)
} __attribute__((packed));

struct SendAimINFO {
    float torque;   // 扭矩 (Nm)
} __attribute__((packed));

inline float normalize_angle_deg(float angle) {
    while (angle >= 180.0f) angle -= 360.0f;
    while (angle < -180.0f) angle += 360.0f;
    return angle;
}

class AimController {
public:
    AimController(
        const std::string& serial_port,
        const std::string& config_path = "config/smc.yaml",
        int send_hz = 1000
    ) : serial_(),
        config_binder_(config_path),
        send_hz_(send_hz),
        send_interval_(std::chrono::microseconds(1000000 / send_hz)),
        running_(false),
        target_deg_(0.0f),
        target_speed_(0.0f)
    {
        // 默认 SMC 参数
        B_ = Eigen::MatrixXd::Identity(1,1);
        Lambda_ = Eigen::MatrixXd::Identity(1,1) * 2.0;
        K_ = Eigen::MatrixXd::Identity(1,1) * 5.0;
        phi_ = 0.5;

        // 配置热更新绑定
        config_binder_.bind({"lambda"}, &Lambda_(0,0));
        config_binder_.bind({"k"}, &K_(0,0));
        config_binder_.bind({"phi"}, &phi_);

        smc_ = std::make_unique<wust_vl_algorithm::SMC_MIMO>(1, B_, Lambda_, K_, phi_);

        SerialDriver::SerialPortConfig cfg{115200, 8,
            boost::asio::serial_port_base::parity::none,
            boost::asio::serial_port_base::stop_bits::one,
            boost::asio::serial_port_base::flow_control::none};
        serial_.init_port(serial_port, cfg);
        serial_.set_error_callback([&](const boost::system::error_code& ec) { WUST_ERROR("serial") << "serial error: " << ec.message(); });
        serial_.set_receive_callback([this](const uint8_t* data, std::size_t len){
            this->onSerialData(data,len);
        });
        serial_.start();
    }

    ~AimController() {
        stop();
        SendAimINFO zero{0.0f};
        serial_.write(toVector(zero));
        serial_.stop();
    }

    void start() {
        if(running_) return;
        running_ = true;
        worker_thread_ = std::thread([this]{ this->loopThread(); });
    }

    void stop() {
        if(!running_) return;
        running_ = false;
        if(worker_thread_.joinable()) worker_thread_.join();
    }

    void setTarget(float deg, float speed=0.0f) {
        target_deg_.store(deg,std::memory_order_relaxed);
        target_speed_.store(speed,std::memory_order_relaxed);
    }

    ReceiveAimINFO getLatestMeasurement() const {
        std::lock_guard<std::mutex> lk(data_mutex_);
        return latest_data_;
    }

private:
    void onSerialData(const uint8_t* data,std::size_t len){
        try {
            std::vector<uint8_t> buf(data,data+len);
            ReceiveAimINFO aim_data = fromVector<ReceiveAimINFO>(buf);
            {
                std::lock_guard<std::mutex> lk(data_mutex_);
                latest_data_ = aim_data;
            }
        } catch(...) {}
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

        float target = target_deg_.load(std::memory_order_relaxed);
        float target_speed = target_speed_.load(std::memory_order_relaxed);

        // 角度误差与速度误差
        float pos_err = normalize_angle_deg(cur.position - target);
        float vel_err = cur.speed - target_speed;

        // 状态向量
        Eigen::VectorXd x(1), x_dot(1), x_ref(1), x_dot_ref(1);
        x << cur.position;
        x_dot << cur.speed;
        x_ref << target;
        x_dot_ref << target_speed;

        // 热更新参数
        config_binder_.reload("config/smc.yaml");

        // 计算滑模面 s
        Eigen::VectorXd s = x_dot - x_dot_ref + Lambda_ * (x - x_ref);

        // SMC 控制律
        Eigen::VectorXd torque_vec = smc_->computeControl(x, x_ref, x_dot);
        float torque = torque_vec(0);

        // 发送控制量
        SendAimINFO send{ torque };
        serial_.write(toVector(send));

        // 每秒打印日志
        auto now = std::chrono::steady_clock::now();
        if (now - last_print >= std::chrono::seconds(1)) {
            WUST_INFO("aim_ctrl") 
                << "[SMC Log] target=" << target << "°, cur=" << cur.position
                << "°, pos_err=" << pos_err 
                << "°, speed_err=" << vel_err 
                << " deg/s, torque=" << torque 
                << " Nm, s=" << s(0);
            last_print = now;
        }

        std::this_thread::sleep_until(loop_start + send_interval_);
    }

    // 停止时发送零扭矩
    SendAimINFO zero{0.0f};
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

    // SMC
    Eigen::MatrixXd B_, Lambda_, K_;
    double phi_;
    std::unique_ptr<wust_vl_algorithm::SMC_MIMO> smc_;
};
int main() {
    const std::string serial_port = "/dev/ttyACM_RMc";
    AimController controller(serial_port, "config/smc.yaml", 3000);

    // 启动控制线程
    controller.start();

    // 注册信号优雅退出（可选）
    SignalHandler sig;
    std::atomic<bool> stop { false };
    sig.start([&] {
        stop.store(true);
        controller.stop();
    });

    std::cout << "AimController started. 输入目标角度（度）并回车，或输入 q 退出。\n";

    const int STDIN_FD = 0;
    while (!sig.shouldExit() && !stop.load()) {
        // 使用 select 等待 stdin 可读，超时用来检查退出标志
        fd_set readfds;
        FD_ZERO(&readfds);
        FD_SET(STDIN_FD, &readfds);

        // 设置超时时间（秒, 微秒）——这里 200ms
        struct timeval tv;
        tv.tv_sec = 0;
        tv.tv_usec = 200 * 1000;

        int ret = select(STDIN_FD + 1, &readfds, nullptr, nullptr, &tv);
        if (ret > 0 && FD_ISSET(STDIN_FD, &readfds)) {
            // stdin 有可读数据（在终端情形下是用户敲了回车）
            std::string line;
            if (!std::getline(std::cin, line)) {
                // 读到 EOF 或错误
                break;
            }
            if (line.empty())
                continue;
            if (line == "q" || line == "quit" || line == "exit") {
                break;
            }
            try {
                float deg = std::stof(line);
                controller.setTarget(deg,0.0);
                std::cout << "set target = " << deg << " deg\n";
            } catch (...) {
                std::cout << "invalid input\n";
            }
        } else if (ret == 0) {
            // timeout: 没有输入，继续循环以检查退出标志
            continue;
        } else {
            // select 出错（通常不会发生），可以选择退出或短暂 sleep
            if (errno == EINTR) {
                // 被信号中断，检查退出标志
                continue;
            } else {
                std::perror("select");
                break;
            }
        }
    }

    controller.stop();
    return 0;
}