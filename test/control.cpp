#include "common/utils/config_binder.hpp"
#include "common/utils/logger.hpp"
#include "common/utils/signal.hpp"
#include "wust_vl/algorithm/control/pid.hpp"
#include "wust_vl/common/drivers/serial_driver.hpp"

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
    float speed; // 转速 (rpm)
    float torque; // 扭矩 (Nm)
} __attribute__((packed));

struct SendAimINFO {
    float torque; // 扭矩 (Nm)
} __attribute__((packed));

inline float normalize_angle_deg(float angle) {
    while (angle >= 180.0f)
        angle -= 360.0f;
    while (angle < -180.0f)
        angle += 360.0f;
    return angle;
}
inline float shortest_angular_distance(float from, float to) {
    return normalize_angle_deg(to - from);
}

/**
 * AimController
 *  - 对外最小接口： setTargetDeg(float) 仅用于设置目标角度（度）
 *  - start()/stop() 启停控制线程（构造后需 start()）
 */
class AimController {
public:
    AimController(
        const std::string& serial_port,
        const std::string& config_path = "config/pid.yaml",
        int send_hz = 1000
    ):
        serial_(),
        config_binder_(config_path),
        send_hz_(send_hz),
        config_path_(config_path),
        send_interval_(std::chrono::microseconds(1000000 / send_hz)),
        running_(false),
        target_deg_(0.0f) {
        // default PID params (会由 config 覆盖)
        Kp_ = 0.05f;
        Ki_ = 0.01f;
        Kd_ = 0.005f;
        out_min_ = -10.0f;
        out_max_ = 10.0f;
        integrator_limit_ = 100.0f;
        deriv_tau_ = 0.01f;
        anti_windup_gain_ = 1.0f;
        derivative_on_measurement_ = false;

        // bind config keys
        config_binder_.bind({ "kp" }, &Kp_);
        config_binder_.bind({ "ki" }, &Ki_);
        config_binder_.bind({ "kd" }, &Kd_);
        config_binder_.bind({ "out_min" }, &out_min_);
        config_binder_.bind({ "out_max" }, &out_max_);
        config_binder_.bind({ "integrator_limit" }, &integrator_limit_);
        config_binder_.bind({ "derivative_tau" }, &deriv_tau_);
        config_binder_.bind({ "anti_windup_gain" }, &anti_windup_gain_);
        config_binder_.bind({ "derivative_on_measurement" }, &derivative_on_measurement_);

        // init serial
        SerialDriver::SerialPortConfig cfg { 115200,
                                             8,
                                             boost::asio::serial_port_base::parity::none,
                                             boost::asio::serial_port_base::stop_bits::one,
                                             boost::asio::serial_port_base::flow_control::none };
        serial_.init_port(serial_port, cfg);
        serial_.set_error_callback([&](const boost::system::error_code& ec) {
            WUST_ERROR("serial") << "serial error: " << ec.message();
        });
        // receive callback -> member lambda capturing this
        serial_.set_receive_callback([this](const uint8_t* data, std::size_t len) {
            this->onSerialData(data, len);
        });

        // create pid (no fixed dt)
        pid_ = std::make_unique<wust_vl_algorithm::PID<float>>(Kp_, Ki_, Kd_, -1.0f);
        applyPidConfig();

        // start serial (not start loop)
        serial_.start();
    }

    ~AimController() {
        stop();
        // ensure we send zero torque on destruction
        SendAimINFO send { 0.0f };
        serial_.write(toVector(send));
        serial_.stop();
    }

    // 禁止拷贝
    AimController(const AimController&) = delete;
    AimController& operator=(const AimController&) = delete;

    // 启动控制线程
    void start() {
        if (running_)
            return;
        running_ = true;
        worker_thread_ = std::thread([this] { this->loopThread(); });
    }

    // 停止控制线程（阻塞直到退出）
    void stop() {
        WUST_INFO("aim_ctrl") << "stop";
        if (!running_)
            return;
        running_ = false;
        if (worker_thread_.joinable())
            worker_thread_.join();
    }

    // 对外唯一控制接口：设置目标角度（度）
    void setTargetDeg(float deg) {
        target_deg_.store(deg, std::memory_order_relaxed);
    }

    // 可选：获取最近一次测量值（线程安全副本）
    ReceiveAimINFO getLatestMeasurement() const {
        std::lock_guard<std::mutex> lk(data_mutex_);
        return latest_data_;
    }

private:
    // 串口接收回调（把数据转换并保存到 latest_data_）
    void onSerialData(const uint8_t* data, std::size_t len) {
        try {
            std::vector<uint8_t> buf(data, data + len);
            ReceiveAimINFO aim_data = fromVector<ReceiveAimINFO>(buf);
            {
                std::lock_guard<std::mutex> lk(data_mutex_);
                latest_data_ = aim_data;
                has_feedback_ = true;
            }
        } catch (const std::exception& e) {
            WUST_ERROR("aim_ctrl") << "serialCallback exception: " << e.what();
        } catch (...) {
            WUST_ERROR("aim_ctrl") << "serialCallback unknown exception";
        }
    }

    // 将当前配置应用到 PID 实例
    void applyPidConfig() {
        if (!pid_)
            return;
        pid_->setGains(Kp_, Ki_, Kd_);
        pid_->setOutputLimits(out_min_, out_max_);
        pid_->setIntegratorLimit(integrator_limit_);
        pid_->setDerivativeFilterTau(deriv_tau_);
        pid_->setAntiWindupGain(anti_windup_gain_);
        pid_->setDerivativeOnMeasurement(derivative_on_measurement_);
    }

    // 主循环线程
    void loopThread() {
        auto last_loop = std::chrono::steady_clock::now();
        auto last_print = last_loop;
        while (running_) {
            auto loop_start = std::chrono::steady_clock::now();
            float dt = std::chrono::duration<float>(loop_start - last_loop).count();
            last_loop = loop_start;
            if (!(dt > 0.0f))
                dt = std::numeric_limits<float>::epsilon() * 100.0f;

            // 读取最新反馈（线程安全）
            ReceiveAimINFO cur;
            {
                std::lock_guard<std::mutex> lk(data_mutex_);
                cur = latest_data_;
            }

            // 读取目标（原子）
            float target = target_deg_.load(std::memory_order_relaxed);

            // 误差（考虑环绕）
            float error = cur.position - target;

            // PID 更新（参数会在每秒重载后应用）
            float torque = pid_->update(target, cur.position, dt);
            torque = std::clamp(torque, out_min_, out_max_);

            // 发送
            SendAimINFO send { torque };
            serial_.write(toVector(send));

            // 每秒重载配置并打印状态
            auto now = std::chrono::steady_clock::now();
            if (now - last_print >= std::chrono::seconds(1)) {
                // reload config and apply (hot reload)
                config_binder_.reload(config_path_);
                applyPidConfig();

                // 日志：打印目标/当前/输出/积分/微分
                float integrator = pid_->getIntegrator();
                float deriv = pid_->getDerivative();
                WUST_INFO("aim_ctrl") << "[Control] target=" << target << "°, cur=" << cur.position
                                      << "°, err=" << error << " => torque=" << torque << " Nm"
                                      << "  integr=" << integrator << " derr=" << deriv;
                last_print = now;
            }

            // wait until next interval (sleep_until)
            std::this_thread::sleep_until(loop_start + send_interval_);
        } // end while

        // stopped: send zero torque as safety
        SendAimINFO zero { 0.0f };
        serial_.write(toVector(zero));
    }

private:
    SerialDriver serial_;

    mutable std::mutex data_mutex_;
    ReceiveAimINFO latest_data_ { 0, 0, 0 };
    bool has_feedback_ { false };

    ConfigBinder config_binder_;

    int send_hz_;
    std::chrono::microseconds send_interval_;

    std::unique_ptr<wust_vl_algorithm::PID<float>> pid_;

    // PID params (backed by config binder)
    float Kp_, Ki_, Kd_;
    float out_min_, out_max_;
    float integrator_limit_;
    float deriv_tau_;
    float anti_windup_gain_;
    bool derivative_on_measurement_;

    std::atomic<bool> running_;
    std::thread worker_thread_;

    std::atomic<float> target_deg_;
    std::string config_path_;

}; // class AimController
int main() {
    const std::string serial_port = "/dev/ttyACM_RMc";
    AimController controller(serial_port, "config/pid.yaml", 3000);

    // 启动控制线程
    controller.start();

    // 注册信号优雅退出（可选）
    SignalHandler sig;
    std::atomic<bool> stop{false};
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
            if (line.empty()) continue;
            if (line == "q" || line == "quit" || line == "exit") {
                break;
            }
            try {
                float deg = std::stof(line);
                controller.setTargetDeg(deg);
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