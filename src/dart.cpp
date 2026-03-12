#include "tasks/auto_guidance/auto_guidance.hpp"
#include "tasks/utils/main_base.hpp"
#include "tasks/utils/utils.hpp"
#include "wust_vl/common/concurrency/ThreadPool.h"
#include "wust_vl/common/drivers/serial_driver.hpp"
#include "wust_vl/common/utils/logger.hpp"
#include "wust_vl/common/utils/timer.hpp"
#include "wust_vl/video/camera.hpp"
ENABLE_BACKWARD()
namespace wust_vision {
struct SendDartCmdData {
    float diff_center_norm = 0;
} __attribute__((packed));
class vision {
public:
    bool init(bool debug_mode) {
        debug_mode_ = debug_mode;
        const char* v = std::getenv("VISION_ROOT");
        if (v)
            std::cout << "[env] VISION_ROOT = " << v << "\n";
        else
            std::cout << "[env] VISION_ROOT not set in this process\n";
        config_ = YAML::LoadFile("/home/hy/wust_vision/config/auto_guidance.yaml");
        std::string log_level_ = config_["logger"]["log_level"].as<std::string>("INFO");
        std::string log_path_ = config_["logger"]["log_path"].as<std::string>("wust_log");
        bool use_logcli = config_["logger"]["use_logcli"].as<bool>();
        bool use_logfile = config_["logger"]["use_logfile"].as<bool>();
        bool use_simplelog = config_["logger"]["use_simplelog"].as<bool>();
        wust_vl::initLogger(log_level_, log_path_, use_logcli, use_logfile, use_simplelog);

        YAML::Node camera_config = config_["camera"];
        camera_ = std::make_unique<wust_vl::video::Camera>();
        camera_->init(camera_config);
        camera_->setFrameCallback(std::bind(&vision::frameCallback, this, std::placeholders::_1));
        std::string camera_info_path =
            utils::expandEnv(camera_config["camera_info_path"].as<std::string>());
        YAML::Node config_camera_info = YAML::LoadFile(camera_info_path);
        std::vector<double> camera_k =
            config_camera_info["camera_matrix"]["data"].as<std::vector<double>>();
        std::vector<double> camera_d =
            config_camera_info["distortion_coefficients"]["data"].as<std::vector<double>>();

        assert(camera_k.size() == 9);
        assert(camera_d.size() == 5);

        cv::Mat K(3, 3, CV_64F);
        std::memcpy(K.data, camera_k.data(), 9 * sizeof(double));

        cv::Mat D(1, 5, CV_64F);
        std::memcpy(D.data, camera_d.data(), 5 * sizeof(double));

        auto camera_info = std::make_pair(K.clone(), D.clone());
        camera_info_ = camera_info;
        auto_guidance_ = auto_guidance::AutoGuidance::create();
        auto_guidance_->setDebug(debug_mode);
        auto_guidance_->init(config_, camera_info);

        max_infer_running_ = config_["max_infer_running"].as<int>();
        thread_pool_ = std::make_unique<wust_vl::common::concurrency::ThreadPool>(
            std::thread::hardware_concurrency() * 2
        );
        std::string device_name = config_["control"]["device_name"].as<std::string>();

        serial_ = std::make_shared<wust_vl::common::drivers::SerialDriver>();
        bool use_serial = config_["control"]["use_serial"].as<bool>();
        if (use_serial) {
            wust_vl::common::drivers::SerialDriver::SerialPortConfig cfg {
                /*baud*/ 115200,
                /*csize*/ 8,
                boost::asio::serial_port_base::parity::none,
                boost::asio::serial_port_base::stop_bits::one,
                boost::asio::serial_port_base::flow_control::none
            };
            serial_->init_port(device_name, cfg);
            serial_->set_receive_callback(std::bind(
                &vision::serialCallback,
                this,
                std::placeholders::_1,
                std::placeholders::_2

            ));
            std::cout << "Serial port opened" << std::endl;
            serial_->set_error_callback([&](const boost::system::error_code& ec) {
                WUST_ERROR("serial") << "serial error: " << ec.message();
            });
        }

        timer_ = std::make_unique<wust_vl::common::utils::Timer>();
        WUST_MAIN("vision") << "starting vision task";
        return true;
    }
    ~vision() {
        run_flag_ = false;
        camera_->stop();
        thread_pool_->waitUntilEmpty();
        if (debug_thread_.joinable()) {
            debug_thread_.join();
        }
    }
    void start() {
        run_flag_ = true;
        camera_->start();
        auto_guidance_->start();
        if (serial_) {
            serial_->start();
        }
        if (timer_) {
            auto timercallback = std::bind(&vision::timerCallback, this, std::placeholders::_1);
            double rate_hz = static_cast<double>(config_["control"]["control_rate"].as<int>());
            timer_->start(rate_hz, timercallback);
        }
        if (debug_mode_) {
            debug_thread_ = std::thread([this]() { this->debugThread(); });
        }
    }
    void serialCallback(const uint8_t* data, std::size_t len) {}
    void frameCallback(wust_vl::video::ImageFrame& frame) {
        if (!run_flag_ || infer_running_count_ >= max_infer_running_) {
            return;
        }
        CommonFrame common_frame;
        if (frame.src_img.empty()) {
            return;
        }
        common_frame.img_frame = std::move(frame);
        common_frame.expanded = cv::Rect(
            0,
            0,
            common_frame.img_frame.src_img.cols,
            common_frame.img_frame.src_img.rows
        );
        common_frame.offset = cv::Point2f(0, 0);
        thread_pool_->enqueue([this, frame = std::move(common_frame)]() mutable {
            infer_running_count_++;
            if (frame.img_frame.src_img.data == nullptr) {
                return;
            }
            if (frame.img_frame.src_img.empty()) {
                return;
            }
            if (auto_guidance_) {
                auto_guidance_->pushInput(frame);
            }
            infer_running_count_--;
        });
    }
    void timerCallback(double dt_ms) {
        if (!run_flag_) {
            return;
        }
        auto target = auto_guidance_->getTarget();
        cx_norm_ = target.center().x / target.image_size_.width * 2.0 - 1.0;
        SendDartCmdData send_data;
        // send_data.cmd_ID = ID_ROBOT_CMD;
        // send_data.time_stamp = std::chrono::duration_cast<std::chrono::milliseconds>(
        //                            std::chrono::steady_clock::now().time_since_epoch()
        // )
        //                            .count();
        // send_data.appear = target.is_tracking_;
        send_data.diff_center_norm = (target.is_tracking_) ? cx_norm_ : 0;
        if (serial_) {
            serial_->write(std::move(wust_vl::common::drivers::toVector(send_data)));
        }
    }

    void debugThread() {
        using namespace std::chrono;

        double us_interval = 1e6 / static_cast<double>(30.0);
        auto kInterval = std::chrono::microseconds(static_cast<int64_t>(us_interval));
        while (run_flag_) {
            auto start_time = steady_clock::now();
            try {
                auto dbg = auto_guidance_->getDebug();

                drawDebugOverlayShm(dbg, false);
                debuglog(dbg.target);
            } catch (std::exception& e) {
                std::cout << "debug thread error: " << e.what() << std::endl;
            }

            auto elapsed = steady_clock::now() - start_time;
            if (elapsed < kInterval) {
                std::this_thread::sleep_for(kInterval - elapsed);
            }
        }
    }
    void checkStateMatchMode() {}
    YAML::Node config_;
    std::unique_ptr<wust_vl::common::concurrency::ThreadPool> thread_pool_;
    std::unique_ptr<auto_guidance::AutoGuidance> auto_guidance_;
    std::unique_ptr<wust_vl::video::Camera> camera_;
    std::unique_ptr<wust_vl::common::utils::Timer> timer_;
    std::shared_ptr<wust_vl::common::drivers::SerialDriver> serial_;
    std::atomic<int> infer_running_count_ { 0 };
    bool run_flag_ = false;
    int max_infer_running_;
    bool debug_mode_ = false;
    std::pair<cv::Mat, cv::Mat> camera_info_;
    std::thread debug_thread_;
    double cx_norm_;
};
} // namespace wust_vision
VISION_MAIN(wust_vision::vision)