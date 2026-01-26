#pragma once
#include "sinple_img_rotate_saver.hpp"
#include "tasks/auto_aim/auto_aim.hpp"
#include "tasks/auto_buff/auto_buff.hpp"
#include "tasks/utils.hpp"
#include "wust_vl/common/concurrency/ThreadPool.h"
#include "wust_vl/common/drivers/serial_driver.hpp"
#include <fmt/core.h>
namespace wust_vision {
struct LoggerConfig: wust_vl::common::utils::ParamGroup {
public:
    static constexpr const char* kKey = "logger";
    static constexpr const char* Logger = "Config: common::logger";
    const char* key() const override {
        return kKey;
    }
    using Ptr = std::shared_ptr<LoggerConfig>;
    LoggerConfig() {}
    static Ptr create() {
        return std::make_shared<LoggerConfig>();
    }

    bool first_load = false;

    void loadSelf(const YAML::Node& node) override {
        if (!first_load) {
            std::string log_level = node["log_level"].as<std::string>();
            std::string log_path = node["log_path"].as<std::string>();
            bool use_logcli = node["use_logcli"].as<bool>();
            bool use_logfile = node["use_logfile"].as<bool>();
            bool use_simplelog = node["use_simplelog"].as<bool>();
            initLogger(log_level, log_path, use_logcli, use_logfile, use_simplelog);
            first_load = true;
        } else {
        }
    }
};

struct MaxInferRunningConfig:
    public wust_vl::common::utils::SimpleConfigBase<MaxInferRunningConfig> {
    static constexpr const char* kKey = "max_infer_running";
    static constexpr const char* Logger = "Config: common::max_infer_running";

    int max_infer_running = 0;

    void loadSelf(const YAML::Node& node) override {
        loadOnceOrUpdate(
            node,
            max_infer_running,
            [](const YAML::Node& n, int& v) { v = n.as<int>(); },
            [](const YAML::Node& n, int& v) {
                int nv = n.as<int>();
                if (nv != v) {
                    v = nv;
                    WUST_DEBUG(Logger) << "max_infer_running change to " << nv;
                }
            }
        );
    }
};

class VisionBase {
public:
    VisionBase(
        std::string common_config,
        std::string camera_config,
        std::string auto_aim_config,
        std::string auto_buff_config
    );
    ~VisionBase();
    bool init(bool debug_mode);
    void start();
    void serialCallback(const uint8_t* data, std::size_t len);
    void frameCallback(wust_vl::video::ImageFrame& frame);
    void checkStateMatchMode() const;
    void timerCallback(double dt_ms);
    void debugThread();
    void autoExposureControl(const cv::Mat& frame);
    void updateBulletSpeed(double bullet_speed);
    void processAimData(const ReceiveAimINFO& aim_data);
    struct ControlConfig: wust_vl::common::utils::ParamGroup {
    public:
        static constexpr const char* kKey = "control";
        static constexpr const char* Logger = "Config: common::control";
        const char* key() const override {
            return kKey;
        }
        using Ptr = std::shared_ptr<ControlConfig>;
        ControlConfig(VisionBase* b) {
            base = b;
            communication_delay_us_param.onChange([this](int o, int n) {
                if (isBaseACtive()) {
                    base->auto_aim_shared_->communication_delay_μs = n;
                    base->auto_buff_shared_->communication_delay_μs = n;
                    WUST_DEBUG(Logger)
                        << "communication_delay_μs from: " << o << " to: " << n << " us";
                }
            });
        }
        static Ptr create(VisionBase* b) {
            return std::make_shared<ControlConfig>(b);
        }
        GEN_PARAM(double, communication_delay_us);
        GEN_PARAM(double, yaw_ramp);
        GEN_PARAM(double, pitch_ramp);
        GEN_PARAM(double, control_rate);
        VisionBase* base;
        bool first_load = false;
        bool isBaseACtive() {
            return base != nullptr;
        }
        void loadSelf(const YAML::Node& node) override {
            if (!isBaseACtive())
                return;
            if (!first_load) {
                communication_delay_us_param.set(node["communication_delay_us"].as<double>());
                yaw_ramp_param.set(node["yaw_ramp"].as<double>());
                pitch_ramp_param.set(node["pitch_ramp"].as<double>());
                control_rate_param.set(node["control_rate"].as<double>());
                std::string device_name = node["device_name"].as<std::string>();
                base->serial_ = std::make_shared<wust_vl::common::drivers::SerialDriver>();
                bool use_serial = node["use_serial"].as<bool>();
                if (use_serial) {
                    wust_vl::common::drivers::SerialDriver::SerialPortConfig cfg {
                        /*baud*/ 115200,
                        /*csize*/ 8,
                        boost::asio::serial_port_base::parity::none,
                        boost::asio::serial_port_base::stop_bits::one,
                        boost::asio::serial_port_base::flow_control::none
                    };
                    base->serial_->init_port(device_name, cfg);
                    base->serial_->set_receive_callback(std::bind(
                        &VisionBase::serialCallback,
                        base,
                        std::placeholders::_1,
                        std::placeholders::_2

                    ));
                    base->serial_->set_error_callback([&](const boost::system::error_code& ec) {
                        WUST_ERROR("serial") << "serial error: " << ec.message();
                    });
                }
                first_load = true;
            } else {
                communication_delay_us_param.load(node);
                yaw_ramp_param.load(node);
                pitch_ramp_param.load(node);
                control_rate_param.load(node);
            }
        }
    };
    ControlConfig::Ptr control_config_;
    struct ShootConfig: wust_vl::common::utils::ParamGroup {
    public:
        static constexpr const char* kKey = "shoot";
        static constexpr const char* Logger = "Config: common::shoot";
        const char* key() const override {
            return kKey;
        }
        using Ptr = std::shared_ptr<ShootConfig>;
        ShootConfig(VisionBase* b) {
            base = b;
            rate_param.onChange([this](int o, int n) {
                WUST_DEBUG(Logger) << "shoot_rate from: " << o << " to: " << n << " HZ";
            });
        }

        static Ptr create(VisionBase* b) {
            return std::make_shared<ShootConfig>(b);
        }
        GEN_PARAM(int, rate);
        GEN_PARAM(double, bullet_speed);
        VisionBase* base;
        bool first_load = false;
        bool isBaseACtive() {
            return base != nullptr;
        }
        void loadSelf(const YAML::Node& node) override {
            if (!isBaseACtive())
                return;
            if (!first_load) {
                rate_param.set(node["rate"].as<int>());
                bullet_speed_param.set(node["bullet_speed"].as<double>());
                base->updateBulletSpeed(bullet_speed_param.get());
                first_load = true;
            } else {
                rate_param.load(node);
            }
        }
    };
    ShootConfig::Ptr shoot_config_;
    struct RecordConfig: wust_vl::common::utils::ParamGroup {
    public:
        static constexpr const char* kKey = "record";
        static constexpr const char* Logger = "Config: common::record";
        const char* key() const override {
            return kKey;
        }
        using Ptr = std::shared_ptr<RecordConfig>;
        RecordConfig(VisionBase* b) {
            base = b;
        }
        static Ptr create(VisionBase* b) {
            return std::make_shared<RecordConfig>(b);
        }

        VisionBase* base;
        bool first_load = false;
        bool isBaseACtive() {
            return base != nullptr;
        }
        void loadSelf(const YAML::Node& node) override {
            if (!isBaseACtive())
                return;
            if (!first_load) {
                bool use_record = node["use_record"].as<bool>(false);
                if (use_record) {
                    std::string folder_path = node["folder_path"].as<std::string>();
                    auto file_name = utils::makeTimestampedFileName();
                    std::string text_path = fmt::format("{}/{}.csv", folder_path, file_name);
                    std::string video_path = fmt::format("{}/{}.avi", folder_path, file_name);

                    std::filesystem::create_directory(folder_path);
                    auto rw = std::make_shared<RotateWriterCSV>(true);
                    base->rotate_writer_ =
                        std::make_shared<wust_vl::common::utils::Recorder<Eigen::Vector3d>>(
                            text_path,
                            rw
                        );
                    auto imgw = std::make_shared<ImgWriter>(
                        video_path,
                        30,
                        cv::VideoWriter::fourcc('M', 'J', 'P', 'G')
                    );
                    base->img_writer_ =
                        std::make_shared<wust_vl::common::utils::Recorder<cv::Mat>>("", imgw);
                }
                bool use_rotate_reader = node["use_rotate_reader"].as<bool>();
                if (use_rotate_reader) {
                    std::string csv_path = node["read_csv_path"].as<std::string>();
                    base->rotate_reader_ = std::make_shared<RotateReaderCSV>(csv_path);
                }
                first_load = true;
            } else {
            }
        }
    };
    RecordConfig::Ptr record_config_;

    LoggerConfig::Ptr logger_config_;

    TFConfig::Ptr tf_config_;

    MaxInferRunningConfig::Ptr max_infer_running_config_;

    int attack_mode_;
    int debug_fps_;
    bool detect_color_;

    wust_vl::common::utils::Parameter common_config_parameter_;
    double bullet_speed_;
    std::unique_ptr<wust_vl::common::concurrency::ThreadPool> thread_pool_;
    std::shared_ptr<auto_aim::AutoAim> auto_aim_;
    std::shared_ptr<auto_buff::AutoBuff> auto_buff_;
    std::shared_ptr<wust_vl::video::Camera> camera_;
    std::shared_ptr<wust_vl::common::drivers::SerialDriver> serial_;
    std::unique_ptr<wust_vl::common::utils::Timer> timer_;
    std::shared_ptr<auto_aim::AutoAimShared> auto_aim_shared_;
    std::shared_ptr<auto_buff::AutoBuffShared> auto_buff_shared_;
    std::shared_ptr<wust_vl::common::utils::MotionBufferGeneric<Motion, 1024>> motion_buffer_;
    std::shared_ptr<wust_vl::common::utils::Recorder<Eigen::Vector3d>> rotate_writer_;
    RotateReaderCSV::RotateReaderCSVPtr rotate_reader_;
    std::shared_ptr<wust_vl::common::utils::Recorder<cv::Mat>> img_writer_;
    std::thread debug_thread_;

    GimbalCmd last_cmd_;
    std::pair<cv::Mat, cv::Mat> camera_info_;
    bool run_flag_ = false;
    bool debug_mode_ = false;
    int use_ncnn_count_ = 0;
    std::atomic<int> infer_running_count_ { 0 };
    std::string common_config_;
    std::string camera_config_;
    std::string auto_aim_config_;
    std::string auto_buff_config_;
};
} // namespace wust_vision