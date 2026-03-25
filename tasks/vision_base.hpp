#pragma once
#include "tasks/auto_aim/auto_aim.hpp"
#include "tasks/auto_buff/auto_buff.hpp"
#include "tasks/imodule.hpp"
#include "tasks/type_common.hpp"
#include "tasks/utils/sinple_img_rotate_saver.hpp"
#include "tasks/utils/utils.hpp"
#include "wust_vl/common/concurrency/ThreadPool.h"
#include "wust_vl/common/drivers/serial_driver.hpp"
#include <fmt/core.h>
#include <map>
#include <vector>
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
            wust_vl::initLogger(log_level, log_path, use_logcli, use_logfile, use_simplelog);
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
template<typename Mode>
class VisionBase {
public:
    VisionBase(
        std::string common_config,
        std::string camera_config,
        std::string auto_aim_config,
        std::string auto_buff_config
    ):
        common_config_(common_config),
        camera_config_(camera_config),
        auto_aim_config_(auto_aim_config),
        auto_buff_config_(auto_buff_config) {}
    ~VisionBase() {
        run_flag_ = false;
        if (debug_thread_.joinable()) {
            debug_thread_.join();
        }

        WUST_MAIN("main") << "vision stop already!";
    }
    bool init(bool debug_mode) {
        try {
            const char* v = std::getenv("VISION_ROOT");
            if (v)
                std::cout << "[env] VISION_ROOT = " << v << "\n";
            else
                std::cout << "[env] VISION_ROOT not set in this process\n";
            debug_mode_ = debug_mode;
            common_config_parameter_.loadFromFile(common_config_);
            control_config_ = ControlConfig::create(this);
            shoot_config_ = ShootConfig::create(this);
            record_config_ = RecordConfig::create(this);
            logger_config_ = LoggerConfig::create();
            tf_config_ = TFConfig::create();
            max_infer_running_config_ = MaxInferRunningConfig::create();
            common_config_parameter_.registerGroup(*control_config_);
            common_config_parameter_.registerGroup(*shoot_config_);
            common_config_parameter_.registerGroup(*record_config_);
            common_config_parameter_.registerGroup(*logger_config_);
            common_config_parameter_.registerGroup(*tf_config_);
            common_config_parameter_.registerGroup(*max_infer_running_config_);
            common_config_parameter_.reloadFromOldPath();
            auto config = common_config_parameter_.getConfig();
            debug_fps_ = config["debug_fps"].as<int>();
            attack_mode_ = config["attack_mode"].as<int>();
            detect_color_ = config["detect_color"].as<int>();

            wust_vl::common::utils::ParameterManager::instance().registerParameter(
                common_config_parameter_
            );
            YAML::Node camera_config = YAML::LoadFile(camera_config_);
            camera_ = std::make_shared<wust_vl::video::Camera>();
            camera_->init(camera_config);
            camera_->setFrameCallback(
                std::bind(&VisionBase::frameCallback, this, std::placeholders::_1)
            );
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

            thread_pool_ = std::make_unique<wust_vl::common::concurrency::ThreadPool>(
                max_infer_running_config_->max_infer_running
            );
            motion_buffer_ =
                std::make_shared<wust_vl::common::utils::MotionBufferGeneric<CarMotion, 1024>>();

            timer_ = std::make_unique<wust_vl::common::utils::Timer>("solve");

            WUST_MAIN("main") << "vision init already!";
        } catch (std::exception& e) {
            std::cerr << "init exception: " << e.what() << std::endl;
        }
        return true;
    }
    void start() {
        run_flag_ = true;
        camera_->start();
        for (auto& module: modules_) {
            if (module.second) {
                module.second->start();
            }
        }
        if (timer_) {
            const auto timercallback =
                std::bind(&VisionBase::timerCallback, this, std::placeholders::_1);
            const double rate_hz = control_config_->control_rate_param.get();
            timer_->start(rate_hz, timercallback);
        }
        if (serial_) {
            serial_->start();
        } else if (rotate_reader_) {
            rotate_reader_->replay([this](const Eigen::Vector3d& ypr) {
                ReceiveAimINFO aim_data;
                aim_data.yaw = ypr(0);
                aim_data.pitch = ypr(1);
                aim_data.roll = ypr(2);
                this->processAimData(aim_data);
            });
        }
        if (debug_mode_) {
            debug_thread_ = std::thread([this]() { this->debugThread(); });
        }
        if (rotate_writer_) {
            rotate_writer_->start();
        }
        if (img_writer_) {
            img_writer_->start();
        }
    }
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
    void frameCallback(wust_vl::video::ImageFrame& img_frame) {
        if (!run_flag_ || infer_running_count_ >= max_infer_running_config_->max_infer_running) {
            return;
        }
        if (img_frame.src_img.empty()) {
            return;
        }

        thread_pool_->enqueue([this, img_frame = std::move(img_frame)]() mutable {
            infer_running_count_++;
            CommonFrame frame;
            frame.detect_color = detect_color_;
            frame.img_frame = std::move(img_frame);
            frame.expanded =
                cv::Rect(0, 0, frame.img_frame.src_img.cols, frame.img_frame.src_img.rows);
            frame.offset = cv::Point2f(0, 0);
            frame.any_ctx = VisionCtx { .motion_buffer = motion_buffer_,
                                        .camera = camera_,
                                        .communication_delay_μs =
                                            control_config_->communication_delay_us_param.get(),
                                        .mode = attack_mode_ };
            if (frame.img_frame.src_img.empty()) {
                infer_running_count_--;
                return;
            }

            if (img_writer_) {
                img_writer_->push(frame.img_frame.src_img);
            }
            typename Mode::AttackMode mode = Mode::toAttackMode(attack_mode_);
            auto module = modules_.at(mode);
            if (module) {
                module->pushInput(frame);
            }

            infer_running_count_--;
        });
    }

    void checkStateMatchMode() const {
        const auto mode = Mode::toAttackMode(attack_mode_);

        auto this_module = modules_.at(mode);
        if (!this_module) {
            return;
        }
        auto this_thread = this_module->getThread();

        auto pause_others = [&]() {
            auto self = this_module;

            for (auto& [_, module]: modules_) {
                if (!module) {
                    continue;
                }
                if (module != self) {
                    if (auto t = module->getThread()) {
                        t->pause();
                    }
                }
            }
        };

        if (!this_thread) {
            pause_others();
            return;
        }

        if (!this_thread->isAlive()) {
            this_thread->resume();
        }

        pause_others();
    }
    void timerCallback(double dt_ms) {
        if (!run_flag_) {
            return;
        }

        GimbalCmd cmd;
        try {
            typename Mode::AttackMode mode = Mode::toAttackMode(attack_mode_);
            auto module = modules_.at(mode);
            if (!module) {
                return;
            }
            cmd = module->solve(bullet_speed_);
        } catch (const std::exception& e) {
            std::cout << "solve error: " << e.what() << std::endl;
        }
        if (!cmd.isValid()) {
            return;
        }
        last_cmd_ = cmd;

        double cmd_pitch = cmd.pitch;
        double cmd_yaw = cmd.yaw;
        SendRobotCmdData send_data;
        send_data.cmd_ID = ID_ROBOT_CMD;
        send_data.time_stamp = std::chrono::duration_cast<std::chrono::milliseconds>(
                                   std::chrono::steady_clock::now().time_since_epoch()
        )
                                   .count();
        if (cmd.distance > 0.5) {
            send_data.appear = cmd.appear;
        } else {
            send_data.appear = false;
        }

        send_data.detect_color = detect_color_;
        send_data.pitch = cmd_pitch + cmd.v_pitch * control_config_->pitch_ramp_param.get();
        send_data.yaw = cmd_yaw + cmd.v_yaw * control_config_->yaw_ramp_param.get();
        send_data.v_pitch = cmd.v_pitch;
        send_data.v_yaw = cmd.v_yaw;
        send_data.a_pitch = cmd.a_pitch;
        send_data.a_yaw = cmd.a_yaw;
        send_data.target_yaw = cmd.target_yaw;
        send_data.target_pitch = cmd.target_pitch;
        send_data.enable_pitch_diff = cmd.enable_pitch_diff;
        send_data.enable_yaw_diff = cmd.enable_yaw_diff;
        send_data.shoot_rate = shoot_config_->rate_param.get();
        if (serial_) {
            serial_->write(std::move(wust_vl::common::drivers::toVector(send_data)));
        }
    }
    bool isWebRunning() {
        static std::atomic<bool> cached { true };
        static std::atomic<long long> last_check_ms { 0 };

        const long long now_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                                     std::chrono::steady_clock::now().time_since_epoch()
        )
                                     .count();
        if (last_check_ms.load() == 0) {
            last_check_ms = now_ms;
            return true;
        }

        if (now_ms - last_check_ms.load() >= 1000) {
            const int ret = std::system("pgrep -x wust_vision_web > /dev/null 2>&1");
            cached = (ret == 0);
            last_check_ms = now_ms;
        }

        return cached.load();
    }
    void debugThread() {
        const double us_interval = 1e6 / static_cast<double>(debug_fps_);
        const auto kInterval = std::chrono::microseconds(static_cast<int64_t>(us_interval));
        while (run_flag_) {
            const auto start_time = std::chrono::steady_clock::now();
            do {
                try {
                    if (!isWebRunning()) {
                        break;
                    }
                    typename Mode::AttackMode mode = Mode::toAttackMode(attack_mode_);
                    auto module = modules_.at(mode);
                    if (module) {
                        module->doDebug();
                    }

                    // utils::XSecOnce(
                    //     [this]() {
                    //         wust_vl::common::utils::ParameterManager::instance()
                    //             .allReloadFromOldPath();
                    //     },
                    //     1.0
                    // );

                } catch (std::exception& e) {
                    std::cout << "debug thread error: " << e.what() << std::endl;
                }
            } while (0);

            const auto elapsed = std::chrono::steady_clock::now() - start_time;
            if (elapsed < kInterval) {
                std::this_thread::sleep_for(kInterval - elapsed);
            }
        }
    }

    void processAimData(const ReceiveAimINFO& aim_data) {
        static wust_vl::common::concurrency::Averager<double> vyaw_avg(100);
        if (!this->run_flag_) {
            return;
        }
        detect_color_ = aim_data.detect_color;
        bullet_speed_ = aim_data.bullet_speed;
        const double roll = -(aim_data.roll) * M_PI / 180.0;
        const double pitch = (aim_data.pitch) * M_PI / 180.0;
        const double yaw = (aim_data.yaw) * M_PI / 180.0;
        const double v_roll = aim_data.roll_vel * M_PI / 180.0;
        const double v_pitch = aim_data.pitch_vel * M_PI / 180.0;
        const double v_yaw = aim_data.yaw_vel * M_PI / 180.0;
        vyaw_avg.add(v_yaw);

        const double v_x = 0.0;
        const double v_y = 0.0;
        const double v_z = 0.0;

        const auto now = std::chrono::steady_clock::now();
        if (motion_buffer_) {
            CarMotion motion { yaw, pitch, roll, 0.0, v_pitch, v_roll, v_x, v_y, v_z };
            motion_buffer_->push(motion, now);
        }
        if (debug_mode_) {
            updateSerialLog(aim_data);
            flushSerialLog();
        }

        static auto last_push_time = std::chrono::steady_clock::now();
        const auto elapsed =
            std::chrono::duration_cast<std::chrono::milliseconds>(now - last_push_time).count();
        if (elapsed >= 10) { // 至少间隔 10ms（100Hz）
            if (rotate_writer_) {
                rotate_writer_->push(Eigen::Vector3d(aim_data.yaw, aim_data.pitch, aim_data.roll));
            }
            last_push_time = now;
        }
    }
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
                base->bullet_speed_ = bullet_speed_param.get();
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
    double bullet_speed_;
    wust_vl::common::utils::Parameter common_config_parameter_;
    std::unique_ptr<wust_vl::common::concurrency::ThreadPool> thread_pool_;
    std::map<typename Mode::AttackMode, IModule::Ptr> modules_;
    std::shared_ptr<wust_vl::video::Camera> camera_;
    std::shared_ptr<wust_vl::common::drivers::SerialDriver> serial_;
    std::unique_ptr<wust_vl::common::utils::Timer> timer_;
    std::shared_ptr<wust_vl::common::utils::MotionBufferGeneric<CarMotion, 1024>> motion_buffer_;
    std::shared_ptr<wust_vl::common::utils::Recorder<Eigen::Vector3d>> rotate_writer_;
    RotateReaderCSV::RotateReaderCSVPtr rotate_reader_;
    std::shared_ptr<wust_vl::common::utils::Recorder<cv::Mat>> img_writer_;
    std::thread debug_thread_;
    GimbalCmd last_cmd_;
    std::pair<cv::Mat, cv::Mat> camera_info_;
    bool run_flag_ = false;
    bool debug_mode_ = false;
    std::atomic<int> infer_running_count_ { 0 };
    std::string common_config_;
    std::string camera_config_;
    std::string auto_aim_config_;
    std::string auto_buff_config_;
};
} // namespace wust_vision