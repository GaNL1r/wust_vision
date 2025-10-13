#pragma once
#include "3rdparty/backward-cpp/backward.hpp"
#include "tasks/auto_aim/auto_aim.hpp"
#include "tasks/auto_buff/auto_buff.hpp"
#include "tasks/utils.hpp"
#include "wust_vl/common/drivers/serial_driver.hpp"
#include "wust_vl/common/utils/config_binder.hpp"
#include <sys/mman.h>
#include <wust_vl/wust_vl.hpp>
/**
 * @class VisionBase
 * @brief 视觉系统的核心类，负责初始化、运行、停止和数据交互。
 *
 *
 * @note 为保证线程安全，init为阻塞式，而start只开放标志位启动各模块线程，非阻塞，在main中通过SignalHandler控制stop
 */
class VisionBase {
public:
    /**
     * @brief 构造函数
     * @param[in] common_config    公共配置文件路径
     * @param[in] camera_config    相机配置文件路径
     * @param[in] auto_aim_config  自瞄模块配置文件路径
     * @param[in] auto_buff_config 能量机关模块配置文件路径
     */
    VisionBase(
        std::string common_config,
        std::string camera_config,
        std::string auto_aim_config,
        std::string auto_buff_config
    );
    ~VisionBase();
    /**
     * @brief 显式停止运行并释放资源
     */
    void stop();
    /**
     * @brief 初始化各模块
     * @return true 表示初始化成功，否则抛出异常
     */
    bool init(bool debug_mode);
    /**
     * @brief 启动所有线程和模块
     */
    void start();
    /**
     * @brief 串口接收回调，本类中主要接收下位机发送机器人运动信息和顶层决策信息 
     */

    void serialCallback(const uint8_t* data, std::size_t len);
    /**
     * @brief wust_vl_video相机封装接收回调
     * @note 该函数嵌入相机视频流线程，为保证帧率，应尽早return，防阻塞
     */
    void frameCallback(wust_vl_video::ImageFrame& frame);
    /**
     * @brief 对于使用wust_vl_concurrency::MonitoredThread对auto_aim/auto_buff异步解析线程的控制，防止空转占用资源
     * @note 此函数与在主循环进行监控，防止嵌入线程卡死
     */
    void checkStateMatchMode();
    /**
     * @brief 定时器回调,主要作为控制解算函数
     * @param[in] dt_ms 与上次回调的时间间隔(ms)
     */
    void timerCallback(double dt_ms);
    /**
     * @brief debug主循环,默认30fps,负责绘制/写入可视化图像，写入绘图log数据与参数绑定器重加载
     */
    void debugThread();
    void autoExposureControl(const cv::Mat& frame);
    std::unique_ptr<ThreadPool> thread_pool_;
    std::unique_ptr<auto_aim::AutoAim> auto_aim_;
    std::unique_ptr<auto_buff::AutoBuff> auto_buff_;
    std::unique_ptr<wust_vl_video::Camera> camera_;
    std::shared_ptr<SerialDriver> serial_;
    std::unique_ptr<Timer> timer_;
    std::shared_ptr<auto_aim::AutoAimShared> auto_aim_shared_;
    std::shared_ptr<auto_buff::AutoBuffShared> auto_buff_shared_;
    std::shared_ptr<MotionBufferGeneric<Motion, 1024>> motion_buffer_;
    std::shared_ptr<wust_vl_utils::ConfigBinder> config_binder_;
    std::shared_ptr<wust_vl_utils::ConfigBinder> auto_aim_config_binder_;
    std::thread debug_thread_;
    YAML::Node config_;
    Eigen::Matrix3d R_camera2gimbal_;
    Eigen::Vector3d t_camera2gimbal_;
    GimbalCmd last_cmd_;
    std::unique_ptr<Averager<double>> yaw_avg_;
    std::unique_ptr<Averager<double>> pitch_avg_;
    std::pair<cv::Mat, cv::Mat> camera_info_;
    int attack_mode_;
    int max_infer_running_;
    bool run_flag_ = true;
    int detect_color_ = 0;
    bool debug_mode_ = false;
    int use_ncnn_count_ = 0;
    std::atomic<int> infer_running_count_ { 0 };
    std::string common_config_;
    std::string camera_config_;
    std::string auto_aim_config_;
    std::string auto_buff_config_;
    struct AutoExposureCfg {
        bool enable = false;
        double target_brightness = 20.0;
        double tolerance = 5.0;
        double step_gain = 10.0;
        double decay_step = 1.0;
        double exposure_min = 100.0;
        double exposure_max = 2500.0;
        double control_interval_ms = 300;

        // 成员函数：从 YAML 加载配置
        bool loadFromYaml(const YAML::Node& root) {
            try {
                if (root["enable"])
                    enable = root["enable"].as<bool>();
                if (root["target_brightness"])
                    target_brightness = root["target_brightness"].as<double>();
                if (root["tolerance"])
                    tolerance = root["tolerance"].as<double>();
                if (root["step_gain"])
                    step_gain = root["step_gain"].as<double>();
                if (root["decay_step"])
                    decay_step = root["decay_step"].as<double>();
                if (root["exposure_min"])
                    exposure_min = root["exposure_min"].as<double>();
                if (root["exposure_max"])
                    exposure_max = root["exposure_max"].as<double>();
                if (root["control_interval_ms"])
                    control_interval_ms = root["control_interval_ms"].as<double>();

                return true;
            } catch (const YAML::Exception& e) {
                std::cerr << "加载 YAML 配置失败: " << e.what() << std::endl;
                return false;
            }
        }
    } auto_exposure_cfg_;
};
template<typename T>
concept VisionLike = requires(T v) {
    {
        v.init(std::declval<bool>())
        } -> std::same_as<bool>;
    {
        v.start()
        } -> std::same_as<void>;
    {
        v.stop()
        } -> std::same_as<void>;
    {
        v.checkStateMatchMode()
        } -> std::same_as<void>;
};

template<VisionLike T>
inline int runVisionMain(int argc, char** argv) {
    bool debug = false;
    if (argc > 1) {
        std::string firstArg = argv[1];
        debug = (firstArg == "true" || firstArg == "1");
        std::cout << "debug: " << firstArg << std::endl;
    }
    std::set_terminate([]() {
        std::cerr << "Uncaught exception, terminating program.\n";
        if (auto e = std::current_exception()) {
            try {
                std::rethrow_exception(e);
            } catch (const std::exception& ex) {
                std::cerr << "Exception: " << ex.what() << std::endl;
            } catch (...) {
                std::cerr << "Unknown exception" << std::endl;
            }
        }
        std::abort();
    });

    try {
        int exit_code = 0;

        {
            T v;
            v.init(debug);
            v.start();

            SignalHandler sig;
            sig.start([&] {});

            bool exit_flag = false;

            while (!sig.shouldExit() && !exit_flag) {
                wust_vl_concurrency::ThreadManager::instance().printStatus();
                auto all_status =
                    wust_vl_concurrency::ThreadManager::instance().getAllThreadStatuses();
                v.checkStateMatchMode();

                for (auto& status: all_status) {
                    if (status.second == wust_vl_concurrency::MonitoredThread::Status::Hung) {
                        std::cerr << status.first << " is Hunging! Exiting program..." << std::endl;
                        exit_flag = true;
                        exit_code = -1;
                        break;
                    }
                }
                std::this_thread::sleep_for(std::chrono::milliseconds(1000));
            }
            auto stop_future = std::async(std::launch::async, [&v]() { v.stop(); });
            if (stop_future.wait_for(std::chrono::seconds(5)) == std::future_status::timeout) {
                std::cerr << "v.stop() timed out, forcing exit!" << std::endl;
                std::exit(1);
            }
            std::cout << "v.stop() finished, v will be destructed now." << std::endl;
        }

        std::cout << "Exiting program..." << std::endl;
        return exit_code;

    } catch (const std::exception& e) {
        std::cerr << "Caught exception in main: " << e.what() << "\n";
        throw;
        return -1;
    } catch (...) {
        std::cerr << "Unknown exception caught in main!\n";
        return -1;
    }
}

#define VISION_MAIN(VISION_TYPE) \
    int main(int argc, char** argv) { \
        return runVisionMain<VISION_TYPE>(argc, argv); \
    }

#define ENABLE_BACKWARD() \
    namespace backward { \
        static backward::SignalHandling sh; \
    }