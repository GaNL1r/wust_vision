#pragma once
#include "3rdparty/backward-cpp/backward.hpp"
#include "main_base.hpp"
#include "sinple_img_rotate_saver.hpp"
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
    void updateBulletSpeed(double bullet_speed);
    void processAimData(const ReceiveAimINFO& aim_data);
    std::unique_ptr<ThreadPool> thread_pool_;
    std::unique_ptr<auto_aim::AutoAim> auto_aim_;
    std::unique_ptr<auto_buff::AutoBuff> auto_buff_;
    std::shared_ptr<wust_vl_video::Camera> camera_;
    std::shared_ptr<SerialDriver> serial_;
    std::unique_ptr<Timer> timer_;
    std::shared_ptr<auto_aim::AutoAimShared> auto_aim_shared_;
    std::shared_ptr<auto_buff::AutoBuffShared> auto_buff_shared_;
    std::shared_ptr<MotionBufferGeneric<Motion, 1024>> motion_buffer_;
    std::shared_ptr<wust_vl_utils::ConfigBinder> config_binder_;
    std::shared_ptr<wust_vl_utils::ConfigBinder> auto_aim_config_binder_;
    std::shared_ptr<wust_vl::Recorder<Eigen::Vector3d>> rotate_writer_;
    RotateReaderCSV::RotateReaderCSVPtr rotate_reader_;
    std::shared_ptr<wust_vl::Recorder<cv::Mat>> img_writer_;
    std::thread debug_thread_;
    YAML::Node config_;
    Eigen::Matrix3d R_camera2gimbal_;
    Eigen::Vector3d t_camera2gimbal_;
    GimbalCmd last_cmd_;

    std::unique_ptr<Averager<double>> pitch_avg_;
    std::pair<cv::Mat, cv::Mat> camera_info_;
    double bullet_speed_;
    int attack_mode_;
    int max_infer_running_;
    bool run_flag_ = false;
    int detect_color_ = 0;
    bool debug_mode_ = false;
    int use_ncnn_count_ = 0;
    int shoot_rate_ = 3;
    std::atomic<int> infer_running_count_ { 0 };
    std::string common_config_;
    std::string camera_config_;
    std::string auto_aim_config_;
    std::string auto_buff_config_;
};
