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
namespace wust_vision {
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
    void debugThread() const;
    void autoExposureControl(const cv::Mat& frame);
    void updateBulletSpeed(double bullet_speed);
    void processAimData(const ReceiveAimINFO& aim_data);

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
    YAML::Node config_;
    Eigen::Matrix3d R_camera2gimbal_;
    Eigen::Vector3d t_camera2gimbal_;
    GimbalCmd last_cmd_;
    double yaw_ramp_ = 0.0;
    double pitch_ramp_ = 0.0;
    std::unique_ptr<wust_vl::common::concurrency::Averager<double>> pitch_avg_;
    std::pair<cv::Mat, cv::Mat> camera_info_;
    double bullet_speed_;
    int attack_mode_;
    int max_infer_running_;
    bool run_flag_ = false;
    int detect_color_ = 0;
    bool debug_mode_ = false;
    int use_ncnn_count_ = 0;
    int shoot_rate_ = 3;
    int debug_fps_ = 30;
    std::atomic<int> infer_running_count_ { 0 };
    std::string common_config_;
    std::string camera_config_;
    std::string auto_aim_config_;
    std::string auto_buff_config_;
};
} // namespace wust_vision