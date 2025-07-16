#include "common/camera_calibrator.hpp"
#include "driver/hik.hpp"
#include "yaml-cpp/yaml.h"
CameraCalibrator calibrator;
HikCamera camera;
int main() {
    calibrator = CameraCalibrator(cv::Size(9, 6), 25.0f, 15, 20.0f, "./calib_imgs");
    auto config = YAML::LoadFile("/home/hy/wust_vision/config/config_common.yaml");
    std::string target_sn = config["camera"]["target_sn"].as<std::string>();
    if (!camera.initializeCamera(target_sn)) {
        WUST_ERROR("vision_logger") << "Camera initialization failed.";
        return 0;
    }

    camera.setParameters(
        config["camera"]["acquisition_frame_rate"].as<int>(),
        config["camera"]["exposure_time"].as<int>(),
        config["camera"]["gain"].as<double>(),
        config["camera"]["adc_bit_depth"].as<std::string>(),
        config["camera"]["pixel_format"].as<std::string>(),
        config["camera"]["acquisitionFrameRateEnable"].as<bool>(),
        config["camera"]["reverse_x"].as<bool>(false),
        config["camera"]["reverse_y"].as<bool>(false)
    );
    camera.enableTrigger(TriggerType::Software, "Software", 0);
    camera.startCamera(false);
    while (true) {
        ImageFrame frame = camera.readImage();
        if (frame.data.empty()) {
            WUST_ERROR("vision_logger") << "Failed to read image from camera.";
            continue;
        }
        auto now = std::chrono::steady_clock::now();
        cv::Mat img = convertToMatrgb(frame);
        if (!calibrator.processFrame(img))
            break;
        auto end = std::chrono::steady_clock::now();
        WUST_INFO("vision_logger")
            << "Calibration time: "
            << std::chrono::duration_cast<std::chrono::milliseconds>(end - now).count() << "ms";
    }
    if (calibrator.isCalibrated()) {
        calibrator.saveToFile("camera_intrinsic.yaml");
    }
}