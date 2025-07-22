#include "common/camera_calibrator.hpp"
#include "common/logger.hpp"
#include "driver/hik.hpp"
#include "yaml-cpp/yaml.h"
#include <pwd.h>

int main() {
    CameraCalibrator calibrator;
    HikCamera camera;
    auto config = YAML::LoadFile("/home/hy/wust_vision/config/camera_calibrator.yaml");
    int size_w = config["calibrator"]["size_w"].as<int>();
    int size_h = config["calibrator"]["size_h"].as<int>();
    float square_size = config["calibrator"]["square_size"].as<float>();
    int required_views = config["calibrator"]["requiredviews"].as<int>();
    float min_shift = config["calibrator"]["minshift"].as<float>();
    calibrator = CameraCalibrator(cv::Size(size_w, size_h), square_size, required_views, min_shift);

    std::string target_sn = config["camera"]["target_sn"].as<std::string>();
    if (!camera.initializeCamera(target_sn)) {
        WUST_ERROR("vision_logger") << "Camera initialization failed.";
        return 0;
    }

    camera.setParameters(
        config["camera"]["acquisition_frame_rate"].as<int>(),
        config["camera"]["exposure_time"].as<int>(),
        config["camera"]["gain"].as<double>(),
        config["camera"]["gamma"].as<double>(),
        config["camera"]["adc_bit_depth"].as<std::string>(),
        config["camera"]["pixel_format"].as<std::string>(),
        config["camera"]["acquisitionFrameRateEnable"].as<bool>(),
        config["camera"]["reverse_x"].as<bool>(false),
        config["camera"]["reverse_y"].as<bool>(false)
    );
    camera.enableTrigger(TriggerType::Software, "Software", 0);
    camera.startCamera(false);
    const char* home = nullptr;

    // 尝试从 SUDO_USER 获取真实用户 home
    const char* sudo_user = std::getenv("SUDO_USER");
    if (sudo_user) {
        struct passwd* pw = getpwnam(sudo_user);
        if (pw) {
            home = pw->pw_dir;
        }
    }

    // 如果不是 sudo，使用 getuid 获取 home
    if (!home) {
        struct passwd* pw = getpwuid(getuid());
        if (pw) {
            home = pw->pw_dir;
        }
    }

    if (!home) {
        throw std::runtime_error("HOME environment variable not set.");
    }

    namespace fs = std::filesystem;
    std::filesystem::path save_path_ = fs::path(home) / "wust_data/calibrator/"
        / std::string(std::to_string(std::time(nullptr)) + ".yaml");
    if (!std::filesystem::exists(save_path_)) {
        std::filesystem::create_directories(save_path_.parent_path());
    }
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
        calibrator.saveToFile(save_path_);
    }
}
