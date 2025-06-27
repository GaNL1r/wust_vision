#include "detect/rune_detector_openvino_wrapper.hpp"

RuneDetectorOpenvinoWrapper::RuneDetectorOpenvinoWrapper(
    const YAML::Node &config) {

  std::string model_path = config["rune_detector"]["model"].as<std::string>();
  std::string device_type =
      config["rune_detector"]["device_type"].as<std::string>("CPU");

  float conf_threshold =
      config["rune_detector"]["confidence_threshold"].as<float>(0.50);
  int top_k = config["rune_detector"]["top_k"].as<int>(128);
  float nms_threshold = config["rune_detector"]["nms_threshold"].as<float>(0.3);
  WUST_INFO("rune_detector")
      << "model_path: " << model_path << "device_type: " << device_type;
  rune_detector_ = std::make_unique<RuneDetectorOpenvino>(
      model_path, device_type, conf_threshold, top_k, nms_threshold);
  rune_detector_->init();
}

RuneDetectorOpenvinoWrapper::~RuneDetectorOpenvinoWrapper() = default;

void RuneDetectorOpenvinoWrapper::pushInput(
    const cv::Mat &rgb_img, std::chrono::steady_clock::time_point timestamp,
    Eigen::Matrix4d T_camera_to_odom) {
  rune_detector_->pushInput(rgb_img, timestamp, T_camera_to_odom);
}

void RuneDetectorOpenvinoWrapper::setCallback(CallbackType cb) {
  rune_detector_->setCallback(std::move(cb));
}

std::tuple<cv::Point2f, cv::Mat>
RuneDetectorOpenvinoWrapper::detectRTag(const cv::Mat &img, int binary_thresh,
                                        const cv::Point2f &prior) {
  return rune_detector_->detectRTag(img, binary_thresh, prior);
}
