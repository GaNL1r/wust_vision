#include "detect/rune_detector_trt_wrapper.hpp"

RuneDetectorTrtWrapper::RuneDetectorTrtWrapper(const YAML::Node &config) {

  std::string model_path = config["rune_detector"]["model"].as<std::string>();

  WUST_INFO("rune_detector") << "model : " << model_path;

  RuneDetectorTrt::Params params;
  params.conf_threshold =
      config["rune_detector"]["confidence_threshold"].as<float>(0.50);
  params.nms_threshold =
      config["rune_detector"]["nms_threshold"].as<float>(0.3);
  params.top_k = config["rune_detector"]["top_k"].as<int>(128);
  WUST_INFO("rune_detector")<<"model_path : "<<model_path;
  rune_detector_ = std::make_unique<RuneDetectorTrt>(model_path, params);
}

RuneDetectorTrtWrapper::~RuneDetectorTrtWrapper() = default;

void RuneDetectorTrtWrapper::pushInput(
    const cv::Mat &rgb_img, std::chrono::steady_clock::time_point timestamp,
    Eigen::Matrix4d T_camera_to_odom) {
  rune_detector_->pushInput(rgb_img, timestamp, T_camera_to_odom);
}

void RuneDetectorTrtWrapper::setCallback(CallbackType cb) {
  rune_detector_->setCallback(std::move(cb));
}

std::tuple<cv::Point2f, cv::Mat>
RuneDetectorTrtWrapper::detectRTag(const cv::Mat &img, int binary_thresh,
                                   const cv::Point2f &prior) {
  return rune_detector_->detectRTag(img, binary_thresh, prior);
}
