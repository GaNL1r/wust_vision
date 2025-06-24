#include "detect/armor_detector_openvino_wrapper.hpp"

ArmorDetectorOpenvinoWrapper::ArmorDetectorOpenvinoWrapper(
    const YAML::Node &config) {
  auto classify_model_path = config["classify_model_path"].as<std::string>();
  auto classify_label_path = config["classify_label_path"].as<std::string>();
  const std::string model_path =
      config["model"]["model_path"].as<std::string>();
  auto device_type = config["model"]["device_type"].as<std::string>();
  float conf_threshold = config["model"]["conf_threshold"].as<float>();
  int top_k = config["model"]["top_k"].as<int>();
  float nms_threshold = config["model"]["nms_threshold"].as<float>();
  float expand_ratio_w = config["light"]["expand_ratio_w"].as<float>();
  float expand_ratio_h = config["light"]["expand_ratio_h"].as<float>();
  int binary_thres = config["light"]["binary_thres"].as<int>();
  WUST_INFO("armor_detector")<<"model_path: "<<model_path<<"device_type: "<<device_type;
  LightParams l_params = {
      .min_ratio = config["light"]["min_ratio"].as<double>(),
      .max_ratio = config["light"]["max_ratio"].as<double>(),
      .max_angle = config["light"]["max_angle"].as<double>()};

  detector_ = std::make_unique<ArmorDetectOpenVino>(
      model_path, classify_model_path, classify_label_path, device_type,
      l_params, conf_threshold, top_k, nms_threshold, expand_ratio_w,
      expand_ratio_h, binary_thres);
}

ArmorDetectorOpenvinoWrapper::~ArmorDetectorOpenvinoWrapper() = default;

void ArmorDetectorOpenvinoWrapper::pushInput(
    const cv::Mat &rgb_img, std::chrono::steady_clock::time_point timestamp,
    Eigen::Matrix4d T_camera_to_odom) {
  detector_->pushInput(rgb_img, timestamp, T_camera_to_odom);
}

void ArmorDetectorOpenvinoWrapper::setCallback(DetectorCallback cb) {
  detector_->setCallback(std::move(cb));
}