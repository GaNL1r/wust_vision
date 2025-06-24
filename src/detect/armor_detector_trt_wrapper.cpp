#include "detect/armor_detector_trt_wrapper.hpp"

ArmorDetectorTrtWrapper::ArmorDetectorTrtWrapper(const YAML::Node &config) {

  auto classify_model_path = config["classify_model_path"].as<std::string>();
  auto classify_label_path = config["classify_label_path"].as<std::string>();
  const std::string model_path =
      config["model"]["model_path"].as<std::string>();
  float expand_ratio_w = config["light"]["expand_ratio_w"].as<float>();
  float expand_ratio_h = config["light"]["expand_ratio_h"].as<float>();
  int binary_thres = config["light"]["binary_thres"].as<int>();

  ArmorDetectTrt::Params params;
  params.input_w = config["model"]["input_w"].as<int>();
  params.input_h = config["model"]["input_h"].as<int>();
  params.num_classes = config["model"]["num_classes"].as<int>();
  params.num_colors = config["model"]["num_colors"].as<int>();
  params.conf_threshold = config["model"]["conf_threshold"].as<float>();
  params.nms_threshold = config["model"]["nms_threshold"].as<float>();
  params.top_k = config["model"]["top_k"].as<int>();
  WUST_INFO("armor_detector") << "model_path: " << model_path;
  LightParams l_params = {
      .min_ratio = config["light"]["min_ratio"].as<double>(),
      .max_ratio = config["light"]["max_ratio"].as<double>(),
      .max_angle = config["light"]["max_angle"].as<double>()};

  detector_ = std::make_unique<ArmorDetectTrt>(
      model_path, params, expand_ratio_h, expand_ratio_w, binary_thres,
      l_params, classify_model_path, classify_label_path);
}

ArmorDetectorTrtWrapper::~ArmorDetectorTrtWrapper() = default;

void ArmorDetectorTrtWrapper::pushInput(
    const cv::Mat &rgb_img, std::chrono::steady_clock::time_point timestamp,
    Eigen::Matrix4d T_camera_to_odom) {
  detector_->pushInput(rgb_img, timestamp, T_camera_to_odom);
}

void ArmorDetectorTrtWrapper::setCallback(DetectorCallback cb) {
  detector_->setCallback(std::move(cb));
}
