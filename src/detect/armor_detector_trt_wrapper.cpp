#include "detect/armor_detector_trt_wrapper.hpp"

ArmorDetectorTrtWrapper::ArmorDetectorTrtWrapper(const YAML::Node &config) {

  auto classify_model_path =
      config["armor_detect"]["classify_model_path"].as<std::string>();
  auto classify_label_path =
      config["armor_detect"]["classify_label_path"].as<std::string>();
  const std::string model_path =
      config["armor_detect"]["model"]["model_path"].as<std::string>();
  float expand_ratio_w =
      config["armor_detect"]["light"]["expand_ratio_w"].as<float>();
  float expand_ratio_h =
      config["armor_detect"]["light"]["expand_ratio_h"].as<float>();
  int binary_thres = config["armor_detect"]["light"]["binary_thres"].as<int>();

  ArmorDetectTrt::Params params;
  params.input_w = config["armor_detect"]["model"]["input_w"].as<int>();
  params.input_h = config["armor_detect"]["model"]["input_h"].as<int>();
  params.num_classes = config["armor_detect"]["model"]["num_classes"].as<int>();
  params.num_colors = config["armor_detect"]["model"]["num_colors"].as<int>();
  params.conf_threshold =
      config["armor_detect"]["model"]["conf_threshold"].as<float>();
  params.nms_threshold =
      config["armor_detect"]["model"]["nms_threshold"].as<float>();
  params.top_k = config["armor_detect"]["model"]["top_k"].as<int>();
  WUST_INFO("armor_detector") << "model_path: " << model_path;
  LightParams l_params = {
      .min_ratio = config["armor_detect"]["light"]["min_ratio"].as<double>(),
      .max_ratio = config["armor_detect"]["light"]["max_ratio"].as<double>(),
      .max_angle = config["armor_detect"]["light"]["max_angle"].as<double>()};
  ArmorParams a_params = {
      .min_light_ratio =
          config["armor_detect"]["armor"]["min_light_ratio"].as<double>(),
      .min_small_center_distance =
          config["armor_detect"]["armor"]["min_small_center_distance"]
              .as<double>(),
      .max_small_center_distance =
          config["armor_detect"]["armor"]["max_small_center_distance"]
              .as<double>(),
      .min_large_center_distance =
          config["armor_detect"]["armor"]["min_large_center_distance"]
              .as<double>(),
      .max_large_center_distance =
          config["armor_detect"]["armor"]["max_large_center_distance"]
              .as<double>(),
      .max_angle = config["armor_detect"]["armor"]["max_angle"].as<double>()};

  detector_ = std::make_unique<ArmorDetectTrt>(
      model_path, params, expand_ratio_h, expand_ratio_w, binary_thres,
      l_params, a_params, classify_model_path, classify_label_path);
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
