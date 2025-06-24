#pragma once
#include "detect/armor_detector_base.hpp"
#include "detect/armor_detector_openvino.hpp"
#include <yaml-cpp/yaml.h>

class ArmorDetectorOpenvinoWrapper : public ArmorDetectorBase {
public:
  ArmorDetectorOpenvinoWrapper(const YAML::Node &config);
  ~ArmorDetectorOpenvinoWrapper() override;

  void pushInput(const cv::Mat &rgb_img,
                 std::chrono::steady_clock::time_point timestamp,
                 Eigen::Matrix4d T_camera_to_odom) override;

  void setCallback(DetectorCallback cb) override;

private:
  std::unique_ptr<ArmorDetectOpenVino> detector_;
};
