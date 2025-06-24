#pragma once
#include "detect/armor_detector_base.hpp"
#include "detect/armor_detector_trt.hpp"
#include <yaml-cpp/yaml.h>

class ArmorDetectorTrtWrapper : public ArmorDetectorBase {
public:
  ArmorDetectorTrtWrapper(const YAML::Node &config);
  ~ArmorDetectorTrtWrapper() override;

  void pushInput(const cv::Mat &rgb_img,
                 std::chrono::steady_clock::time_point timestamp,
                 Eigen::Matrix4d T_camera_to_odom) override;

  void setCallback(DetectorCallback cb) override;

private:
  std::unique_ptr<ArmorDetectTrt> detector_;
};
