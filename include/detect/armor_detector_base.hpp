// detector_base.hpp
#pragma once
#include "type/type.hpp"
#include <opencv2/core/mat.hpp>

class ArmorDetectorBase {
public:
  virtual ~ArmorDetectorBase() = default;

  virtual void pushInput(const cv::Mat &rgb_img,
                         std::chrono::steady_clock::time_point timestamp,
                         Eigen::Matrix4d T_camera_to_odom) = 0;

  using DetectorCallback = std::function<void(
      const std::vector<ArmorObject> &, std::chrono::steady_clock::time_point,
      const cv::Mat &, Eigen::Matrix4d)>;

  virtual void setCallback(DetectorCallback cb) = 0;
};
