// rune_detector_base.hpp
#pragma once
#include "type/type.hpp"
#include <opencv2/core/mat.hpp>

class RuneDetectorBase {
public:
  virtual ~RuneDetectorBase() = default;

  using CallbackType = std::function<void(std::vector<RuneObject> &,
                                          std::chrono::steady_clock::time_point,
                                          const cv::Mat &, Eigen::Matrix4d)>;

  virtual void pushInput(const cv::Mat &rgb_img,
                         std::chrono::steady_clock::time_point timestamp,
                         Eigen::Matrix4d T_camera_to_odom) = 0;

  virtual void setCallback(CallbackType cb) = 0;
  virtual std::tuple<cv::Point2f, cv::Mat>
  detectRTag(const cv::Mat &img, int binary_thresh,
             const cv::Point2f &prior) = 0;
};
