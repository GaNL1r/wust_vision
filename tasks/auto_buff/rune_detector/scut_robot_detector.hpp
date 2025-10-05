#pragma once
#include "tasks/auto_buff/rune_optimize/ba_solver.hpp"
#include "tasks/auto_buff/type.hpp"
#include "vc/contour_proc/contour_wrapper.hpp"
#include "vc/core/debug_tools.h"
#include "vc/core/debug_tools/param_view_manager.h"
#include "vc/core/debug_tools/window_auto_layout.h"
#include "vc/core/yml_manager.hpp"
#include "vc/detector/rune_detector.h"
#include "vc/feature/rune_center.h"
#include "vc/feature/rune_combo.h"
#include "vc/feature/rune_fan.h"
#include "vc/feature/rune_fan_active.h"
#include "vc/feature/rune_fan_hump.h"
#include "vc/feature/rune_fan_inactive.h"
#include "vc/feature/rune_group.h"
#include "vc/feature/rune_target_active.h"
#include "vc/feature/rune_target_inactive.h"
class ScutRobotDetector {
public:
    ScutRobotDetector(const std::pair<cv::Mat, cv::Mat>& camera_info, const YAML::Node& config);
    rune::RuneFan detect(
        const CommonFrame& frame,
        Eigen::Vector3d gimbal,
        Eigen::Matrix4d T_camera_to_odom,
        bool debug,
        cv::Mat& debug_img
    );
    std::pair<cv::Mat, cv::Mat> camera_info_;
    std::vector<FeatureNode_ptr> rune_groups_ {};
    std::unique_ptr<rune::BaSolver> ba_solver_;
    static inline std::unique_ptr<ScutRobotDetector>
    make_detector(const std::pair<cv::Mat, cv::Mat>& camera_info, const YAML::Node& config) {
        return std::make_unique<ScutRobotDetector>(camera_info, config);
    }
};