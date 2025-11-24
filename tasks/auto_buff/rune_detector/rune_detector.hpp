#pragma once

#include "tasks/auto_buff/type.hpp"

class RuneDetectorCV {
public:
    using DetectorCallback =
        std::function<void(const rune::RuneFan&, const CommonFrame&, cv::Mat&)>;
    using Ptr = std::unique_ptr<RuneDetectorCV>;
    RuneDetectorCV(const YAML::Node& node);
    void pushInput(CommonFrame& frame, bool is_big);
    void setCallback(DetectorCallback callback) {
        callback_ = callback;
    }
    cv::Mat preProcess(const cv::Mat& src, bool use_red = true);
    inline rune::RuneCenter getRuneCenter(
        const std::vector<std::vector<cv::Point>>& contours,
        const std::vector<cv::Vec4i>& hierarchy,
        cv::Mat& debug_img,
        std::vector<bool>& used_flags
    );
    inline std::vector<rune::RunePan> markRuneTarget(
        const std::vector<std::vector<cv::Point>>& contours,
        const std::vector<cv::Vec4i>& hierarchy,
        std::vector<bool>& used_flags
    );
    inline void markInvalidContours(
        cv::Mat& color,
        cv::Mat& debug_img,
        const std::vector<std::vector<cv::Point>>& contours,
        std::vector<bool>& used_flags,
        bool filter_red = true,
        double diff_thresh = 40.0
    );
    static inline std::unique_ptr<RuneDetectorCV> make_detector(const YAML::Node& node) {
        return std::make_unique<RuneDetectorCV>(node);
    }
    DetectorCallback callback_;
    cv::Mat tmp_R_;
    int current_id_ = 0;
    struct Params {
        double rune_center_min_area = 100.0;
        double rune_center_max_area = 2000.0;
        double rune_center_1x1ratio_tol = 0.7;
        double rune_center_fill_ratio_min = 0.7;

        double rune_target_min_area = 100.0;
        double rune_target_max_area = 3000.0;
        double rune_target_max_square_ratio = 1.3;
        double rune_target_cluster_radius = 70.0;

        void load(const YAML::Node& node) {
            // center params
            rune_center_min_area = node["rune_center_min_area"]
                ? node["rune_center_min_area"].as<double>()
                : rune_center_min_area;
            rune_center_max_area = node["rune_center_max_area"]
                ? node["rune_center_max_area"].as<double>()
                : rune_center_max_area;
            rune_center_1x1ratio_tol = node["rune_center_1x1ratio_tol"]
                ? node["rune_center_1x1ratio_tol"].as<double>()
                : rune_center_1x1ratio_tol;
            rune_center_fill_ratio_min = node["rune_center_fill_ratio_min"]
                ? node["rune_center_fill_ratio_min"].as<double>()
                : rune_center_fill_ratio_min;

            // target params
            rune_target_min_area = node["rune_target_min_area"]
                ? node["rune_target_min_area"].as<double>()
                : rune_target_min_area;
            rune_target_max_area = node["rune_target_max_area"]
                ? node["rune_target_max_area"].as<double>()
                : rune_target_max_area;
            rune_target_max_square_ratio = node["rune_target_max_square_ratio"]
                ? node["rune_target_max_square_ratio"].as<double>()
                : rune_target_max_square_ratio;
            rune_target_cluster_radius = node["rune_target_cluster_radius"]
                ? node["rune_target_cluster_radius"].as<double>()
                : rune_target_cluster_radius;
        }
    } params_;
};