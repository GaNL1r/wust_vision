#pragma once

#include "tasks/auto_buff/type.hpp"

class RuneDetectorCV {
public:
    using DetectorCallback =
        std::function<void(const rune::RuneFan&, const CommonFrame&, cv::Mat&)>;
    using Ptr = std::unique_ptr<RuneDetectorCV>;
    RuneDetectorCV();
    void pushInput(CommonFrame& frame,bool is_big);
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
    static inline std::unique_ptr<RuneDetectorCV> make_detector() {
        return std::make_unique<RuneDetectorCV>();
    }
    DetectorCallback callback_;
    cv::Mat tmp_R_;
    int current_id_ = 0;
};