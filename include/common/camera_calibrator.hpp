#pragma once
#include <array>
#include <opencv2/opencv.hpp>

class CameraCalibrator {
public:
    CameraCalibrator(
        cv::Size boardSize = { 9, 6 },
        float squareSize = 25.f,
        int requiredViews = 15,
        float minShift = 20.f,
        const std::string& saveDir = "./calib_imgs"
    );

    // 传入每帧图像，内部自动采集、显示和监听按键
    // 返回 false 表示用户按 ESC 退出，需要终止主循环
    bool processFrame(const cv::Mat& frame);

    bool isCalibrated() const {
        return calibrated_;
    }
    void saveToFile(const std::string& filename) const;

private:
    cv::Size boardSize_;
    float squareSize_;
    int requiredViews_;
    float minCornerShift_;
    std::string saveDir_;
    bool calibrated_ = false;

    std::vector<std::vector<cv::Point3f>> objectPoints_;
    std::vector<std::vector<cv::Point2f>> imagePoints_;
    std::vector<cv::Point2f> lastCorners_;
    std::array<bool, 5> coverageFlags_ = { false, false, false, false, false };
    cv::Size imageSize_;

    cv::Mat cameraMatrix_, distCoeffs_;

    int savedCount_ = 0;

    std::vector<cv::Point3f> generate3DPoints() const;
    bool isDifferent(const std::vector<cv::Point2f>& c1, const std::vector<cv::Point2f>& c2) const;
    void updateCoverage(const std::vector<cv::Point2f>& corners, cv::Size imgSize);
    void calibrate(const cv::Size& imageSize);
};
