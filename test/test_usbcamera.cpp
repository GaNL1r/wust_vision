
#include <iostream>
#include <opencv2/opencv.hpp>

int main() {
    cv::VideoCapture cap(0);
    if (!cap.isOpened())
        return -1;

    double fps = cap.get(cv::CAP_PROP_FPS);
    int delay = static_cast<int>(1000.0 / fps);

    cv::Mat frame;
    while (true) {
        cap >> frame;
        if (frame.empty()) {
            continue;
        }

        // ---- 转灰度 (亮度) ----
        cv::Mat gray;
        cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);

        // ---- 计算亮度梯度方向 & 幅度 ----
        cv::Mat gx, gy;
        cv::Sobel(gray, gx, CV_32F, 1, 0, 3);
        cv::Sobel(gray, gy, CV_32F, 0, 1, 3);

        cv::Mat magnitude;
        cv::magnitude(gx, gy, magnitude);

        // ---- 归一化显示 ----
        cv::Mat mag8;
        cv::normalize(magnitude, mag8, 0, 255, cv::NORM_MINMAX, CV_8U);

        // ---- 二值化 ----
        cv::Mat binary;
        cv::threshold(mag8, binary, 100, 255, cv::THRESH_BINARY);

        // ---- 去噪 ----
        cv::medianBlur(binary, binary, 3);

        // ---- 提取轮廓 ----
        std::vector<std::vector<cv::Point>> contours;
        std::vector<cv::Vec4i> hierarchy;
        cv::findContours(binary, contours, hierarchy, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

        // ---- 绘制轮廓 ----
        for (size_t i = 0; i < contours.size(); i++) {
            cv::drawContours(frame, contours, (int)i, cv::Scalar(0, 255, 0), 2);
        }

        // ---- 显示 ----
        cv::imshow("Gradient Magnitude", mag8);
        cv::imshow("Binary", binary);
        cv::imshow("Contours", frame);

        if (cv::waitKey(delay) == 27)
            break;
    }

    return 0;
}
