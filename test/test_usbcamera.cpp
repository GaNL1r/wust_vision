#include <iostream>
#include <opencv2/opencv.hpp>

#include <chrono>
#include <iostream>
#include <opencv2/opencv.hpp>

// int main() {
//     // 设置 JPEG 压缩质量

//     cv::VideoCapture cap("/dev/video2", cv::CAP_V4L2);

//     if (!cap.isOpened()) {
//         std::cerr << "❌ 无法打开摄像头！" << std::endl;
//         return -1;
//     }

//     cap.set(cv::CAP_PROP_FOURCC, cv::VideoWriter::fourcc('M', 'J', 'P', 'G'));
//     cap.set(cv::CAP_PROP_FRAME_WIDTH, 1280);
//     cap.set(cv::CAP_PROP_FRAME_HEIGHT, 720);
//     cap.set(cv::CAP_PROP_FPS, 60);

//     cap.set(cv::CAP_PROP_AUTO_EXPOSURE, 1);
//     cap.set(cv::CAP_PROP_EXPOSURE, 1);
//     cv::Mat frame;
//     auto last_time = std::chrono::steady_clock::now();
//     int frame_count = 0;

//     while (true) {
//         cap >> frame;
//         if (frame.empty())
//             break;

//         frame_count++;
//         auto now = std::chrono::steady_clock::now();
//         double elapsed = std::chrono::duration<double>(now - last_time).count();
//         if (elapsed >= 1.0) {
//             std::cout << "FPS: " << frame_count << std::endl;
//             frame_count = 0;
//             last_time = now;
//         }
//         cv::imshow("USB Camera", frame);
//         cv::waitKey(1);
//     }

//     cap.release();
//     return 0;
// }
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
