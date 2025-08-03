// Copyright 2025 XiaoJian Wu
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
#pragma once

#include "opencv2/opencv.hpp"
#include <Eigen/Dense>

struct ImageFrame {
    std::vector<uint8_t> data;
    int width;
    int height;
    int step;
    std::chrono::steady_clock::time_point timestamp;
    Eigen::Matrix3d R_gimbal2odom;
    Eigen::Vector3d v;
};

inline cv::Mat convertToMatrgb(const ImageFrame& frame) {
    if (frame.data.empty()) {
        return cv::Mat();
    }
    // 直接用frame.data指针创建Mat，零拷贝
    // 注意这里用const_cast是因为cv::Mat构造需要非const指针
    return cv::Mat(
               frame.height,
               frame.width,
               CV_8UC3,
               const_cast<uint8_t*>(frame.data.data()),
               frame.step
    )
        .clone();
}

// 对于BGR版本，仍需颜色转换，但可以先用零拷贝包装：
inline cv::Mat convertToMatbgr(const ImageFrame& frame) {
    if (frame.data.empty()) {
        return cv::Mat();
    }
    // 零拷贝构造RGB Mat视图
    cv::Mat rgb = cv::Mat(
        frame.height,
        frame.width,
        CV_8UC3,
        const_cast<uint8_t*>(frame.data.data()),
        frame.step
    );
    cv::Mat bgr;
    cv::cvtColor(rgb, bgr, cv::COLOR_RGB2BGR);
    return bgr; // 返回转换后新内存，无法避免拷贝
}
