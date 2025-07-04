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

struct ImageFrame {
    std::vector<uint8_t> data;
    int width;
    int height;
    int step;
    std::chrono::steady_clock::time_point timestamp;
};

inline cv::Mat convertToMat(const ImageFrame& frame) {
    if (frame.data.empty()) {
        return cv::Mat();
    }
    cv::Mat rgb(frame.height, frame.width, CV_8UC3);
    memcpy(rgb.data, frame.data.data(), frame.height * frame.step);
    return rgb;
}
inline cv::Mat convertToMatrgb(const ImageFrame& frame) {
    if (frame.data.empty()) {
        return cv::Mat();
    }
    cv::Mat img(frame.height, frame.width, CV_8UC3);
    memcpy(img.data, frame.data.data(), frame.height * frame.step);
    return img;
}
inline cv::Mat convertToMatbgr(const ImageFrame& frame) {
    if (frame.data.empty()) {
        return cv::Mat();
    }
    cv::Mat img(frame.height, frame.width, CV_8UC3);
    memcpy(img.data, frame.data.data(), frame.height * frame.step);

    cv::cvtColor(img, img, cv::COLOR_RGB2BGR);
    return img;
}
