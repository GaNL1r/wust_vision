// Maintained by Shenglin Qin, Chengfu Zou
// Copyright (C) FYT Vision Group. All rights reserved.
// Copyright 2025 Xiaojian Wu
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

#include "detect/armor_detect/light_corner_corrector.hpp"
#include "wust_vl/common/utils/logger.hpp"

#include <cmath>
#include <iostream>
#include <numeric>
#include <ostream>
void LightCornerCorrector::correctCorners(armor::ArmorObject& armor) noexcept {
    constexpr int PASS_OPTIMIZE_WIDTH = 3;
    if (armor.lights.empty() || armor.whole_gray_img.empty())
        return;

    const cv::Point2f offset { static_cast<float>(armor.new_x), static_cast<float>(armor.new_y) };

    auto translate = [&](cv::Point2f& p) { p += offset; };

    for (auto& light: armor.lights) {
        if (light.width <= PASS_OPTIMIZE_WIDTH) {
            translate(light.top);
            translate(light.center);
            translate(light.bottom);
            continue;
        }

        SymmetryAxis axis = findSymmetryAxis(armor.whole_gray_img, light);
        light.center = axis.centroid;
        light.axis = axis.direction;

        if (auto t = findCorner(armor.whole_gray_img, light, axis, "top"); t.x > 0)
            light.top = t;
        if (auto b = findCorner(armor.whole_gray_img, light, axis, "bottom"); b.x > 0)
            light.bottom = b;

        translate(light.top);
        translate(light.center);
        translate(light.bottom);
    }

    armor.pts_binary = { armor.lights[0].top,
                         armor.lights[0].bottom,
                         armor.lights[1].bottom,
                         armor.lights[1].top };
    armor.is_ok =
        std::all_of(armor.pts_binary.begin(), armor.pts_binary.end(), [](const cv::Point2f& p) {
            return p.x >= 0 && p.y >= 0;
        });

    if (armor.is_ok) {
        armor.center = (armor.lights[0].center + armor.lights[1].center) * 0.5f;
    } else {
        armor.center = std::accumulate(
                           armor.pts.begin(),
                           armor.pts.end(),
                           cv::Point2f { 0, 0 },
                           [](const cv::Point2f& a, const cv::Point2f& b) { return a + b; }
                       )
            * 0.25f;
    }
}

void LightCornerCorrector::correctCorners_nonmatch(
    armor::ArmorObject& armor,
    const cv::Mat& gray_img
) noexcept {
    // If the width of the light is too small, the correction is not performed
    constexpr int PASS_OPTIMIZE_WIDTH = 3;
    if (armor.lights.empty())
        return;
    if (gray_img.empty()) {
        return;
    }

    if (armor.lights[0].width > PASS_OPTIMIZE_WIDTH) {
        // Find the symmetry axis of the light
        SymmetryAxis left_axis = findSymmetryAxis(gray_img, armor.lights[0]);
        armor.lights[0].center = left_axis.centroid;
        armor.lights[0].axis = left_axis.direction;
        // Find the corner of the light
        if (cv::Point2f t = findCorner(gray_img, armor.lights[0], left_axis, "top"); t.x > 0) {
            armor.lights[0].top = t;
        }
        if (cv::Point2f b = findCorner(gray_img, armor.lights[0], left_axis, "bottom"); b.x > 0) {
            armor.lights[0].bottom = b;
        }
    }

    if (armor.lights[1].width > PASS_OPTIMIZE_WIDTH) {
        // Find the symmetry axis of the light
        SymmetryAxis right_axis = findSymmetryAxis(gray_img, armor.lights[1]);
        armor.lights[1].center = right_axis.centroid;
        armor.lights[1].axis = right_axis.direction;
        // Find the corner of the light
        if (cv::Point2f t = findCorner(gray_img, armor.lights[1], right_axis, "top"); t.x > 0) {
            armor.lights[1].top = t;
        }
        if (cv::Point2f b = findCorner(gray_img, armor.lights[1], right_axis, "bottom"); b.x > 0) {
            armor.lights[1].bottom = b;
        }
    }
    armor.is_ok = true;
}

SymmetryAxis
LightCornerCorrector::findSymmetryAxis(const cv::Mat& gray_img, const armor::Light& light) {
    constexpr float MAX_BRIGHTNESS = 25.0f;
    constexpr float SCALE = 0.07f;

    cv::Rect light_box = light.boundingRect();
    float expand_w = light_box.width * SCALE;
    float expand_h = light_box.height * SCALE;

    light_box.x = static_cast<int>(light_box.x - expand_w);
    light_box.y = static_cast<int>(light_box.y - expand_h);
    light_box.width = static_cast<int>(light_box.width + 2 * expand_w);
    light_box.height = static_cast<int>(light_box.height + 2 * expand_h);

    light_box &= cv::Rect(0, 0, gray_img.cols, gray_img.rows);
    if (light_box.width <= 0 || light_box.height <= 0)
        return {};

    cv::Mat roi = gray_img(light_box);
    cv::Mat roi_float;
    roi.convertTo(roi_float, CV_32F);

    double min_val, max_val;
    cv::minMaxLoc(roi_float, &min_val, &max_val);
    double range = std::max(max_val - min_val, 1e-5);
    cv::normalize(roi_float - min_val, roi_float, 0.0, MAX_BRIGHTNESS, cv::NORM_MINMAX);

    cv::Moments m = cv::moments(roi_float, false);
    float m00 = std::max(static_cast<float>(m.m00), 1e-5f);
    cv::Point2f centroid(
        static_cast<float>(m.m10 / m00) + light_box.x,
        static_cast<float>(m.m01 / m00) + light_box.y
    );
    std::vector<cv::Point2f> points;
    for (int y = 0; y < roi_float.rows; ++y) {
        const float* row_ptr = roi_float.ptr<float>(y);
        for (int x = 0; x < roi_float.cols; ++x) {
            float val = row_ptr[x];
            if (val > 1.0f) {
                points.emplace_back(static_cast<float>(x), static_cast<float>(y));
            }
        }
    }

    if (points.empty())
        return {};

    cv::Mat data(points);
    data = data.reshape(1);
    cv::PCA pca(data, cv::Mat(), cv::PCA::DATA_AS_ROW);

    cv::Point2f axis(pca.eigenvectors.at<float>(0, 0), pca.eigenvectors.at<float>(0, 1));
    axis /= std::max(static_cast<float>(cv::norm(axis)), 1e-5f);

    if (axis.y > 0)
        axis = -axis;

    float mean_val = static_cast<float>(cv::mean(roi_float)[0]);

    return SymmetryAxis { .centroid = centroid, .direction = axis, .mean_val = mean_val };
}

cv::Point2f LightCornerCorrector::findCorner(
    const cv::Mat& gray_img,
    const armor::Light& light,
    const SymmetryAxis& axis,
    std::string order
) {
    constexpr float START = 0.8f / 2;
    constexpr float END = 1.2f / 2;

    auto inImage = [&gray_img](int x, int y) -> bool {
        return (x >= 0 && x < gray_img.cols && y >= 0 && y < gray_img.rows);
    };

    int oper = (order == "top") ? 1 : -1;
    float L = light.length;
    float dx = axis.direction.x * oper;
    float dy = axis.direction.y * oper;

    std::vector<cv::Point2f> candidates;
    int n = std::max(1, static_cast<int>(std::round(light.width - 2)));
    int half_n = n / 2;

    for (int i = -half_n; i <= half_n; ++i) {
        float x0 = axis.centroid.x + L * START * dx + i;
        float y0 = axis.centroid.y + L * START * dy;

        float max_diff = 0;
        bool found = false;
        cv::Point2f corner(x0, y0);
        float prev_val = 0;

        for (float t = 0.0f; t < L * (END - START); t += 1.0f) {
            int x = static_cast<int>(x0 + dx * t);
            int y = static_cast<int>(y0 + dy * t);
            if (!inImage(x, y))
                break;

            float cur_val = gray_img.at<uchar>(y, x);

            if (t > 0) {
                float diff = prev_val - cur_val;
                if (diff > max_diff && prev_val > axis.mean_val) {
                    max_diff = diff;
                    corner = cv::Point2f(static_cast<float>(x - dx), static_cast<float>(y - dy));
                    found = true;
                }
            }
            prev_val = cur_val;
        }

        if (found)
            candidates.emplace_back(corner);
    }

    if (!candidates.empty()) {
        cv::Point2f sum(0, 0);
        for (const auto& pt: candidates)
            sum += pt;
        return sum * (1.0f / candidates.size());
    } else {
        return cv::Point2f(-1, -1);
    }
}
