// Copyright Chen Jun 2023. Licensed under the MIT License.
//
// Additional modifications and features by Chengfu Zou, Labor. Licensed under
// Apache License 2.0.
//
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

#include "tasks/auto_aim/armor_detect/opencv/armor_detector_opencv.hpp"
// std
#include <algorithm>
#include <cmath>
#include <execution>
#include <iostream>
#include <vector>
// OpenCV
#include <opencv2/core.hpp>
#include <opencv2/core/base.hpp>
#include <opencv2/core/mat.hpp>
#include <opencv2/core/types.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
// project
#include "wust_vl/common/utils/logger.hpp"
#include "wust_vl/common/utils/timer.hpp"
#include <fmt/core.h>

ArmorDetectOpenCV::ArmorDetectOpenCV(
    const std::string& classify_model_path,
    const std::string& classify_label_path,
    const int& bin_thres,
    const double& classifier_threshold,
    const armor::LightParams& l,
    const armor::ArmorParams& a
):
    binary_thres_(bin_thres),
    light_params_(l),
    armor_params_(a),
    classifier_threshold_(classifier_threshold) {
    corner_corrector_ = std::make_unique<LightCornerCorrector>();
    number_classifier_ =
        std::make_unique<NumberClassifier>(classify_model_path, classify_label_path);
}
void hance(
    armor::ArmorObject& armor,
    const cv::Mat& src,
    int win_width = 400,
    int win_height = 400
) {
    cv::namedWindow("roi", cv::WINDOW_NORMAL);
    cv::resizeWindow("roi", win_width, win_height);

    for (auto& light: armor.lights) {
        cv::Rect light_box = light.boundingRect();

        // 扩展 ROI
        constexpr float SCALE = 0.07f;
        float expand_w = light_box.width * SCALE;
        float expand_h = light_box.height * SCALE;

        int x1 = std::max(0, static_cast<int>(light_box.x - expand_w));
        int y1 = std::max(0, static_cast<int>(light_box.y - expand_h));
        int x2 = std::min(src.cols, static_cast<int>(light_box.x + light_box.width + expand_w));
        int y2 = std::min(src.rows, static_cast<int>(light_box.y + light_box.height + expand_h));

        cv::Rect roi_rect(x1, y1, x2 - x1, y2 - y1);
        if (roi_rect.width <= 0 || roi_rect.height <= 0)
            continue;

        cv::Mat roi = src(roi_rect);
        if (src.channels() == 3) {
            cv::cvtColor(roi, roi, cv::COLOR_BGR2GRAY);
        } else {
            cv::cvtColor(roi, roi, cv::COLOR_BayerRG2GRAY);
        }

        cv::imshow("roi", roi);
        cv::waitKey(1);
    }
}
std::vector<armor::ArmorObject>
ArmorDetectOpenCV::detect(const cv::Mat& input, int detect_color) noexcept {
    if (input.empty())
        return {};
    std::vector<armor::Light> lights_;
    cv::Mat bayer_img;
    //cv::cvtColor(input, bayer_img, cv::COLOR_)
    cv::Mat binary_img, gray_img;
    //auto t1 = time_utils::now();
    std::tie(binary_img, gray_img) = preprocessImage(input);
    // cv::imshow("bin", binary_img);
    // cv::waitKey(1);
    //auto t2 = time_utils::now();
    lights_ = findLights(input, binary_img);
    std::vector<armor::ArmorObject> armors = matchLights(lights_, detect_color);
    //auto t3 = time_utils::now();
    std::vector<armor::ArmorObject> valid_armors;
    std::mutex valid_mutex;

    std::for_each(
        std::execution::par,
        armors.begin(),
        armors.end(),
        [this, &input, gray_img, &valid_armors, &valid_mutex](armor::ArmorObject& armor) {
            try {
                armor.number_img = extractNumber(gray_img, armor);
                if (armor.number_img.empty())
                    return;

                if (!number_classifier_->classifyNumber(armor))
                    return;

                if (armor.confidence < classifier_threshold_)
                    return;

                if (corner_corrector_) {
                    corner_corrector_->correctCorners(armor, gray_img);
                }
                //hance(armor, input);
                {
                    std::lock_guard<std::mutex> lock(valid_mutex);
                    valid_armors.push_back(armor);
                }

            } catch (const std::exception& e) {
                std::cerr << "[detect] Exception: " << e.what() << std::endl;
            }
        }
    );

    return valid_armors;
}

std::tuple<cv::Mat, cv::Mat> ArmorDetectOpenCV::preprocessImage(const cv::Mat& img) noexcept {
    cv::Mat gray_img;
    cv::Mat binary_img;

    if (img.empty()) {
        return { binary_img, gray_img }; // 空图直接返回空
    }

    if (img.channels() == 3) {
        // 彩色图，RGB -> 灰度
        cv::cvtColor(img, gray_img, cv::COLOR_RGB2GRAY);
    } else if (img.channels() == 1) {
        // 单通道图，假设是 BayerRG
        cv::cvtColor(img, gray_img, cv::COLOR_BayerRG2GRAY);
    } else {
        // 非法通道数，直接返回空
        return { binary_img, gray_img };
    }

    // 二值化
    cv::threshold(gray_img, binary_img, binary_thres_, 255, cv::THRESH_BINARY);

    return { binary_img, gray_img };
}

std::vector<armor::Light>
ArmorDetectOpenCV::findLights(const cv::Mat& img, const cv::Mat& binary_img) noexcept {
    using std::vector;
    vector<vector<cv::Point>> contours;
    vector<cv::Vec4i> hierarchy;
    cv::findContours(binary_img, contours, hierarchy, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_NONE);

    vector<armor::Light> lights;
    cv::Mat color_img;
    if (img.channels() == 3) {
        color_img = img;
    } else if (img.channels() == 1) {
        cv::cvtColor(img, color_img, cv::COLOR_BayerRG2BGR);
    } else {
        return {};
    }

    for (const auto& contour: contours) {
        if (contour.size() < 6)
            continue;

        auto light = armor::Light(contour);

        if (isLight(light)) {
            // 统计 R/B 通道
            int sum_r = 0, sum_b = 0;
            for (const auto& point: contour) {
                const cv::Vec3b* row_ptr = color_img.ptr<cv::Vec3b>(point.y);
                const cv::Vec3b& pixel = row_ptr[point.x];
                sum_r += pixel[0];
                sum_b += pixel[2];
            }

            // 判断颜色
            if (std::abs(sum_r - sum_b) / static_cast<int>(contour.size())
                > light_params_.color_diff_thresh) {
                light.color = sum_r > sum_b ? 0 : 1; // 0=红, 1=蓝
            }

            lights.emplace_back(light);
        }
    }
    std::sort(lights.begin(), lights.end(), [](const armor::Light& l1, const armor::Light& l2) {
        return l1.center.x < l2.center.x;
    });
    return lights;
}
bool ArmorDetectOpenCV::isLight(const armor::Light& light) noexcept {
    // The ratio of light (short side / long side)
    float ratio = light.width / light.length;
    bool ratio_ok = light_params_.min_ratio < ratio && ratio < light_params_.max_ratio;

    bool angle_ok = light.tilt_angle < light_params_.max_angle;

    bool is_light = ratio_ok && angle_ok;

    return is_light;
}

std::vector<armor::ArmorObject>
ArmorDetectOpenCV::matchLights(const std::vector<armor::Light>& lights, int detect_color) noexcept {
    std::vector<armor::ArmorObject> armors;
    // Loop all the pairing of lights
    for (auto light_1 = lights.begin(); light_1 != lights.end(); light_1++) {
        if (light_1->color != detect_color)
            continue;
        double max_iter_width = light_1->length * armor_params_.max_large_center_distance;

        for (auto light_2 = light_1 + 1; light_2 != lights.end(); light_2++) {
            if (light_2->color != detect_color)
                continue;
            if (containLight(light_1 - lights.begin(), light_2 - lights.begin(), lights)) {
                continue;
            }
            if (light_2->center.x - light_1->center.x > max_iter_width)
                break;

            auto type = isArmor(*light_1, *light_2);
            if (type != armor::ArmorType::INVALID) {
                armor::ArmorObject armor(*light_1, *light_2);
                armor.type = type;
                if (detect_color == 0) {
                    armor.color = armor::ArmorColor::RED;
                } else {
                    armor.color = armor::ArmorColor::BLUE;
                }

                armors.emplace_back(armor);
            }
        }
    }

    return armors;
}

// Check if there is another light in the boundingRect formed by the 2 lights
bool ArmorDetectOpenCV::containLight(
    const int i,
    const int j,
    const std::vector<armor::Light>& lights
) noexcept {
    if (i < 0 || j < 0 || i >= static_cast<int>(lights.size())
        || j >= static_cast<int>(lights.size()) || i >= j)
    {
        std::cerr << "[containLight] index out of range: i=" << i << ", j=" << j
                  << ", lights.size=" << lights.size() << std::endl;
        return false;
    }

    const armor::Light &light_1 = lights[i], light_2 = lights[j];
    auto points =
        std::vector<cv::Point2f> { light_1.top, light_1.bottom, light_2.top, light_2.bottom };
    auto bounding_rect = cv::boundingRect(points);
    double avg_length = (light_1.length + light_2.length) / 2.0;
    double avg_width = (light_1.width + light_2.width) / 2.0;

    for (int k = i + 1; k < j; k++) {
        const armor::Light& test_light = lights[k];

        if (test_light.width > 2 * avg_width) {
            continue;
        }
        if (test_light.length < 0.5 * avg_length) {
            continue;
        }

        if (bounding_rect.contains(test_light.top) || bounding_rect.contains(test_light.bottom)
            || bounding_rect.contains(test_light.center))
        {
            return true;
        }
    }
    return false;
}

armor::ArmorType
ArmorDetectOpenCV::isArmor(const armor::Light& light_1, const armor::Light& light_2) noexcept {
    // Ratio of the length of 2 lights (short side / long side)
    float light_length_ratio = light_1.length < light_2.length ? light_1.length / light_2.length
                                                               : light_2.length / light_1.length;
    bool light_ratio_ok = light_length_ratio > armor_params_.min_light_ratio;

    // Distance between the center of 2 lights (unit : light length)
    float avg_light_length = (light_1.length + light_2.length) / 2;
    float center_distance = cv::norm(light_1.center - light_2.center) / avg_light_length;
    bool center_distance_ok = (armor_params_.min_small_center_distance <= center_distance
                               && center_distance < armor_params_.max_small_center_distance)
        || (armor_params_.min_large_center_distance <= center_distance
            && center_distance < armor_params_.max_large_center_distance);

    // Angle of light center connection
    cv::Point2f diff = light_1.center - light_2.center;
    float angle = std::abs(std::atan(diff.y / diff.x)) / CV_PI * 180;
    bool angle_ok = angle < armor_params_.max_angle;

    bool is_armor = light_ratio_ok && center_distance_ok && angle_ok;

    // Judge armor type
    armor::ArmorType type;
    if (is_armor) {
        type = center_distance > armor_params_.min_large_center_distance ? armor::ArmorType::LARGE
                                                                         : armor::ArmorType::SMALL;
    } else {
        type = armor::ArmorType::INVALID;
    }

    return type;
}

cv::Mat ArmorDetectOpenCV::extractNumber(const cv::Mat& src, const armor::ArmorObject& armor)
    const noexcept {
    // Light length in image
    const int light_length = 12;
    // Image size after warp
    const int warp_height = 28;
    const int small_armor_width = 32;
    const int large_armor_width = 54;
    // Number ROI size
    const cv::Size roi_size(20, 28);

    // Warp perspective transform
    cv::Point2f lights_vertices[4] = { armor.lights[0].bottom,
                                       armor.lights[0].top,
                                       armor.lights[1].top,
                                       armor.lights[1].bottom };

    const int top_light_y = (warp_height - light_length) / 2 - 1;
    const int bottom_light_y = top_light_y + light_length;
    const int warp_width =
        armor.type == armor::ArmorType::SMALL ? small_armor_width : large_armor_width;
    cv::Point2f target_vertices[4] = {
        cv::Point(0, bottom_light_y),
        cv::Point(0, top_light_y),
        cv::Point(warp_width - 1, top_light_y),
        cv::Point(warp_width - 1, bottom_light_y),
    };
    cv::Mat number_image;
    auto rotation_matrix = cv::getPerspectiveTransform(lights_vertices, target_vertices);
    cv::warpPerspective(src, number_image, rotation_matrix, cv::Size(warp_width, warp_height));

    // Get ROI
    number_image =
        number_image(cv::Rect(cv::Point((warp_width - roi_size.width) / 2, 0), roi_size));

    // Binarize
    cv::threshold(number_image, number_image, 0, 255, cv::THRESH_BINARY | cv::THRESH_OTSU);

    return number_image;
}

void ArmorDetectOpenCV::setCallback(DetectorCallback callback) {
    this->infer_callback_ = callback;
}
void ArmorDetectOpenCV::topts(armor::ArmorObject& armor) {
    if (armor.lights.size() != 2) {
        armor.is_ok = false;
        return;
    }
    armor.pts[0] = armor.lights[0].top;
    armor.pts[1] = armor.lights[0].bottom;
    armor.pts[2] = armor.lights[1].bottom;
    armor.pts[3] = armor.lights[1].top;
}
void ArmorDetectOpenCV::pushInput(CommonFrame& frame) {
    frame.id = current_id_++;
    std::vector<armor::ArmorObject> objs_result;
    objs_result = detect(frame.src_img, frame.detect_color);

    if (this->infer_callback_) {
        this->infer_callback_(objs_result, frame);
        return;
    }
    return;
}