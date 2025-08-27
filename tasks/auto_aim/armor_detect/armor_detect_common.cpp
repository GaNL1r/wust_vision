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
#include "tasks/auto_aim/armor_detect/armor_detect_common.hpp"
#include "wust_vl/common/utils/timer.hpp"
#include <execution>
ArmorDetectCommon::ArmorDetectCommon(const ArmorDetectCommonParams& params) {
    params_ = params;
    number_classifier_ = std::make_unique<NumberClassifier>(
        params_.classify_model_path,
        params_.classify_label_path
    );
    corner_corrector_ = std::make_unique<LightCornerCorrector>();
}
bool ArmorDetectCommon::extractNetImage(const cv::Mat& src, armor::ArmorObject& armor) {
    const int light_length = 12;
    const int warp_height = 28;
    const int small_armor_width = 32;
    const int large_armor_width = 54;
    const cv::Size roi_size(20, 28);
    // === Step 0: 安全检查 ===
    if (src.empty() || src.data == nullptr || src.cols < 10 || src.rows < 10) {
        std::cerr << "[extractNetImage] input src is empty or too small!" << std::endl;
        return false;
    }

    // 判断装甲板类型
    bool is_large =
        (armor.number == armor::ArmorNumber::NO1 || armor.number == armor::ArmorNumber::BASE);

    // pts 数量检查
    std::vector<cv::Point2f> pts_vec(std::begin(armor.pts), std::end(armor.pts));
    if (pts_vec.size() != 4) {
        return false;
    }

    // 计算扩展 bbox
    cv::Rect bbox = cv::boundingRect(pts_vec);
    int new_width = static_cast<int>(bbox.width * params_.expand_ratio_w);
    int new_height = static_cast<int>(bbox.height * params_.expand_ratio_h);
    int new_x = std::max(0, static_cast<int>(bbox.x - (new_width - bbox.width) / 2));
    int new_y = std::max(0, static_cast<int>(bbox.y - (new_height - bbox.height) / 2));

    // 限制不越界
    new_width = std::min(new_width, src.cols - new_x);
    new_height = std::min(new_height, src.rows - new_y);

    if (new_width <= 0 || new_height <= 0)
        return false;

    cv::Rect expanded_rect(new_x, new_y, new_width, new_height);
    cv::Mat litroi_color = src(expanded_rect);
    if (litroi_color.empty())
        return false;

    cv::Mat litroi_gray;
    try {
        cv::cvtColor(litroi_color, litroi_gray, cv::COLOR_BGR2GRAY);
    } catch (...) {
        return false;
    }
    armor.whole_gray_img = litroi_gray;

    cv::Mat litroi_binary;
    try {
        cv::threshold(
            litroi_gray,
            litroi_binary,
            params_.binary_thres,
            255,
            cv::THRESH_BINARY | cv::THRESH_OTSU
        );
    } catch (...) {
        return false;
    }

    cv::Mat gray_src;
    try {
        cv::cvtColor(src, gray_src, cv::COLOR_BGR2GRAY);
    } catch (...) {
        return false;
    }

    // 光条透视变换
    cv::Point2f lights_vertices[4] = { armor.pts[0], armor.pts[1], armor.pts[2], armor.pts[3] };
    int top_light_y = (warp_height - light_length) / 2 - 1;
    int bottom_light_y = top_light_y + light_length;
    const int warp_width = is_large ? large_armor_width : small_armor_width;

    cv::Point2f target_vertices[4] = { cv::Point2f(0, bottom_light_y),
                                       cv::Point2f(0, top_light_y),
                                       cv::Point2f(warp_width - 1, top_light_y),
                                       cv::Point2f(warp_width - 1, bottom_light_y) };

    cv::Mat number_image, warp_mat;
    try {
        warp_mat = cv::getPerspectiveTransform(lights_vertices, target_vertices);
        cv::warpPerspective(gray_src, number_image, warp_mat, cv::Size(warp_width, warp_height));
        number_image =
            number_image(cv::Rect(cv::Point((warp_width - roi_size.width) / 2, 0), roi_size));
        cv::threshold(number_image, number_image, 0, 255, cv::THRESH_BINARY | cv::THRESH_OTSU);
        cv::flip(number_image, armor.number_img, 0);

    } catch (...) {
        return false;
    }

    armor.whole_binary_img = litroi_binary;
    armor.whole_rgb_img = litroi_color;
    armor.new_x = new_x;
    armor.new_y = new_y;

    return true;
}

bool ArmorDetectCommon::refineLightsFromArmorPts(armor::ArmorObject& armor) const {
    cv::Point2f armor_center = (armor.pts[0] + armor.pts[1] + armor.pts[2] + armor.pts[3]) * 0.25;

    std::vector<std::pair<int, double>> light_distances;
    for (int i = 0; i < static_cast<int>(armor.lights.size()); ++i) {
        double dist = cv::norm(armor.lights[i].center - armor_center);
        light_distances.emplace_back(i, dist);
    }

    std::sort(light_distances.begin(), light_distances.end(), [](const auto& a, const auto& b) {
        return a.second < b.second;
    });

    if (light_distances.size() >= 2) {
        const armor::Light& l1 = armor.lights[light_distances[0].first];
        const armor::Light& l2 = armor.lights[light_distances[1].first];

        if (l1.center.x < l2.center.x) {
            armor.lights[0] = l1;
            armor.lights[1] = l2;
        } else {
            armor.lights[0] = l2;
            armor.lights[1] = l1;
        }
        return true;
    } else {
        return false;
    }
}
std::vector<armor::Light> ArmorDetectCommon::findLights(
    const cv::Mat& rgb_img,
    const cv::Mat& binary_img,
    armor::ArmorObject& armor
) noexcept {
    using std::vector;
    vector<vector<cv::Point>> contours;
    vector<cv::Vec4i> hierarchy;

    cv::findContours(binary_img, contours, hierarchy, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_NONE);

    vector<armor::Light> all_lights;

    for (const auto& contour: contours) {
        if (contour.size() < 6)
            continue;

        auto light = armor::Light(contour);
        if (isLight(light)) {
            all_lights.emplace_back(light);
        }
    }

    std::sort(
        all_lights.begin(),
        all_lights.end(),
        [](const armor::Light& l1, const armor::Light& l2) { return l1.center.x < l2.center.x; }
    );

    armor.lights = all_lights;

    return all_lights;
}

bool ArmorDetectCommon::isLight(const armor::Light& light) noexcept {
    // The ratio of light (short side / long side)
    float ratio = light.width / light.length;
    bool ratio_ok =
        params_.light_params.min_ratio < ratio && ratio < params_.light_params.max_ratio;

    bool angle_ok = light.tilt_angle < params_.light_params.max_angle;

    bool is_light = ratio_ok && angle_ok;

    return is_light;
}
bool ArmorDetectCommon::isArmor(const armor::Light& light_1, const armor::Light& light_2) noexcept {
    if (light_1.length <= 1e-3f || light_2.length <= 1e-3f) {
        return false;
    }
    // Ratio of the length of 2 lights (short side / long side)
    float light_length_ratio = light_1.length < light_2.length ? light_1.length / light_2.length
                                                               : light_2.length / light_1.length;
    bool light_ratio_ok = light_length_ratio > params_.armor_params.min_light_ratio;

    // Distance between the center of 2 lights (unit : light length)
    float avg_light_length = (light_1.length + light_2.length) / 2;
    float center_distance = cv::norm(light_1.center - light_2.center) / avg_light_length;
    bool center_distance_ok = (params_.armor_params.min_small_center_distance <= center_distance
                               && center_distance < params_.armor_params.max_small_center_distance)
        || (params_.armor_params.min_large_center_distance <= center_distance
            && center_distance < params_.armor_params.max_large_center_distance);

    // Angle of light center connection
    cv::Point2f diff = light_1.center - light_2.center;
    float angle = std::abs(std::atan(diff.y / diff.x)) / CV_PI * 180;
    bool angle_ok = angle < params_.armor_params.max_angle;

    bool is_armor = light_ratio_ok && center_distance_ok && angle_ok;

    return is_armor;
}
std::vector<armor::ArmorObject> ArmorDetectCommon::detectNet(
    const cv::Mat& src_img,
    std::vector<armor::ArmorObject>& objs_result,
    int detect_color
) {
    std::vector<armor::ArmorObject> armors;
    std::mutex armors_mutex;

    std::for_each(std::execution::par, objs_result.begin(), objs_result.end(), [&](auto& armor_in) {
        armor::ArmorObject armor = armor_in;
        // 颜色过滤
        if (detect_color == 0 && armor.color == armor::ArmorColor::BLUE)
            return;
        if (detect_color == 1 && armor.color == armor::ArmorColor::RED)
            return;

        try {
            bool ok = false;
            try {
                ok = extractNetImage(src_img, armor); // 可能崩溃的函数
            } catch (const cv::Exception& e) {
                std::cerr << "[detectNet] OpenCV exception in extractNetImage: " << e.what()
                          << std::endl;
                return;
            } catch (const std::exception& e) {
                std::cerr << "[detectNet] exception in extractNetImage: " << e.what() << std::endl;
                return;
            } catch (...) {
                std::cerr << "[detectNet] unknown error in extractNetImage." << std::endl;
                return;
            }

            if (!ok)
                return;

            // 分类
            number_classifier_->classifyNumber(armor);
            if (armor.confidence < params_.classifier_threshold)
                return;
            // 颜色过滤
            if (armor.color == armor::ArmorColor::NONE || armor.color == armor::ArmorColor::PURPLE)
            {
                armor.is_ok = false;
                std::lock_guard<std::mutex> lock(armors_mutex);
                armors.emplace_back(armor);
                return;
            }

            // 灯条与角点修正
            findLights(armor.whole_rgb_img, armor.whole_binary_img, armor);
            if (refineLightsFromArmorPts(armor)) {
                if (isArmor(armor.lights[0], armor.lights[1])) {
                    corner_corrector_->correctCorners(armor);
                }
            }

            // 存入结果
            {
                std::lock_guard<std::mutex> lock(armors_mutex);
                armors.emplace_back(armor);
            }

        } catch (...) {
            std::cerr << "[ArmorDetectCommon::detectNet] something failed (unexpected)."
                      << std::endl;
        }
    });

    return armors;
}
