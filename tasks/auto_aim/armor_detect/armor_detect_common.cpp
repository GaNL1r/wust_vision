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
#include "tasks/utils.hpp"
#include "wust_vl/common/utils/timer.hpp"
ArmorDetectCommon::ArmorDetectCommon(const ArmorDetectCommonParams& params) {
    params_ = params;
    number_classifier_ = NumberClassifierFactory::createNumberClassifier(
        params.classify_backend,
        params.classify_model_path,
        params.classify_label_path
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
    auto ordered = armor.sortCorners(armor.pts);

    armor::Light l1, l2;
    l1.length = cv::norm(ordered[1] - ordered[0]);
    l1.center = (ordered[0] + ordered[1]) / 2.0;

    l2.length = cv::norm(ordered[2] - ordered[3]);
    l2.center = (ordered[2] + ordered[3]) / 2.0;
    float avg_light_length = (l1.length + l2.length) / 2;
    float center_distance = cv::norm(l1.center - l2.center) / avg_light_length;
    // 判断装甲板类型
    bool is_large = center_distance > params_.armor_params.min_large_center_distance;
    // (armor.number == armor::ArmorNumber::NO1 || armor.number == armor::ArmorNumber::BASE);

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
    cv::cvtColor(litroi_color, litroi_gray, cv::COLOR_BGR2GRAY);
    if (litroi_gray.empty())
        return false;
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
    cv::Point2f offset = cv::Point2f(new_x, new_y);
    // 光条透视变换
    cv::Point2f lights_vertices[4] = { armor.pts[1] - offset,
                                       armor.pts[0] - offset,
                                       armor.pts[3] - offset,
                                       armor.pts[2] - offset };
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
        cv::warpPerspective(litroi_gray, number_image, warp_mat, cv::Size(warp_width, warp_height));
        number_image =
            number_image(cv::Rect(cv::Point((warp_width - roi_size.width) / 2, 0), roi_size));
        cv::threshold(number_image, armor.number_img, 0, 255, cv::THRESH_BINARY | cv::THRESH_OTSU);
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
    armor.center = (armor.pts[0] + armor.pts[1] + armor.pts[2] + armor.pts[3]) * 0.25f;
    if (armor.lights.size() < 2)
        return false;

    auto ordered = armor.sortCorners(armor.pts);
    cv::Point2f l_centers[2] = { (ordered[0] + ordered[1]) * 0.5f,
                                 (ordered[2] + ordered[3]) * 0.5f };

    int idx[2] = { -1, -1 };
    double min_dist[2] = { std::numeric_limits<double>::max(), std::numeric_limits<double>::max() };

    for (int i = 0; i < static_cast<int>(armor.lights.size()); ++i) {
        for (int k = 0; k < 2; ++k) {
            double d = cv::norm(armor.lights[i].center - l_centers[k]);
            if (d < min_dist[k]) {
                min_dist[k] = d;
                idx[k] = i;
            }
        }
    }
    if (idx[0] == idx[1]) {
        min_dist[1] = std::numeric_limits<double>::max();
        for (int i = 0; i < static_cast<int>(armor.lights.size()); ++i) {
            if (i == idx[0])
                continue;
            double d = cv::norm(armor.lights[i].center - l_centers[1]);
            if (d < min_dist[1]) {
                min_dist[1] = d;
                idx[1] = i;
            }
        }
    }

    if (idx[0] < 0 || idx[1] < 0)
        return false;

    const auto& a = armor.lights[idx[0]];
    const auto& b = armor.lights[idx[1]];

    if (a.center.x < b.center.x) {
        armor.lights[0] = a;
        armor.lights[1] = b;
    } else {
        armor.lights[0] = b;
        armor.lights[1] = a;
    }

    return true;
}

std::vector<armor::Light> ArmorDetectCommon::findLights(
    const cv::Mat& color_img,
    const cv::Mat& binary_img,
    armor::ArmorObject& armor
) noexcept {
    std::vector<std::vector<cv::Point>> contours;
    std::vector<cv::Vec4i> hierarchy;

    cv::findContours(binary_img, contours, hierarchy, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_NONE);
    std::vector<armor::Light> all_lights;
    for (const auto& contour: contours) {
        if (contour.size() < 6)
            continue;

        auto light = armor::Light(contour);
        if (isLight(light)) {
            int sum_r = 0, sum_b = 0;
            for (const auto& point: contour) {
                const cv::Vec3b* row_ptr = color_img.ptr<cv::Vec3b>(point.y);
                const cv::Vec3b& pixel = row_ptr[point.x];
                sum_r += pixel[0];
                sum_b += pixel[2];
            }

            if (std::abs(sum_r - sum_b) / static_cast<int>(contour.size())
                > params_.light_params.color_diff_thresh)
            {
                light.color = sum_r > sum_b ? 0 : 1; // 0=红, 1=蓝
                all_lights.emplace_back(light);
            }
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
    bool color_ok = light_1.color == light_2.color;
    bool is_armor = light_ratio_ok && center_distance_ok && angle_ok;

    return is_armor;
}

std::vector<armor::ArmorObject> ArmorDetectCommon::detectNet(
    const cv::Mat& src_img,
    std::vector<armor::ArmorObject>& objs_result,
    Eigen::Matrix3f transform_matrix,
    int detect_color
) {
    std::vector<armor::ArmorObject> armors;

    if (!src_img.data || src_img.empty()) {
        std::cout << "img data nullptr or empty" << std::endl;
        return armors;
    }

    if (objs_result.empty()) {
        return armors;
    }

    auto start = time_utils::now();

    for (auto& armor_in: objs_result) {
        armor::ArmorObject armor = armor_in;

        if ((detect_color == 0 && armor.color == armor::ArmorColor::BLUE)
            || (detect_color == 1 && armor.color == armor::ArmorColor::RED))
        {
            continue;
        }

        bool ok = false;
        ok = extractNetImage(src_img, armor);

        if (!ok)
            continue;

        number_classifier_->classifyNumber(armor);

        if (armor.confidence < params_.classifier_threshold)
            continue;
        if (armor.color == armor::ArmorColor::NONE || armor.color == armor::ArmorColor::PURPLE) {
            armor.is_ok = false;
            armor.transform(transform_matrix);
            armors.emplace_back(armor);
            continue;
        }

        findLights(armor.whole_rgb_img, armor.whole_binary_img, armor);

        if (refineLightsFromArmorPts(armor)) {
            if (isArmor(armor.lights[0], armor.lights[1])) {
                armor.is_ok = true;
                corner_corrector_->correctCorners(armor, armor.whole_gray_img);
                for (auto& light: armor.lights) {
                    const cv::Point2f offset { static_cast<float>(armor.new_x),
                                               static_cast<float>(armor.new_y) };
                    light.addOffset(offset);
                }
            }
        }

        if (armor.is_ok) {
            armor.is_ok = armor.checkOkptsRight(params_.max_pts_error);
        }

        if (!armor.is_ok) {
            auto ordered = armor.sortCorners(armor.pts);

            armor::Light l1, l2;
            l1.length = cv::norm(ordered[1] - ordered[0]);
            l1.center = (ordered[0] + ordered[1]) / 2.0;

            l2.length = cv::norm(ordered[2] - ordered[3]);
            l2.center = (ordered[2] + ordered[3]) / 2.0;

            if (!isArmor(l1, l2)) {
                continue;
            }
        }

        armor.transform(transform_matrix);

        armors.emplace_back(armor);
    }

    return armors;
}
