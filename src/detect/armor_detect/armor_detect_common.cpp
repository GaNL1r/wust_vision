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
#include "detect/armor_detect/armor_detect_common.hpp"
#include "common/gobal.hpp"

ArmorDetectCommon::ArmorDetectCommon(
    const std::string& classify_model_path,
    const std::string& classify_label_path,
    const LightParams& l,
    const ArmorParams& a,
    double classifier_threshold,
    float expand_ratio_w,
    float expand_ratio_h,
    int binary_thres_
):
    light_params_(l),
    armor_params_(a),
    binary_thres_(binary_thres_),
    classifier_threshold(classifier_threshold),
    expand_ratio_w_(expand_ratio_w),
    expand_ratio_h_(expand_ratio_h) {
    number_classifier_ =
        std::make_unique<NumberClassifier>(classify_model_path, classify_label_path);
    corner_corrector = std::make_unique<LightCornerCorrector>();
}
bool ArmorDetectCommon::extractNetImage(const cv::Mat& src, ArmorObject& armor) {
    const int light_length = 12;
    const int warp_height = 28;
    const int small_armor_width = 32;
    const int large_armor_width = 54;
    const cv::Size roi_size(20, 28);
    CV_Assert(!src.empty());
    CV_Assert(src.data != nullptr);

    // === Step 1: 初始检查 ===
    if (src.empty() || src.cols < 10 || src.rows < 10) {
        std::cerr << "[extractImage] Source image is empty or too small!" << std::endl;
        return false;
    }

    // 判断装甲板类型
    bool is_large = (armor.number == ArmorNumber::NO1 || armor.number == ArmorNumber::BASE);

    // pts 数量检查
    std::vector<cv::Point2f> pts_vec(std::begin(armor.pts), std::end(armor.pts));
    if (pts_vec.size() != 4) {
        std::cerr << "[extractImage] Armor points must be 4!" << std::endl;
        return false;
    }

    // Step 2: 计算并限制扩展的 bbox
    cv::Rect bbox = cv::boundingRect(pts_vec);

    int new_width = static_cast<int>(bbox.width * expand_ratio_w_);
    int new_height = static_cast<int>(bbox.height * expand_ratio_h_);
    int new_x = static_cast<int>(bbox.x - (new_width - bbox.width) / 2);
    int new_y = static_cast<int>(bbox.y - (new_height - bbox.height) / 2);

    // 保证不越界
    new_x = std::max(0, new_x);
    new_y = std::max(0, new_y);
    if (new_x + new_width > src.cols)
        new_width = src.cols - new_x;
    if (new_y + new_height > src.rows)
        new_height = src.rows - new_y;

    if (new_width <= 0 || new_height <= 0) {
        std::cerr << "[extractImage] Invalid expanded ROI size!" << std::endl;
        return false;
    }

    armor.new_x = new_x;
    armor.new_y = new_y;

    // Step 3: 获取 ROI 并转换灰度、二值图
    cv::Rect expanded_rect(new_x, new_y, new_width, new_height);
    if (expanded_rect.x < 0 || expanded_rect.y < 0
        || expanded_rect.x + expanded_rect.width > src.cols
        || expanded_rect.y + expanded_rect.height > src.rows || expanded_rect.width <= 0
        || expanded_rect.height <= 0)
    {
        std::cerr << "[extractImage] Expanded rect invalid: " << expanded_rect << std::endl;
        return false;
    }

    const int MIN_SIZE = 10;
    if (expanded_rect.width < MIN_SIZE || expanded_rect.height < MIN_SIZE) {
        std::cerr << "[extractImage] expanded_rect too small: " << expanded_rect << std::endl;
        return false;
    }

    cv::Mat litroi_color = src(expanded_rect).clone();
    if (litroi_color.empty())
        return false;

    cv::Mat litroi_gray;
    try {
        cv::cvtColor(litroi_color, litroi_gray, cv::COLOR_RGB2GRAY);
    } catch (const cv::Exception& e) {
        std::cerr << "[extractImage] cvtColor failed: " << e.what() << std::endl;
        return false;
    }

    armor.whole_gray_img = litroi_gray;

    cv::Mat litroi_binary;
    try {
        cv::threshold(
            litroi_gray,
            litroi_binary,
            binary_thres_,
            255,
            cv::THRESH_BINARY | cv::THRESH_OTSU
        );
    } catch (...) {
        std::cerr << "[extractImage] Thresholding failed." << std::endl;
        return false;
    }

    // Step 4: 装甲板透视变换
    cv::Point2f lights_vertices[4] = { armor.pts[0], armor.pts[1], armor.pts[2], armor.pts[3] };

    const int top_light_y = (warp_height - light_length) / 2 - 1;
    const int bottom_light_y = top_light_y + light_length;
    const int warp_width = is_large ? large_armor_width : small_armor_width;

    cv::Point2f target_vertices[4] = {
        cv::Point(0, bottom_light_y),
        cv::Point(0, top_light_y),
        cv::Point(warp_width - 1, top_light_y),
        cv::Point(warp_width - 1, bottom_light_y),
    };

    cv::Mat number_image, warp_mat;
    try {
        warp_mat = cv::getPerspectiveTransform(lights_vertices, target_vertices);
        cv::warpPerspective(src, number_image, warp_mat, cv::Size(warp_width, warp_height));
    } catch (const cv::Exception& e) {
        std::cerr << "[extractImage] warpPerspective failed: " << e.what() << std::endl;
        return false;
    }

    // Step 5: 获取中心 ROI
    if (number_image.empty() || number_image.cols < roi_size.width
        || number_image.rows < roi_size.height)
    {
        std::cerr << "[extractImage] number_image is too small!" << std::endl;
        return false;
    }

    cv::Rect center_roi((warp_width - roi_size.width) / 2, 0, roi_size.width, roi_size.height);
    number_image = number_image(center_roi);

    // 灰度+二值化
    try {
        cv::cvtColor(number_image, number_image, cv::COLOR_RGB2GRAY);
        cv::threshold(number_image, number_image, 0, 255, cv::THRESH_BINARY | cv::THRESH_OTSU);
    } catch (...) {
        std::cerr << "[extractImage] post-warp threshold or color conversion failed." << std::endl;
        return false;
    }

    // 翻转
    cv::Mat flipped_image;
    try {
        cv::flip(number_image, flipped_image, 0);
    } catch (...) {
        std::cerr << "[extractImage] flip failed." << std::endl;
        return false;
    }

    // Step 6: 赋值并返回
    armor.number_img = flipped_image;
    armor.whole_binary_img = litroi_binary;
    armor.whole_rgb_img = litroi_color;

    return true;
}

bool ArmorDetectCommon::refineLightsFromArmorPts(ArmorObject& armor) const {
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
        const Light& l1 = armor.lights[light_distances[0].first];
        const Light& l2 = armor.lights[light_distances[1].first];

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
std::vector<Light> ArmorDetectCommon::findLights(
    const cv::Mat& rgb_img,
    const cv::Mat& binary_img,
    ArmorObject& armor
) noexcept {
    using std::vector;
    vector<vector<cv::Point>> contours;
    vector<cv::Vec4i> hierarchy;

    cv::findContours(binary_img, contours, hierarchy, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_NONE);

    vector<Light> all_lights;

    for (const auto& contour: contours) {
        if (contour.size() < 6)
            continue;

        auto light = Light(contour);
        if (isLight(light)) {
            all_lights.emplace_back(light);
        }
    }

    std::sort(all_lights.begin(), all_lights.end(), [](const Light& l1, const Light& l2) {
        return l1.center.x < l2.center.x;
    });

    armor.lights = all_lights;

    return all_lights;
}

bool ArmorDetectCommon::isLight(const Light& light) noexcept {
    // The ratio of light (short side / long side)
    float ratio = light.width / light.length;
    bool ratio_ok = light_params_.min_ratio < ratio && ratio < light_params_.max_ratio;

    bool angle_ok = light.tilt_angle < light_params_.max_angle;

    bool is_light = ratio_ok && angle_ok;

    return is_light;
}
bool ArmorDetectCommon::isArmor(const Light& light_1, const Light& light_2) noexcept {
    if (light_1.length <= 1e-3f || light_2.length <= 1e-3f) {
        return false;
    }
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

    return is_armor;
}
std::vector<ArmorObject>
ArmorDetectCommon::detectNet(const cv::Mat& src_img, std::vector<ArmorObject>& objs_result) {
    std::vector<ArmorObject> armors;

    for (auto& armor: objs_result) {
        // if (armor.color == ArmorColor::NONE || armor.color == ArmorColor::PURPLE) {
        //     continue;
        // }
        if (gobal::detect_color_ == 0 && armor.color == ArmorColor::BLUE) {
            continue;
        } else if (gobal::detect_color_ == 1 && armor.color == ArmorColor::RED) {
            continue;
        }

        try {
            if (extractNetImage(src_img, armor)) {
                number_classifier_->classifyNumber(armor);
                if (armor.confidence < classifier_threshold) {
                    continue;
                }

                findLights(armor.whole_rgb_img, armor.whole_binary_img, armor);
                if (refineLightsFromArmorPts(armor)) {
                    if (isArmor(armor.lights[0], armor.lights[1])) {
                        corner_corrector->correctCorners(armor);
                    }
                }
                armors.emplace_back(armor);
            }
        } catch (...) {
            std::cout << "[ArmorDetectCommon::detectNet] extractNetImage failed." << std::endl;
        }
    }
    return armors;
}