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

#include "tasks/auto_aim/armor_detect/light_corner_corrector.hpp"
namespace wust_vision {
namespace auto_aim {
    struct LightCornerCorrector::Impl {
    public:
        struct SymmetryAxis {
            cv::Point2f centroid;
            cv::Point2f direction;
            float mean_val; // Mean brightness
        };
        Impl() noexcept {}

        void correctCorners(ArmorObject& armor, const cv::Mat& gray_img) const noexcept {
            constexpr int PASS_OPTIMIZE_WIDTH = 3;
            if (armor.lights.empty() || gray_img.empty())
                return;

            for (auto& light: armor.lights) {
                if (light.width <= PASS_OPTIMIZE_WIDTH) {
                    continue;
                }
                SymmetryAxis axis = findSymmetryAxis(gray_img, light);
                light.center = axis.centroid;
                light.axis = axis.direction;

                if (auto t = findCorner(gray_img, light, axis, true); t.x > 0)
                    light.top = t;
                if (auto b = findCorner(gray_img, light, axis, false); b.x > 0)
                    light.bottom = b;
            }
        }

        SymmetryAxis findSymmetryAxis(const cv::Mat& gray_img, const Light& light) const noexcept {
            constexpr float MAX_BRIGHTNESS = 25.0f;
            constexpr float SCALE = 0.07f;

            cv::Rect light_box = light.boundingRect();

            const float expand_w = light_box.width * SCALE;
            const float expand_h = light_box.height * SCALE;

            light_box.x -= static_cast<int>(expand_w);
            light_box.y -= static_cast<int>(expand_h);
            light_box.width += static_cast<int>(2 * expand_w);
            light_box.height += static_cast<int>(2 * expand_h);

            light_box &= cv::Rect(0, 0, gray_img.cols, gray_img.rows);
            if (light_box.width <= 0 || light_box.height <= 0)
                return {};

            const cv::Mat roi = gray_img(light_box);
            cv::Mat roi_float;
            roi.convertTo(roi_float, CV_32F);

            double min_val, max_val;
            cv::minMaxLoc(roi_float, &min_val, &max_val);

            const float range = std::max(static_cast<float>(max_val - min_val), 1e-5f);
            const float scale = MAX_BRIGHTNESS / range;

            roi_float = (roi_float - static_cast<float>(min_val)) * scale;

            const cv::Moments m = cv::moments(roi_float, false);
            const float m00 = std::max(static_cast<float>(m.m00), 1e-5f);

            const cv::Point2f centroid(
                static_cast<float>(m.m10 / m00) + light_box.x,
                static_cast<float>(m.m01 / m00) + light_box.y
            );

            std::vector<cv::Point2f> points;
            points.reserve(roi_float.total() / 2);

            for (int y = 0; y < roi_float.rows; ++y) {
                const float* row_ptr = roi_float.ptr<float>(y);
                for (int x = 0; x < roi_float.cols; ++x) {
                    if (row_ptr[x] > 1.0f) {
                        points.emplace_back(static_cast<float>(x), static_cast<float>(y));
                    }
                }
            }

            if (points.empty())
                return {};
            cv::Mat data(static_cast<int>(points.size()), 2, CV_32F);
            for (size_t i = 0; i < points.size(); ++i) {
                data.at<float>(static_cast<int>(i), 0) = points[i].x;
                data.at<float>(static_cast<int>(i), 1) = points[i].y;
            }

            const cv::PCA pca(data, cv::Mat(), cv::PCA::DATA_AS_ROW);

            cv::Point2f axis(pca.eigenvectors.at<float>(0, 0), pca.eigenvectors.at<float>(0, 1));

            const float norm = std::max(static_cast<float>(cv::norm(axis)), 1e-5f);
            axis /= norm;
            if (axis.y > 0)
                axis = -axis;

            const float mean_val = static_cast<float>(cv::mean(roi_float)[0]);

            return SymmetryAxis { .centroid = centroid, .direction = axis, .mean_val = mean_val };
        }

        cv::Point2f findCorner(
            const cv::Mat& gray_img,
            const Light& light,
            const SymmetryAxis& axis,
            bool is_top
        ) const noexcept {
            constexpr float START = 0.8f / 2;
            constexpr float END = 1.2f / 2;

            const int cols = gray_img.cols;
            const int rows = gray_img.rows;

            const int oper = is_top ? 1 : -1;

            const float L = light.length;
            const float dx = axis.direction.x * oper;
            const float dy = axis.direction.y * oper;

            const float t_start = L * START;
            const float t_range = L * (END - START);

            const float cx = axis.centroid.x;
            const float cy = axis.centroid.y;

            const int n = std::max(1, static_cast<int>(std::round(light.width - 2)));
            int half_n = n >> 1;

            std::vector<cv::Point2f> candidates;
            candidates.reserve(n);

            for (int i = -half_n; i <= half_n; ++i) {
                const float x0 = cx + t_start * dx + static_cast<float>(i);
                const float y0 = cy + t_start * dy;

                float max_diff = 0.0f;
                bool found = false;

                cv::Point2f best_corner(x0, y0);

                float prev_val = 0.0f;
                bool has_prev = false;

                for (float t = 0.0f; t < t_range; t += 1.0f) {
                    const int x = static_cast<int>(x0 + dx * t);
                    const int y = static_cast<int>(y0 + dy * t);

                    if (x < 0 || x >= cols || y < 0 || y >= rows)
                        break;

                    const uchar* row_ptr = gray_img.ptr<uchar>(y);
                    const float cur_val = static_cast<float>(row_ptr[x]);

                    if (has_prev) {
                        const float diff = prev_val - cur_val;
                        if (diff > max_diff && prev_val > axis.mean_val) {
                            max_diff = diff;
                            best_corner =
                                cv::Point2f(static_cast<float>(x) - dx, static_cast<float>(y) - dy);
                            found = true;
                        }
                    }

                    prev_val = cur_val;
                    has_prev = true;
                }

                if (found)
                    candidates.emplace_back(best_corner);
            }

            if (!candidates.empty()) {
                cv::Point2f sum(0.f, 0.f);
                for (const auto& pt: candidates)
                    sum += pt;
                return sum * (1.0f / static_cast<float>(candidates.size()));
            }

            return cv::Point2f(-1.f, -1.f);
        }
    };
    LightCornerCorrector::LightCornerCorrector() noexcept {
        _impl = std::make_unique<Impl>();
    }
    LightCornerCorrector::~LightCornerCorrector() {
        _impl.reset();
    }
    void LightCornerCorrector::correctCorners(ArmorObject& armor, const cv::Mat& gray_img)
        const noexcept {
        _impl->correctCorners(armor, gray_img);
    }
} // namespace auto_aim
} // namespace wust_vision