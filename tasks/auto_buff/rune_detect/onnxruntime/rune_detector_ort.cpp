// Copyright 2023 Yunlong Feng
//
// Additional modifications and features by Chengfu Zou, 2024.
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

#include "rune_detector_ort.hpp"
// std
#include <algorithm>
#include <numeric>
#include <unordered_map>
// third party
#include <opencv2/imgproc.hpp>
// project
#include "tasks/auto_buff/rune_detect/rune_infer.hpp"
#include "tasks/auto_buff/type.hpp"
static void letterbox_into_CHW_float_rgb(
    const cv::Mat& img,
    float* output_ptr,
    Eigen::Matrix3f& transform_matrix,
    int dst_width,
    int dst_height,
    bool normalize = true, // 是否做归一化
    bool use_imagenet = true, // normalize=true 时：是否使用 ImageNet mean/std
    const std::array<float, 3>& mean = { 0.485f, 0.456f, 0.406f }, // in RGB order
    const std::array<float, 3>& stdv = { 0.229f, 0.224f, 0.225f }, // in RGB order
    float pad_value = 114.0f // pad 的像素值（原始 0..255 量级）
) {
    // 校验
    CV_Assert(!img.empty());
    CV_Assert(img.type() == CV_8UC3);

    int img_h = img.rows;
    int img_w = img.cols;

    // 计算缩放与填充（与你原始实现一致）
    float scale = std::min(dst_height * 1.0f / img_h, dst_width * 1.0f / img_w);
    int resize_h = std::max(1, static_cast<int>(round(img_h * scale)));
    int resize_w = std::max(1, static_cast<int>(round(img_w * scale)));

    int pad_h = dst_height - resize_h;
    int pad_w = dst_width - resize_w;
    int top = static_cast<int>(round(pad_h / 2.0f - 0.1f));
    int left = static_cast<int>(round(pad_w / 2.0f - 0.1f));

    // 逆仿射变换矩阵（从网络坐标 -> 原图坐标）
    transform_matrix << 1.0f / scale, 0, -left / scale, 0, 1.0f / scale, -top / scale, 0, 0, 1;

    // 先 resize
    cv::Mat resized;
    cv::resize(img, resized, cv::Size(resize_w, resize_h), 0, 0, cv::INTER_LINEAR);

    const int H = dst_height;
    const int W = dst_width;
    const int HW = H * W;
    // precompute pad_value per channel according to normalize mode
    std::array<float, 3> pad_chan {};
    if (!normalize) {
        // 直接使用 0..255 的 pad value
        pad_chan = { pad_value, pad_value, pad_value }; // RGB order
    } else {
        if (use_imagenet) {
            // pad: (pad/255 - mean)/std
            for (int c = 0; c < 3; ++c) {
                float pv = pad_value / 255.0f;
                pad_chan[c] = (pv - mean[c]) / stdv[c];
            }
        } else {
            // 仅缩放到 [0,1]
            float pv = pad_value / 255.0f;
            pad_chan = { pv, pv, pv };
        }
    }

    // 1) fill whole buffer with pad values (CHW: R plane, G, B)
    // output_ptr layout: [R_plane (H*W), G_plane (H*W), B_plane (H*W)]
    for (int c = 0; c < 3; ++c) {
        float* plane_ptr = output_ptr + c * HW;
        for (int i = 0; i < HW; ++i)
            plane_ptr[i] = pad_chan[c];
    }

    // 2) copy resized image into region [top:top+resize_h, left:left+resize_w]
    //    注意：img/resized 是 BGR，且我们要输出 RGB 且放在 CHW
    for (int ry = 0; ry < resize_h; ++ry) {
        int dst_y = top + ry;
        if (dst_y < 0 || dst_y >= H)
            continue;
        const cv::Vec3b* src_row = resized.ptr<cv::Vec3b>(ry);
        // pointers to start of each plane row
        float* r_plane_row = output_ptr + 0 * HW + dst_y * W;
        float* g_plane_row = output_ptr + 1 * HW + dst_y * W;
        float* b_plane_row = output_ptr + 2 * HW + dst_y * W;

        for (int rx = 0; rx < resize_w; ++rx) {
            int dst_x = left + rx;
            if (dst_x < 0 || dst_x >= W)
                continue;

            cv::Vec3b pix = src_row[rx]; // B G R

            // get per-channel raw value
            float R = static_cast<float>(pix[2]);
            float G = static_cast<float>(pix[1]);
            float B = static_cast<float>(pix[0]);

            if (!normalize) {
                r_plane_row[dst_x] = R;
                g_plane_row[dst_x] = G;
                b_plane_row[dst_x] = B;
            } else {
                // scale to [0,1]
                float r = R / 255.0f;
                float g = G / 255.0f;
                float b = B / 255.0f;
                if (use_imagenet) {
                    r_plane_row[dst_x] = (r - mean[0]) / stdv[0]; // mean/std in RGB order
                    g_plane_row[dst_x] = (g - mean[1]) / stdv[1];
                    b_plane_row[dst_x] = (b - mean[2]) / stdv[2];
                } else {
                    r_plane_row[dst_x] = r;
                    g_plane_row[dst_x] = g;
                    b_plane_row[dst_x] = b;
                }
            }
        }
    }
}
RuneDetectorOnnxRuntime::RuneDetectorOnnxRuntime(
    std::string provider,
    std::string model_type,
    const std::filesystem::path& model_path,
    float conf_threshold,
    int top_k,
    float nms_threshold
):
    model_path_(model_path),
    conf_threshold_(conf_threshold),
    top_k_(top_k),
    nms_threshold_(nms_threshold) {
    auto model = rune_infer::modeFromString(model_type);
    rune_infer_ =
        std::make_unique<rune_infer::RuneInfer>(model, conf_threshold, nms_threshold, top_k);
    provider_ = string2OrtProvider(provider);
    init();
}

void RuneDetectorOnnxRuntime::init() {
    onnxruntime_net_ = std::make_unique<ml_net::OnnxRuntimeNet>();
    ml_net::OnnxRuntimeNet::Params params;
    params.model_path = model_path_;
    params.provider = provider_;
    onnxruntime_net_->init(params);

    strides_ = { 8, 16, 32 };
    rune_infer_->generateGridsAndStride(
        rune_infer_->getInputW(),
        rune_infer_->getInputH(),
        strides_,
        grid_strides_
    );
}

void RuneDetectorOnnxRuntime::pushInput(CommonFrame& frame) {
    frame.id = current_id_++;
    processCallback(frame);
}

void RuneDetectorOnnxRuntime::setCallback(CallbackType callback) {
    infer_callback_ = callback;
}

bool RuneDetectorOnnxRuntime::processCallback(const CommonFrame& frame) {
    Eigen::Matrix3f transform_matrix;

    std::vector<float> input_tensor_values(rune_infer_->getInputW() * rune_infer_->getInputH() * 3);
    letterbox_into_CHW_float_rgb(
        frame.src_img,
        input_tensor_values.data(),
        transform_matrix,
        rune_infer_->getInputW(),
        rune_infer_->getInputH(),
        rune_infer_->getUseNorm()
    );

    auto output_data = onnxruntime_net_->infer(input_tensor_values);

    auto output_shape = onnxruntime_net_->getOutputShape();
    int rows = static_cast<int>(output_shape[1]);
    int cols = static_cast<int>(output_shape[2]);
    cv::Mat output_buffer(rows, cols, CV_32F, output_data);
    // Parsed variable
    auto objs_result = rune_infer_->postProcess(output_buffer, transform_matrix, grid_strides_);

    // objs_result.erase(
    //     std::remove_if(
    //         objs_result.begin(),
    //         objs_result.end(),
    //         [c = static_cast<EnemyColor>(gobal::detect_color)](const auto& objs_result) {
    //             return objs_result.color != c;
    //         }
    //     ),
    //     objs_result.end()
    // );

    // Call callback function
    if (this->infer_callback_) {
        this->infer_callback_(objs_result, frame);
        return true;
    }

    return false;
}

std::tuple<cv::Point2f, cv::Mat> RuneDetectorOnnxRuntime::detectRTag(
    const cv::Mat& img,
    int binary_thresh,
    const cv::Point2f& prior,
    bool precise
) {
    if (!img.data || img.cols <= 0 || img.rows <= 0) {
        std::cerr << "[detectRTag] Invalid input image." << std::endl;
        return {};
    }

    if (prior.x < 0 || prior.x >= img.cols || prior.y < 0 || prior.y >= img.rows) {
        std::cerr << "[detectRTag] Prior out of bounds: " << prior
                  << " for image size: " << img.cols << "x" << img.rows << std::endl;
        return { prior, cv::Mat::zeros(cv::Size(200, 200), CV_8UC3) };
    }
    int px = static_cast<int>(std::floor(prior.x));
    int py = static_cast<int>(std::floor(prior.y));
    if (px < 0 || px >= img.cols || py < 0 || py >= img.rows) {
        std::cerr << "[detectRTag] Prior out of bounds: " << prior
                  << " for image size: " << img.cols << "x" << img.rows << std::endl;
        return { prior, cv::Mat::zeros(cv::Size(200, 200), CV_8UC3) };
    }

    // ROI calculation
    cv::Rect roi;
    if (precise) {
        roi = cv::Rect(prior.x - 30, prior.y - 30, 60, 60) & cv::Rect(0, 0, img.cols, img.rows);
    } else {
        roi = cv::Rect(prior.x - 100, prior.y - 100, 200, 200) & cv::Rect(0, 0, img.cols, img.rows);
    }

    if (roi.width == 0 || roi.height == 0) {
        std::cerr << "[detectRTag] ROI is zero-sized: " << roi << std::endl;
        return { prior, cv::Mat::zeros(200, 200, CV_8UC3) };
    }

    // Create ROI

    const cv::Point2f prior_in_roi = prior - cv::Point2f(roi.tl());

    cv::Mat img_roi = img(roi);

    // Gray -> Binary -> Dilate
    cv::Mat gray_img;
    cv::cvtColor(img_roi, gray_img, cv::COLOR_BGR2GRAY);
    cv::Mat binary_img;
    cv::threshold(gray_img, binary_img, 0, 255, cv::THRESH_BINARY | cv::THRESH_OTSU);
    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3));
    cv::dilate(binary_img, binary_img, kernel);

    // Find contours
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(binary_img, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_NONE);

    auto it = std::find_if(
        contours.begin(),
        contours.end(),
        [p = prior_in_roi](const std::vector<cv::Point>& contour) -> bool {
            return cv::boundingRect(contour).contains(p);
        }
    );

    // For visualization
    cv::cvtColor(binary_img, binary_img, cv::COLOR_GRAY2BGR);

    if (it == contours.end()) {
        return { prior, binary_img };
    }

    cv::drawContours(binary_img, contours, it - contours.begin(), cv::Scalar(0, 255, 0), 2);

    cv::Point2f center = std::accumulate(it->begin(), it->end(), cv::Point(0, 0));
    center /= static_cast<float>(it->size());
    center += cv::Point2f(roi.tl());

    return { center, binary_img };
}
