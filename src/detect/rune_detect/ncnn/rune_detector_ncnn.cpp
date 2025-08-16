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

#include "detect/rune_detect/ncnn/rune_detector_ncnn.hpp"
// std
#include <algorithm>
#include <numeric>
#include <unordered_map>
// third party
#include <opencv2/imgproc.hpp>
// project
#include "common/gobal.hpp"
#include "detect/rune_detect/rune_infer.hpp"
#include "type/type.hpp"

static ncnn::Mat letterbox_to_ncnn(
    const cv::Mat& img,
    Eigen::Matrix3f& transform_matrix,
    int out_w,
    int out_h,
    bool use_norm = true,
    bool use_imagenet = true
) {
    const int img_w = img.cols;
    const int img_h = img.rows;

    float scale = std::min(out_w * 1.0f / img_w, out_h * 1.0f / img_h);
    int resize_w = static_cast<int>(round(img_w * scale));
    int resize_h = static_cast<int>(round(img_h * scale));

    int pad_w = out_w - resize_w;
    int pad_h = out_h - resize_h;
    int pad_left = static_cast<int>(round(pad_w / 2.0f - 0.1f));
    int pad_top = static_cast<int>(round(pad_h / 2.0f - 0.1f));

    transform_matrix << 1.0f / scale, 0, -pad_left / scale, 0, 1.0f / scale, -pad_top / scale, 0, 0,
        1;

    ncnn::Mat out = ncnn::Mat::from_pixels_resize(
        img.data,
        ncnn::Mat::PIXEL_BGR2RGB,
        img_w,
        img_h,
        resize_w,
        resize_h
    );

    int pad_right = out_w - resize_w - pad_left;
    int pad_bottom = out_h - resize_h - pad_top;

    ncnn::Mat padded;
    ncnn::copy_make_border(
        out,
        padded,
        pad_top,
        pad_bottom,
        pad_left,
        pad_right,
        ncnn::BORDER_CONSTANT,
        114.f
    );
    if (use_norm) {
        // 两种常用策略：
        // A) 仅 scale 到 [0,1] -> mean = {0,0,0}, norm = {1/255,1/255,1/255}
        // B) ImageNet (x/255 - mean)/std:
        //    mean_vals = mean * 255, norm_vals = 1/(std * 255)
        std::array<float, 3> mean_vals;
        std::array<float, 3> norm_vals;

        if (use_imagenet) {
            // 注意：这里顺序为 RGB（因为 from_pixels_resize 用的是 PIXEL_BGR2RGB）
            const std::array<float, 3> mean = { 0.485f, 0.456f, 0.406f }; // R,G,B
            const std::array<float, 3> stdv = { 0.229f, 0.224f, 0.225f }; // R,G,B

            for (int c = 0; c < 3; ++c) {
                mean_vals[c] = mean[c] * 255.0f; // mean * 255
                norm_vals[c] = 1.0f / (stdv[c] * 255.0f); // 1 / (std * 255)
            }
        } else {
            // 只做 /255 -> [0,1]
            mean_vals = { 0.f, 0.f, 0.f };
            norm_vals = { 1.0f / 255.0f, 1.0f / 255.0f, 1.0f / 255.0f };
        }

        // 执行归一化（会把数据转为 float 并按通道处理）
        padded.substract_mean_normalize(mean_vals.data(), norm_vals.data());
    }

    return padded;
}

RuneDetectorNCNN::RuneDetectorNCNN(
    std::string model_type,
    std::string input_name_,
    std::string output_name_,
    const std::string& model_path_param,
    const std::string& model_path_bin,
    float conf_threshold,
    int top_k,
    float nms_threshold,
    bool use_gpu,
    int cpu_threads,
    bool use_lightmode,
    int device_id
):
    input_name_(input_name_),
    output_name_(output_name_),
    model_path_param_(model_path_param),
    model_path_bin_(model_path_bin),
    conf_threshold_(conf_threshold),
    top_k_(top_k),
    nms_threshold_(nms_threshold),
    use_gpu_(use_gpu),
    cpu_threads_(cpu_threads),
    use_lightmode_(use_lightmode) {
    auto model = rune_infer::modeFromString(model_type);
    rune_infer_ =
        std::make_unique<rune_infer::RuneInfer>(model, conf_threshold, nms_threshold, top_k);
    init(device_id);
}
RuneDetectorNCNN::~RuneDetectorNCNN() {
    ncnn_net_.reset();
}

void RuneDetectorNCNN::init(int device_id) {
    ml_net::NCNNNet::Params params;
    params.model_path_param = model_path_param_;
    params.model_path_bin = model_path_bin_;
    params.input_name = input_name_;
    params.output_name = output_name_;
    params.use_vulkan = use_gpu_;
    params.device_id = device_id;
    params.use_light_mode = use_lightmode_;
    params.cpu_threads = cpu_threads_;
    ncnn_net_ = std::make_unique<ml_net::NCNNNet>();
    ncnn_net_->init(params);
    strides_ = { 8, 16, 32 };
    grid_strides_.clear();
    rune_infer_->generateGridsAndStride(
        rune_infer_->getInputW(),
        rune_infer_->getInputH(),
        strides_,
        grid_strides_
    );

    WUST_INFO("rune_ncnn") << "ncnn: model loaded successfully";
}

void RuneDetectorNCNN::pushInput(CommonFrame& frame) {
    frame.id = current_id_++;

    processCallback(frame);
}

void RuneDetectorNCNN::setCallback(CallbackType callback) {
    infer_callback_ = callback;
}

bool RuneDetectorNCNN::processCallback(const CommonFrame& frame) {
    Eigen::Matrix3f transform_matrix; // transform matrix from resized image to source image.
    // cv::Mat resized_img = letterbox(frame.src_img, transform_matrix);
    // ncnn::Mat in = ncnn::Mat::from_pixels(
    //     resized_img.data,
    //     ncnn::Mat::PIXEL_BGR2RGB, // OpenCV 默认 BGR，转为 RGB
    //     INPUT_W,
    //     INPUT_H
    // );
    ncnn::Mat in = letterbox_to_ncnn(
        frame.src_img,
        transform_matrix,
        rune_infer_->getInputW(),
        rune_infer_->getInputH(),
        rune_infer_->getUseNorm()
    );

    auto out = ncnn_net_->infer(in);

    cv::Mat output_buffer(out.h, out.w, CV_32F, out.data);
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

std::tuple<cv::Point2f, cv::Mat> RuneDetectorNCNN::detectRTag(
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
