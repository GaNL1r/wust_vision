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

#include "rune_detector_openvino.hpp"
// std
#include <algorithm>
#include <numeric>
#include <unordered_map>
// third party
#include <opencv2/imgproc.hpp>
// project
#include "tasks/auto_buff/rune_detect/rune_infer.hpp"
#include "tasks/auto_buff/type.hpp"

RuneDetectorOpenvino::RuneDetectorOpenvino(
    std::string model_type,
    const std::filesystem::path& model_path,
    const std::string& device_name,
    float conf_threshold,
    int top_k,
    float nms_threshold,
    bool use_throughputmode_
):
    model_path_(model_path),
    device_name_(device_name),
    conf_threshold_(conf_threshold),
    top_k_(top_k),
    nms_threshold_(nms_threshold),
    use_throughputmode_(use_throughputmode_) {
    auto model = rune_infer::modeFromString(model_type);
    rune_infer_ =
        std::make_unique<rune_infer::RuneInfer>(model, conf_threshold, nms_threshold, top_k);
    init();
}

void RuneDetectorOpenvino::init() {
    openvino_net_ = std::make_unique<ml_net::OpenvinoNet>();
    auto ppp_init_fun = [this](ov::preprocess::PrePostProcessor& ppp) {
        // 1) 设置输入输出 tensor 的类型、布局、颜色格式等
        ppp.input()
            .tensor()
            .set_element_type(ov::element::u8)
            .set_layout("NHWC")
            .set_color_format(ov::preprocess::ColorFormat::BGR);

        // 预处理管线：u8→f32、BGR→RGB
        float scale = rune_infer_->getUseNorm() ? 255.0f : 1.0f;
        ppp.input()
            .preprocess()
            .convert_element_type(ov::element::f32)
            .convert_color(ov::preprocess::ColorFormat::RGB)
            .scale({ scale, scale, scale });

        // 告诉引擎：模型内部期望的布局是 NCHW
        ppp.input().model().set_layout("NCHW");

        // 输出也要 f32
        ppp.output().tensor().set_element_type(ov::element::f32);
    };
    ml_net::OpenvinoNet::Params params;
    params.model_path = model_path_;
    params.device_name = device_name_;
    params.mode = use_throughputmode_ ? ov::hint::PerformanceMode::THROUGHPUT
                                      : ov::hint::PerformanceMode::LATENCY;
    openvino_net_->init(params, ppp_init_fun);

    strides_ = { 8, 16, 32 };
    rune_infer_->generateGridsAndStride(
        rune_infer_->getInputW(),
        rune_infer_->getInputH(),
        strides_,
        grid_strides_
    );
}

void RuneDetectorOpenvino::pushInput(CommonFrame& frame) {
    frame.id = current_id_++;
    processCallback(frame);
}

void RuneDetectorOpenvino::setCallback(CallbackType callback) {
    infer_callback_ = callback;
}

bool RuneDetectorOpenvino::processCallback(const CommonFrame& frame) {
    Eigen::Matrix3f transform_matrix;
    // BGR->RGB, u8(0-255)->f32(0.0-1.0), HWC->NCHW
    // note: TUP's model no need to normalize
    // cv::Mat blob = cv::dnn::blobFromImage(
    //     resized_img,
    //     1.,
    //     cv::Size(INPUT_W, INPUT_H),
    //     cv::Scalar(0, 0, 0),
    //     true
    // );

    // // Feed blob into input
    // auto input_port = compiled_model_->input();
    // ov::Tensor input_tensor(
    //     input_port.get_element_type(),
    //     ov::Shape(std::vector<size_t> { 1, 3, INPUT_W, INPUT_H }),
    //     blob.ptr(0)
    // );
    // ov::Tensor input_tensor = ov::Tensor(
    //     compiled_model_->input().get_element_type(), // u8
    //     compiled_model_->input().get_shape(), // {1, H, W, 3}
    //     resized_img.data // 原始 BGR 数据
    // );
    cv::Mat resized_img = rune_infer_->letterbox(
        frame.src_img,
        transform_matrix,
        rune_infer_->getInputW(),
        rune_infer_->getInputH()
    );
    auto input_info = openvino_net_->getInputInfo();
    auto input_tensor = ov::Tensor(input_info.first, input_info.second, resized_img.data);
    auto output = openvino_net_->infer(input_tensor);

    // Process output data
    auto output_shape = output.get_shape();
    // 3549 x 21 Matrix
    cv::Mat output_buffer(output_shape[1], output_shape[2], CV_32F, output.data());

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

std::tuple<cv::Point2f, cv::Mat> RuneDetectorOpenvino::detectRTag(
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
