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

#include "detect/rune_detect/onnxruntime/rune_detector_ort.hpp"
// std
#include <algorithm>
#include <numeric>
#include <unordered_map>
// third party
#include <opencv2/imgproc.hpp>
// project
#include "common/gobal.hpp"
#include "type/type.hpp"

static constexpr int INPUT_W = 480; // Width of input
static constexpr int INPUT_H = 480; // Height of input
static constexpr int NUM_CLASSES = 2; // Number of classes
static constexpr int NUM_COLORS = 2; // Number of color
static constexpr int NUM_POINTS = 5;
static constexpr int NUM_POINTS_2 = 2 * NUM_POINTS;
static constexpr float MERGE_CONF_ERROR = 0.15;
static constexpr float MERGE_MIN_IOU = 0.9;
// 由于训练失误，网络的颜色是反的
static std::unordered_map<int, EnemyColor> DNN_COLOR_TO_ENEMY_COLOR = { { 0, EnemyColor::BLUE },
                                                                        { 1, EnemyColor::RED } };

static cv::Mat letterbox(
    const cv::Mat& img,
    Eigen::Matrix3f& transform_matrix,
    std::vector<int> new_shape = { INPUT_W, INPUT_H }
) {
    // Get current image shape [height, width]

    int img_h = img.rows;
    int img_w = img.cols;

    // Compute scale ratio(new / old) and target resized shape
    float scale = std::min(new_shape[1] * 1.0 / img_h, new_shape[0] * 1.0 / img_w);
    int resize_h = static_cast<int>(round(img_h * scale));
    int resize_w = static_cast<int>(round(img_w * scale));

    // Compute padding
    int pad_h = new_shape[1] - resize_h;
    int pad_w = new_shape[0] - resize_w;

    // Resize and pad image while meeting stride-multiple constraints
    cv::Mat resized_img;
    cv::resize(img, resized_img, cv::Size(resize_w, resize_h));

    // divide padding into 2 sides
    float half_h = pad_h * 1.0 / 2;
    float half_w = pad_w * 1.0 / 2;

    // Compute padding boarder
    int top = static_cast<int>(round(half_h - 0.1));
    int bottom = static_cast<int>(round(half_h + 0.1));
    int left = static_cast<int>(round(half_w - 0.1));
    int right = static_cast<int>(round(half_w + 0.1));

    /* clang-format off */
    /* *INDENT-OFF* */

    // Compute point transform_matrix
    transform_matrix << 1.0 / scale, 0, -half_w / scale,
                        0, 1.0 / scale, -half_h / scale,
                        0, 0, 1;

    /* *INDENT-ON* */
    /* clang-format on */

    // Add border
    cv::copyMakeBorder(
        resized_img,
        resized_img,
        top,
        bottom,
        left,
        right,
        cv::BORDER_CONSTANT,
        cv::Scalar(114, 114, 114)
    );

    return resized_img;
}

// Generate grids and stride for post processing
// target_w: Width of input.
// target_h: Height of input.
// strides A vector of stride.
// grid_strides Grid stride generated in this function
static void generateGridsAndStride(
    const int target_w,
    const int target_h,
    std::vector<int>& strides,
    std::vector<GridAndStride>& grid_strides
) {
    for (auto stride: strides) {
        int num_grid_w = target_w / stride;
        int num_grid_h = target_h / stride;

        for (int g1 = 0; g1 < num_grid_h; g1++) {
            for (int g0 = 0; g0 < num_grid_w; g0++) {
                grid_strides.emplace_back(GridAndStride { g0, g1, stride });
            }
        }
    }
}

// Decode output tensor
static void generateProposals(
    std::vector<RuneObject>& output_objs,
    std::vector<float>& scores,
    std::vector<cv::Rect>& rects,
    const cv::Mat& output_buffer,
    const Eigen::Matrix<float, 3, 3>& transform_matrix,
    float conf_threshold,
    std::vector<GridAndStride> grid_strides
) {
    const int num_anchors = grid_strides.size();

    for (int anchor_idx = 0; anchor_idx < num_anchors; anchor_idx++) {
        float confidence = output_buffer.at<float>(anchor_idx, NUM_POINTS_2);
        if (confidence < conf_threshold) {
            continue;
        }

        const int grid0 = grid_strides[anchor_idx].grid0;
        const int grid1 = grid_strides[anchor_idx].grid1;
        const int stride = grid_strides[anchor_idx].stride;

        double color_score, class_score;
        cv::Point color_id, class_id;
        cv::Mat color_scores =
            output_buffer.row(anchor_idx).colRange(NUM_POINTS_2 + 1, NUM_POINTS_2 + 1 + NUM_COLORS);
        cv::Mat num_scores = output_buffer.row(anchor_idx)
                                 .colRange(
                                     NUM_POINTS_2 + 1 + NUM_COLORS,
                                     NUM_POINTS_2 + 1 + NUM_COLORS + NUM_CLASSES
                                 );
        // Argmax
        cv::minMaxLoc(color_scores, NULL, &color_score, NULL, &color_id);
        cv::minMaxLoc(num_scores, NULL, &class_score, NULL, &class_id);

        float x_1 = (output_buffer.at<float>(anchor_idx, 0) + grid0) * stride;
        float y_1 = (output_buffer.at<float>(anchor_idx, 1) + grid1) * stride;
        float x_2 = (output_buffer.at<float>(anchor_idx, 2) + grid0) * stride;
        float y_2 = (output_buffer.at<float>(anchor_idx, 3) + grid1) * stride;
        float x_3 = (output_buffer.at<float>(anchor_idx, 4) + grid0) * stride;
        float y_3 = (output_buffer.at<float>(anchor_idx, 5) + grid1) * stride;
        float x_4 = (output_buffer.at<float>(anchor_idx, 6) + grid0) * stride;
        float y_4 = (output_buffer.at<float>(anchor_idx, 7) + grid1) * stride;
        float x_5 = (output_buffer.at<float>(anchor_idx, 8) + grid0) * stride;
        float y_5 = (output_buffer.at<float>(anchor_idx, 9) + grid1) * stride;

        Eigen::Matrix<float, 3, 5> apex_norm;
        Eigen::Matrix<float, 3, 5> apex_dst;

        /* clang-format off */
        /* *INDENT-OFF* */
        apex_norm << x_1, x_2, x_3, x_4, x_5,
                    y_1, y_2, y_3, y_4, y_5,
                    1,   1,   1,   1,   1;
        /* *INDENT-ON* */
        /* clang-format on */

        apex_dst = transform_matrix * apex_norm;

        RuneObject obj;

        obj.pts.r_center = cv::Point2f(apex_dst(0, 0), apex_dst(1, 0));
        obj.pts.bottom_left = cv::Point2f(apex_dst(0, 1), apex_dst(1, 1));
        obj.pts.top_left = cv::Point2f(apex_dst(0, 2), apex_dst(1, 2));
        obj.pts.top_right = cv::Point2f(apex_dst(0, 3), apex_dst(1, 3));
        obj.pts.bottom_right = cv::Point2f(apex_dst(0, 4), apex_dst(1, 4));

        auto rect = cv::boundingRect(obj.pts.toVector2f());

        obj.box = rect;
        obj.color = DNN_COLOR_TO_ENEMY_COLOR[color_id.x];
        obj.type = static_cast<RuneType>(class_id.x);
        obj.prob = confidence;

        rects.push_back(rect);
        scores.push_back(confidence);
        output_objs.push_back(std::move(obj));
    }
}

// Calculate intersection area between Object a and Object b.
static inline float intersectionArea(const RuneObject& a, const RuneObject& b) {
    cv::Rect_<float> inter = a.box & b.box;
    return inter.area();
}

static void nmsMergeSortedBboxes(
    std::vector<RuneObject>& faceobjects,
    std::vector<int>& indices,
    float nms_threshold
) {
    indices.clear();

    const int n = faceobjects.size();

    std::vector<float> areas(n);
    for (int i = 0; i < n; i++) {
        areas[i] = faceobjects[i].box.area();
    }

    for (int i = 0; i < n; i++) {
        RuneObject& a = faceobjects[i];

        int keep = 1;
        for (size_t j = 0; j < indices.size(); j++) {
            RuneObject& b = faceobjects[indices[j]];

            // intersection over union
            float inter_area = intersectionArea(a, b);
            float union_area = areas[i] + areas[indices[j]] - inter_area;
            float iou = inter_area / union_area;
            if (iou > nms_threshold || isnan(iou)) {
                keep = 0;
                // Stored for Merge
                if (a.type == b.type && a.color == b.color && iou > MERGE_MIN_IOU
                    && abs(a.prob - b.prob) < MERGE_CONF_ERROR)
                {
                    a.pts.children.push_back(b.pts);
                }
                // cout<<b.pts_x.size()<<endl;
            }
        }

        if (keep) {
            indices.push_back(i);
        }
    }
}

RuneDetectorOnnxRuntime::RuneDetectorOnnxRuntime(
    const std::filesystem::path& model_path,
    float conf_threshold,
    int top_k,
    float nms_threshold,
    bool use_gpu_,
    int device_id_
):
    model_path_(model_path),
    conf_threshold_(conf_threshold),
    top_k_(top_k),
    nms_threshold_(nms_threshold),
    use_gpu_(use_gpu_),
    device_id_(device_id_) {
    init();
}

void RuneDetectorOnnxRuntime::init() {
    // 1) 初始化 ORT 环境
    env_ = std::make_unique<Ort::Env>(ORT_LOGGING_LEVEL_WARNING, "ArmorDetectONNX");
    Ort::SessionOptions session_options;
    session_options.SetIntraOpNumThreads(4);

    // 启用优化
    session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);

    // 如果需要使用 GPU
    if (use_gpu_) {
        OrtSessionOptionsAppendExecutionProvider_CUDA(session_options, 0); // GPU: CUDA 0 号设备
    }

    // 2) 加载模型
    session_ = std::make_unique<Ort::Session>(*env_, model_path_.c_str(), session_options);

    // 3) 获取输入输出信息
    Ort::AllocatorWithDefaultOptions allocator;

    // ✅ GetInputNameAllocated 返回智能指针，自动释放
    Ort::AllocatedStringPtr input_name_ptr = session_->GetInputNameAllocated(0, allocator);
    input_name_ = std::string(input_name_ptr.get()); // ✅ 正确复制字符串内容

    auto input_type_info = session_->GetInputTypeInfo(0);
    auto input_tensor_info = input_type_info.GetTensorTypeAndShapeInfo();
    input_dims_ = input_tensor_info.GetShape(); // 一般是 {1, 3, 640, 640}

    // ✅ GetOutputNameAllocated 同理
    Ort::AllocatedStringPtr output_name_ptr = session_->GetOutputNameAllocated(0, allocator);
    output_name_ = std::string(output_name_ptr.get()); // ✅ 避免悬空指针

    // 4) 初始化 strides/grid 等（YOLO 结构需要）
    strides_ = { 8, 16, 32 };
    generateGridsAndStride(INPUT_W, INPUT_H, strides_, grid_strides_);
}

void RuneDetectorOnnxRuntime::pushInput(
    const cv::Mat& rgb_img,
    std::chrono::steady_clock::time_point timestamp,
    Eigen::Matrix4d T_camera_to_odom
) {
    // Reprocess
    Eigen::Matrix3f transform_matrix; // transform matrix from resized image to source image.
    cv::Mat resized_img = letterbox(rgb_img, transform_matrix);
    processCallback(resized_img, transform_matrix, timestamp, rgb_img, T_camera_to_odom);
}

void RuneDetectorOnnxRuntime::setCallback(CallbackType callback) {
    infer_callback_ = callback;
}

bool RuneDetectorOnnxRuntime::processCallback(
    const cv::Mat resized_img,
    Eigen::Matrix3f transform_matrix,
    std::chrono::steady_clock::time_point timestamp,
    const cv::Mat& src_img,
    Eigen::Matrix4d T_camera_to_odom
) {
    // BGR->RGB, u8(0-255)->f32(0.0-1.0), HWC->NCHW
    // note: TUP's model no need to normalize
    cv::Mat img_float;
    resized_img.convertTo(img_float, CV_32F, 1.0); // 归一化到 0~1
    cv::cvtColor(img_float, img_float, cv::COLOR_BGR2RGB);

    // HWC -> CHW
    std::vector<float> input_tensor_values(INPUT_W * INPUT_H * 3);
    int channel_size = INPUT_W * INPUT_H;
    for (int c = 0; c < 3; ++c)
        for (int h = 0; h < INPUT_H; ++h)
            for (int w = 0; w < INPUT_W; ++w)
                input_tensor_values[c * channel_size + h * INPUT_W + w] =
                    img_float.at<cv::Vec3f>(h, w)[c];

    // 2) 创建输入张量
    Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
        memory_info,
        input_tensor_values.data(),
        input_tensor_values.size(),
        input_dims_.data(), // e.g. {1,3,640,640}
        input_dims_.size()
    );

    // 3) 推理
    const char* input_names[] = { input_name_.c_str() };
    const char* output_names[] = { output_name_.c_str() };

    auto output_tensors =
        session_->Run(Ort::RunOptions { nullptr }, input_names, &input_tensor, 1, output_names, 1);

    // 4) 获取输出张量数据
    float* output_data = output_tensors.front().GetTensorMutableData<float>();

    // 假设输出维度已知，例如 [1, 3549, 21]
    auto output_shape = output_tensors.front().GetTensorTypeAndShapeInfo().GetShape();
    int rows = static_cast<int>(output_shape[1]);
    int cols = static_cast<int>(output_shape[2]);

    // 5) 用 cv::Mat 包装输出，方便后续处理
    cv::Mat output_buffer(rows, cols, CV_32F, output_data);
    // Parsed variable
    std::vector<RuneObject> objs_tmp, objs_result;
    std::vector<cv::Rect> rects;
    std::vector<float> scores;
    std::vector<int> indices;

    // Parse YOLO output
    generateProposals(
        objs_tmp,
        scores,
        rects,
        output_buffer,
        transform_matrix,
        this->conf_threshold_,
        this->grid_strides_
    );

    // TopK
    std::sort(objs_tmp.begin(), objs_tmp.end(), [](const RuneObject& a, const RuneObject& b) {
        return a.prob > b.prob;
    });
    if (objs_tmp.size() > static_cast<size_t>(this->top_k_)) {
        objs_tmp.resize(this->top_k_);
    }

    nmsMergeSortedBboxes(objs_tmp, indices, this->nms_threshold_);

    for (size_t i = 0; i < indices.size(); i++) {
        objs_result.push_back(std::move(objs_tmp[indices[i]]));

        if (objs_result[i].pts.children.size() > 0) {
            const float N = static_cast<float>(objs_result[i].pts.children.size() + 1);
            FeaturePoints pts_final = std::accumulate(
                objs_result[i].pts.children.begin(),
                objs_result[i].pts.children.end(),
                objs_result[i].pts
            );
            objs_result[i].pts = pts_final / N;
        }
    }
    objs_result.erase(
        std::remove_if(
            objs_result.begin(),
            objs_result.end(),
            [c = static_cast<EnemyColor>(gobal::detect_color)](const auto& objs_result) {
                return objs_result.color != c;
            }
        ),
        objs_result.end()
    );

    // NMS & TopK
    // cv::dnn::NMSBoxes(
    //   rects, scores, this->conf_threshold_, this->nms_threshold_, indices, 1.0,
    //   this->top_k_);
    // for (size_t i = 0; i < indices.size(); ++i) {
    //   objs_result.push_back(std::move(objs_tmp[i]));
    // }

    // Call callback function
    if (this->infer_callback_) {
        this->infer_callback_(objs_result, timestamp, src_img, T_camera_to_odom);
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
