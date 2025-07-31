// Copyright 2025 Zikang Xie
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

#include "detect/armor_detect/openvino/armor_detector_openvino.hpp"
#include "common/gobal.hpp"
#include "common/logger.hpp"
#include <functional>
#include <opencv2/highgui.hpp>
static const int INPUT_W = 416; // Width of input
static const int INPUT_H = 416; // Height of input
static constexpr int NUM_CLASSES = 8; // Number of classes
static constexpr int NUM_COLORS = 4; // Number of color
static constexpr float MERGE_CONF_ERROR = 0.15;
static constexpr float MERGE_MIN_IOU = 0.9;
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

static void letterbox_into(
    const cv::Mat& img,
    uint8_t* dst_data,
    Eigen::Matrix3f& transform_matrix,
    int dst_w,
    int dst_h
) {
    int img_h = img.rows;
    int img_w = img.cols;

    float scale = std::min(dst_h * 1.0f / img_h, dst_w * 1.0f / img_w);
    int resize_h = static_cast<int>(round(img_h * scale));
    int resize_w = static_cast<int>(round(img_w * scale));

    int pad_h = dst_h - resize_h;
    int pad_w = dst_w - resize_w;

    float half_h = pad_h / 2.0f;
    float half_w = pad_w / 2.0f;

    int top = static_cast<int>(round(half_h - 0.1f));
    int bottom = static_cast<int>(round(half_h + 0.1f));
    int left = static_cast<int>(round(half_w - 0.1f));
    int right = static_cast<int>(round(half_w + 0.1f));

    transform_matrix << 1.0f / scale, 0, -half_w / scale, 0, 1.0f / scale, -half_h / scale, 0, 0, 1;

    cv::Mat resized_img;
    cv::resize(img, resized_img, cv::Size(resize_w, resize_h));

    cv::Mat final_img(dst_h, dst_w, CV_8UC3, cv::Scalar(114, 114, 114));

    resized_img.copyTo(final_img(cv::Rect(left, top, resize_w, resize_h)));

    std::memcpy(dst_data, final_img.data, dst_h * dst_w * 3);
}

static void generate_grids_and_stride(
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
static void generate_proposals(
    std::vector<armor::ArmorObject>& output_objs,
    std::vector<float>& scores,
    std::vector<cv::Rect>& rects,
    const cv::Mat& output_buffer,
    const Eigen::Matrix<float, 3, 3>& transform_matrix,
    float conf_threshold,
    std::vector<GridAndStride> grid_strides
) {
    const int num_anchors = grid_strides.size();

    for (int anchor_idx = 0; anchor_idx < num_anchors; anchor_idx++) {
        float confidence = output_buffer.at<float>(anchor_idx, 8);
        if (confidence < conf_threshold) {
            continue;
        }

        const int grid0 = grid_strides[anchor_idx].grid0;
        const int grid1 = grid_strides[anchor_idx].grid1;
        const int stride = grid_strides[anchor_idx].stride;

        double color_score, num_score;
        cv::Point color_id, num_id;
        cv::Mat color_scores = output_buffer.row(anchor_idx).colRange(9, 9 + NUM_COLORS);
        cv::Mat num_scores =
            output_buffer.row(anchor_idx).colRange(9 + NUM_COLORS, 9 + NUM_COLORS + NUM_CLASSES);
        // Argmax
        cv::minMaxLoc(color_scores, NULL, &color_score, NULL, &color_id);
        cv::minMaxLoc(num_scores, NULL, &num_score, NULL, &num_id);

        float x_1 = (output_buffer.at<float>(anchor_idx, 0) + grid0) * stride;
        float y_1 = (output_buffer.at<float>(anchor_idx, 1) + grid1) * stride;
        float x_2 = (output_buffer.at<float>(anchor_idx, 2) + grid0) * stride;
        float y_2 = (output_buffer.at<float>(anchor_idx, 3) + grid1) * stride;
        float x_3 = (output_buffer.at<float>(anchor_idx, 4) + grid0) * stride;
        float y_3 = (output_buffer.at<float>(anchor_idx, 5) + grid1) * stride;
        float x_4 = (output_buffer.at<float>(anchor_idx, 6) + grid0) * stride;
        float y_4 = (output_buffer.at<float>(anchor_idx, 7) + grid1) * stride;

        Eigen::Matrix<float, 3, 4> apex_norm;
        Eigen::Matrix<float, 3, 4> apex_dst;

        /* clang-format off */
        /* *INDENT-OFF* */
        apex_norm << x_1, x_2, x_3, x_4,
                    y_1, y_2, y_3, y_4,
                    1,   1,   1,   1;
        /* *INDENT-ON* */
        /* clang-format on */

        apex_dst = transform_matrix * apex_norm;

        armor::ArmorObject obj;

        obj.pts.resize(4);

        obj.pts[0] = cv::Point2f(apex_dst(0, 0), apex_dst(1, 0));
        obj.pts[1] = cv::Point2f(apex_dst(0, 1), apex_dst(1, 1));
        obj.pts[2] = cv::Point2f(apex_dst(0, 2), apex_dst(1, 2));
        obj.pts[3] = cv::Point2f(apex_dst(0, 3), apex_dst(1, 3));

        auto rect = cv::boundingRect(obj.pts);

        obj.box = rect;
        obj.color = static_cast<armor::ArmorColor>(color_id.x);
        obj.number = static_cast<armor::ArmorNumber>(num_id.x);
        obj.prob = confidence;

        rects.push_back(rect);
        scores.push_back(confidence);
        output_objs.push_back(std::move(obj));
    }
}

/**
 * @brief Calculate intersection area between two objects.
 * @param a Object a.
 * @param b Object b.
 * @return Area of intersection.
 */
static inline float intersection_area(const armor::ArmorObject& a, const armor::ArmorObject& b) {
    cv::Rect_<float> inter = a.box & b.box;
    return inter.area();
}

static void nms_merge_sorted_bboxes(
    std::vector<armor::ArmorObject>& faceobjects,
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
        armor::ArmorObject& a = faceobjects[i];

        int keep = 1;
        for (size_t j = 0; j < indices.size(); j++) {
            armor::ArmorObject& b = faceobjects[indices[j]];

            // intersection over union
            float inter_area = intersection_area(a, b);
            float union_area = areas[i] + areas[indices[j]] - inter_area;
            float iou = inter_area / union_area;
            if (iou > nms_threshold || isnan(iou)) {
                keep = 0;
                // Stored for Merge
                if (a.number == b.number && a.color == b.color && iou > MERGE_MIN_IOU
                    && abs(a.prob - b.prob) < MERGE_CONF_ERROR)
                {
                    for (int i = 0; i < 4; i++) {
                        b.pts.push_back(a.pts[i]);
                    }
                }
            }
        }

        if (keep) {
            indices.push_back(i);
        }
    }
}
ArmorDetectOpenVino::ArmorDetectOpenVino(
    const std::filesystem::path& model_path,
    const std::string& device_name,
    const ArmorDetectCommonParams& armor_detect_common_params,
    float conf_threshold,
    int top_k,
    float nms_threshold,
    bool use_throughputmode,
    bool use_armor_detect_common
):

    model_path_(model_path),
    device_name_(device_name),
    conf_threshold_(conf_threshold),
    top_k_(top_k),
    nms_threshold_(nms_threshold),
    use_throughputmode_(use_throughputmode),
    use_armor_detect_common_(use_armor_detect_common) {
    if (use_armor_detect_common_) {
        armor_detect_common_ = std::make_unique<ArmorDetectCommon>(armor_detect_common_params);
    }

    init();
}

void ArmorDetectOpenVino::init() {
    // 1) Core／Model 读取
    if (!ov_core_) {
        ov_core_ = std::make_unique<ov::Core>();
    }
    // load IR
    model_ = ov_core_->read_model(model_path_);

    // 2) PrePostProcessor 配置
    ov::preprocess::PrePostProcessor ppp(model_);

    // 告诉引擎：输入 tensor 是 u8、NHWC、BGR
    ppp.input()
        .tensor()
        .set_element_type(ov::element::u8)
        .set_layout("NHWC")
        .set_color_format(ov::preprocess::ColorFormat::BGR);

    // 预处理管线：u8→f32、BGR→RGB、除以 255
    ppp.input()
        .preprocess()
        .convert_element_type(ov::element::f32)
        .convert_color(ov::preprocess::ColorFormat::RGB)
        .scale({ 1.0f, 1.0f, 1.0f });

    // 告诉引擎：模型内部期望的布局是 NCHW
    ppp.input().model().set_layout("NCHW");

    // 输出也要 f32
    ppp.output().tensor().set_element_type(ov::element::f32);

    // 把预处理节点「贴」到模型里
    model_ = ppp.build();

    // 3) 编译模型（可以带 performance_mode hint）
    ov::hint::PerformanceMode mode = use_throughputmode_ ? ov::hint::PerformanceMode::THROUGHPUT
                                                         : ov::hint::PerformanceMode::LATENCY;
    compiled_model_ = std::make_unique<ov::CompiledModel>(
        ov_core_->compile_model(model_, device_name_, ov::hint::performance_mode(mode))
    );

    // 4) 生成 grid_strides、ThreadPool 等
    strides_ = { 8, 16, 32 };
    generate_grids_and_stride(INPUT_W, INPUT_H, strides_, grid_strides_);
}

ArmorDetectOpenVino::~ArmorDetectOpenVino() {
    compiled_model_.reset();
    ov_core_.reset();
}

void ArmorDetectOpenVino::setCallback(DetectorCallback callback) {
    infer_callback_ = callback;
}
bool ArmorDetectOpenVino::processCallback(const CommonFrame& frame) {
    auto start = std::chrono::steady_clock::now();
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

    // Feed blob into input
    // auto input_port = compiled_model_->input();
    // ov::Tensor input_tensor(
    //     input_port.get_element_type(),
    //     ov::Shape(std::vector<size_t> { 1, 3, INPUT_W, INPUT_H }),
    //     blob.ptr(0)
    // );

    // // Start inference
    // auto infer_request = compiled_model_->create_infer_request();
    // infer_request.set_input_tensor(input_tensor);
    // infer_request.infer();
    // cv::Mat resized_img = letterbox(frame.src_img, transform_matrix);
    // ov::Tensor input_tensor = ov::Tensor(
    //     compiled_model_->input().get_element_type(),
    //     compiled_model_->input().get_shape(),
    //     resized_img.data
    // );
    auto input_tensor = ov::Tensor(
        compiled_model_->input().get_element_type(),
        compiled_model_->input().get_shape()
    );
    auto* data_ptr = input_tensor.data<uint8_t>();

    letterbox_into(frame.src_img, data_ptr, transform_matrix, INPUT_W, INPUT_H);
    // 3) InferRequest
    auto infer_request = compiled_model_->create_infer_request();
    infer_request.set_input_tensor(input_tensor);
    infer_request.infer();
    auto output = infer_request.get_output_tensor();

    // Process output data
    auto output_shape = output.get_shape();
    // 3549 x 21 Matrix
    cv::Mat output_buffer(output_shape[1], output_shape[2], CV_32F, output.data());

    // Parsed variable
    std::vector<armor::ArmorObject> objs_tmp, objs_result;
    std::vector<cv::Rect> rects;
    std::vector<float> scores;
    std::vector<int> indices;

    // Parse YOLO output
    generate_proposals(
        objs_tmp,
        scores,
        rects,
        output_buffer,
        transform_matrix,
        this->conf_threshold_,
        this->grid_strides_
    );
    auto end = std::chrono::steady_clock::now();
    // WUST_INFO("TRT") << "TRT"
    //                  << "Infer time: "
    //                  << std::chrono::duration_cast<std::chrono::microseconds>(end - start).count()
    //         / 1000.0 << "ms";
    // TopK
    std::sort(
        objs_tmp.begin(),
        objs_tmp.end(),
        [](const armor::ArmorObject& a, const armor::ArmorObject& b) { return a.prob > b.prob; }
    );
    if (objs_tmp.size() > static_cast<size_t>(this->top_k_)) {
        objs_tmp.resize(this->top_k_);
    }

    nms_merge_sorted_bboxes(objs_tmp, indices, this->nms_threshold_);

    for (size_t i = 0; i < indices.size(); i++) {
        objs_result.push_back(std::move(objs_tmp[indices[i]]));

        if (objs_result[i].pts.size() >= 8) {
            auto n = objs_result[i].pts.size();
            cv::Point2f pts_final[4];

            for (size_t j = 0; j < n; j++) {
                pts_final[j % 4] += objs_result[i].pts[j];
            }

            objs_result[i].pts.resize(4);
            for (int j = 0; j < 4; j++) {
                pts_final[j].x /= static_cast<float>(n) / 4.0;
                pts_final[j].y /= static_cast<float>(n) / 4.0;
                objs_result[i].pts[j] = pts_final[j];
            }
        }
    }

    std::vector<armor::ArmorObject> armors;
    if (use_armor_detect_common_) {
        armors = armor_detect_common_->detectNet(frame.src_img, objs_result);
        // Call callback function
        if (this->infer_callback_) {
            this->infer_callback_(armors, frame);
            return true;
        }
    } else {
        for (auto obj: objs_result) {
            if (gobal::detect_color == 0 && obj.color == armor::ArmorColor::BLUE) {
                continue;
            } else if (gobal::detect_color == 1 && obj.color == armor::ArmorColor::RED) {
                continue;
            }
        }
        if (this->infer_callback_) {
            this->infer_callback_(armors, frame);
            return true;
        }
    }

    return false;
}
void ArmorDetectOpenVino::pushInput(const CommonFrame& frame) {
    processCallback(frame);
}