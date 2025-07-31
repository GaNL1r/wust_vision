#include "detect/armor_detect/onnxruntime/armor_detector_ort.hpp"
#include "common/gobal.hpp"

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
static void letterbox_into_CHW_uint8_rgb(
    const cv::Mat& img,
    float* output_ptr,
    Eigen::Matrix3f& transform_matrix,
    int dst_width,
    int dst_height
) {
    int img_h = img.rows;
    int img_w = img.cols;

    float scale = std::min(dst_height * 1.0f / img_h, dst_width * 1.0f / img_w);
    int resize_h = static_cast<int>(round(img_h * scale));
    int resize_w = static_cast<int>(round(img_w * scale));

    int pad_h = dst_height - resize_h;
    int pad_w = dst_width - resize_w;
    int top = static_cast<int>(round(pad_h / 2.0f - 0.1f));
    int left = static_cast<int>(round(pad_w / 2.0f - 0.1f));

    // Compute inverse affine transform matrix
    transform_matrix << 1.0f / scale, 0, -left / scale, 0, 1.0f / scale, -top / scale, 0, 0, 1;

    // Resize input image first
    cv::Mat resized;
    cv::resize(img, resized, cv::Size(resize_w, resize_h), 0, 0, cv::INTER_LINEAR);

    // Fill CHW uint8 RGB into output_ptr with padding (but still stored in float)
    for (int c = 0; c < 3; ++c) {
        for (int y = 0; y < dst_height; ++y) {
            for (int x = 0; x < dst_width; ++x) {
                int out_index = c * dst_height * dst_width + y * dst_width + x;
                if (y >= top && y < top + resize_h && x >= left && x < left + resize_w) {
                    int ry = y - top;
                    int rx = x - left;
                    cv::Vec3b pixel = resized.at<cv::Vec3b>(ry, rx); // BGR
                    output_ptr[out_index] = static_cast<float>(pixel[2 - c]); // R,G,B → CHW
                } else {
                    output_ptr[out_index] = 114.0f; // padding value (not normalized)
                }
            }
        }
    }
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
ArmorDetectOnnxRuntime::ArmorDetectOnnxRuntime(
    const std::filesystem::path& model_path,
    const ArmorDetectCommonParams& armor_detect_common_params,
    float conf_threshold,
    int top_k,
    float nms_threshold,
    bool use_gpu_,
    int device_id_,
    bool use_armor_detect_common
):

    model_path_(model_path),
    conf_threshold_(conf_threshold),
    top_k_(top_k),
    nms_threshold_(nms_threshold),
    use_gpu_(use_gpu_),
    device_id_(device_id_),
    use_armor_detect_common_(use_armor_detect_common) {
    if (use_armor_detect_common_) {
        armor_detect_common_ = std::make_unique<ArmorDetectCommon>(armor_detect_common_params);
    }

    init();
}

void ArmorDetectOnnxRuntime::init() {
    env_ = std::make_unique<Ort::Env>(ORT_LOGGING_LEVEL_WARNING, "ArmorDetectONNX");
    Ort::SessionOptions session_options;
    session_options.SetIntraOpNumThreads(4);

    session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);

    if (use_gpu_) {
        OrtSessionOptionsAppendExecutionProvider_CUDA(session_options, device_id_);
    }

    session_ = std::make_unique<Ort::Session>(*env_, model_path_.c_str(), session_options);

    Ort::AllocatorWithDefaultOptions allocator;

    Ort::AllocatedStringPtr input_name_ptr = session_->GetInputNameAllocated(0, allocator);
    input_name_ = std::string(input_name_ptr.get());

    auto input_type_info = session_->GetInputTypeInfo(0);
    auto input_tensor_info = input_type_info.GetTensorTypeAndShapeInfo();
    input_dims_ = input_tensor_info.GetShape();

    Ort::AllocatedStringPtr output_name_ptr = session_->GetOutputNameAllocated(0, allocator);
    output_name_ = std::string(output_name_ptr.get());

    strides_ = { 8, 16, 32 };
    generate_grids_and_stride(INPUT_W, INPUT_H, strides_, grid_strides_);
}

ArmorDetectOnnxRuntime::~ArmorDetectOnnxRuntime() {
    // 释放资源
    session_.reset();
    env_.reset();
}

void ArmorDetectOnnxRuntime::setCallback(DetectorCallback callback) {
    infer_callback_ = callback;
}
bool ArmorDetectOnnxRuntime::processCallback(const CommonFrame& frame) {
    Eigen::Matrix3f transform_matrix;

    std::vector<float> input_tensor_values(INPUT_W * INPUT_H * 3);
    letterbox_into_CHW_uint8_rgb(
        frame.src_img,
        input_tensor_values.data(),
        transform_matrix,
        INPUT_W,
        INPUT_H
    );

    Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
        memory_info,
        input_tensor_values.data(),
        input_tensor_values.size(),
        input_dims_.data(),
        input_dims_.size()
    );

    const char* input_names[] = { input_name_.c_str() };
    const char* output_names[] = { output_name_.c_str() };
    auto output_tensors =
        session_->Run(Ort::RunOptions { nullptr }, input_names, &input_tensor, 1, output_names, 1);

    float* output_data = output_tensors.front().GetTensorMutableData<float>();
    auto output_shape = output_tensors.front().GetTensorTypeAndShapeInfo().GetShape();
    int rows = static_cast<int>(output_shape[1]);
    int cols = static_cast<int>(output_shape[2]);
    cv::Mat output_buffer(rows, cols, CV_32F, output_data);

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
void ArmorDetectOnnxRuntime::pushInput(const CommonFrame& frame) {
    processCallback(frame);
}