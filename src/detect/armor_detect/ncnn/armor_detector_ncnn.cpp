#include "detect/armor_detect/ncnn/armor_detector_ncnn.hpp"
#include "common/gobal.hpp"
#include "common/logger.hpp"
#include <ncnn/simpleomp.h>
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
    std::vector<ArmorObject>& output_objs,
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

        ArmorObject obj;

        obj.pts.resize(4);

        obj.pts[0] = cv::Point2f(apex_dst(0, 0), apex_dst(1, 0));
        obj.pts[1] = cv::Point2f(apex_dst(0, 1), apex_dst(1, 1));
        obj.pts[2] = cv::Point2f(apex_dst(0, 2), apex_dst(1, 2));
        obj.pts[3] = cv::Point2f(apex_dst(0, 3), apex_dst(1, 3));

        auto rect = cv::boundingRect(obj.pts);

        obj.box = rect;
        obj.color = static_cast<ArmorColor>(color_id.x);
        obj.number = static_cast<ArmorNumber>(num_id.x);
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
static inline float intersection_area(const ArmorObject& a, const ArmorObject& b) {
    cv::Rect_<float> inter = a.box & b.box;
    return inter.area();
}

static void nms_merge_sorted_bboxes(
    std::vector<ArmorObject>& faceobjects,
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
        ArmorObject& a = faceobjects[i];

        int keep = 1;
        for (size_t j = 0; j < indices.size(); j++) {
            ArmorObject& b = faceobjects[indices[j]];

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

ArmorDetectNCNN::ArmorDetectNCNN(
    std::string input_name_,
    std::string output_name_,
    const std::string& model_path_param,
    const std::string& model_path_bin,
    const std::string& classify_model_path,
    const std::string& classify_label_path,
    const LightParams& l,
    const ArmorParams& a,
    double classifier_threshold,
    float conf_threshold,
    int top_k,
    float nms_threshold,
    float expand_ratio_w,
    float expand_ratio_h,
    int binary_thres_,
    bool use_gpu,
    int cpu_threads,
    bool use_lightmode,
    bool use_armor_detect_common,
    int device_id
):
    input_name_(input_name_),
    output_name_(output_name_),
    light_params_(l),
    armor_params_(a),
    model_path_bin_(model_path_bin),
    model_path_param_(model_path_param),
    conf_threshold_(conf_threshold),
    top_k_(top_k),
    nms_threshold_(nms_threshold),
    use_gpu(use_gpu),
    cpu_threads(cpu_threads),
    use_lightmode(use_lightmode),
    use_armor_detect_common(use_armor_detect_common) {
    if (use_armor_detect_common) {
        armor_detect_common_ = std::make_unique<ArmorDetectCommon>(
            classify_model_path,
            classify_label_path,
            l,
            a,
            classifier_threshold,
            expand_ratio_w,
            expand_ratio_h,
            binary_thres_
        );
    }

    init(device_id);
}
ArmorDetectNCNN::~ArmorDetectNCNN() {
    net_.clear();

    corner_corrector.reset();

    // if (opt_.use_vulkan_compute&&!gobal::ncnn_gpu_destroyed) {
    //     ncnn::destroy_gpu_instance();
    //     gobal::ncnn_gpu_destroyed = true;
    // }
}

void ArmorDetectNCNN::init(int device_id) {
    if (use_gpu) {
        ncnn::create_gpu_instance();
        opt_.use_vulkan_compute = true;
        ncnn::VulkanDevice* vkdev = ncnn::get_gpu_device(device_id);
        if (vkdev) {
            net_.set_vulkan_device(vkdev);
        }
        WUST_INFO("armor_ncnn") << "ncnn: use gpu";
    } else {
        opt_.use_vulkan_compute = false;
        WUST_INFO("armor_ncnn") << "ncnn: use cpu";
    }
    if (use_lightmode) {
        opt_.lightmode = true;
    }

    opt_.num_threads = cpu_threads;
    net_.opt = opt_;

    if (net_.load_param(model_path_param_.c_str()) != 0) {
        WUST_ERROR("armor_ncnn") << "Failed to load param";
        return;
    }
    if (net_.load_model(model_path_bin_.c_str()) != 0) {
        WUST_ERROR("armor_ncnn") << "Failed to load model";
        return;
    }

    int ret = net_.load_param((model_path_param_).c_str());
    if (ret != 0) {
        WUST_ERROR("armor_ncnn") << "Failed to load param file: " << model_path_param_;
        return;
    }

    ret = net_.load_model((model_path_bin_).c_str());
    if (ret != 0) {
        WUST_ERROR("armor_ncnn") << "Failed to load bin file: " << model_path_bin_;
        return;
    }

    // input_name_ = "images";
    // output_name_ = "output";

    strides_ = { 8, 16, 32 };
    generate_grids_and_stride(INPUT_W, INPUT_H, strides_, grid_strides_);
    corner_corrector = std::make_unique<LightCornerCorrector>();
}

void ArmorDetectNCNN::setCallback(DetectorCallback callback) {
    infer_callback_ = callback;
}
bool ArmorDetectNCNN::processCallback(
    const cv::Mat resized_img,
    Eigen::Matrix3f transform_matrix,
    std::chrono::steady_clock::time_point timestamp,
    const cv::Mat& src_img,
    const Eigen::Matrix4d& T_camera_to_odom,
    const Eigen::Vector3d& v
) {
    ncnn::Mat in =
        ncnn::Mat::from_pixels(resized_img.data, ncnn::Mat::PIXEL_BGR2RGB, INPUT_W, INPUT_H);

    ncnn::Extractor ex = net_.create_extractor();

    ex.input(input_name_.c_str(), in);

    ncnn::Mat out;
    ex.extract(output_name_.c_str(), out);

    cv::Mat output_buffer(out.h, out.w, CV_32F, out.data);

    std::vector<ArmorObject> objs_tmp, objs_result;
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
    std::sort(objs_tmp.begin(), objs_tmp.end(), [](const ArmorObject& a, const ArmorObject& b) {
        return a.prob > b.prob;
    });
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
    if (use_armor_detect_common) {
        std::vector<ArmorObject> armors = armor_detect_common_->detectNet(src_img, objs_result);
        // Call callback function
        if (this->infer_callback_) {
            this->infer_callback_(armors, timestamp, src_img, T_camera_to_odom, v);
            return true;
        }
    } else {
        if (this->infer_callback_) {
            this->infer_callback_(objs_result, timestamp, src_img, T_camera_to_odom, v);
            return true;
        }
    }

    return false;
}
void ArmorDetectNCNN::pushInput(
    const cv::Mat& rgb_img,
    std::chrono::steady_clock::time_point timestamp,
    const Eigen::Matrix4d& T_camera_to_odom,
    const Eigen::Vector3d& v
) {
    Eigen::Matrix3f transform_matrix;
    cv::Mat resized_img = letterbox(rgb_img, transform_matrix);
    processCallback(resized_img, transform_matrix, timestamp, rgb_img, T_camera_to_odom, v);
}
