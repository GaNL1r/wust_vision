#include "detect/armor_detect/armor_infer.hpp"
#include <algorithm>
#include <cmath>

namespace armor_infer {

static const int RP_INPUT_W = 640;
static const int RP_INPUT_H = 640;
static constexpr int RP_NUM_CLASSES = 9;
static constexpr int RP_NUM_COLORS = 4;
static const int TUP_INPUT_W = 416;
static const int TUP_INPUT_H = 416;
static constexpr int TUP_NUM_CLASSES = 8;
static constexpr int TUP_NUM_COLORS = 4;
static constexpr float MERGE_CONF_ERROR = 0.15f;
static constexpr float MERGE_MIN_IOU = 0.9f;

ArmorInfer::ArmorInfer(Mode mode, float conf_threshold, float nms_threshold, int top_k):
    mode_(mode),
    conf_threshold_(conf_threshold),
    nms_threshold_(nms_threshold),
    top_k_(top_k) {
    if (mode_ == Mode::TUP) {
        input_w_ = TUP_INPUT_W;
        input_h_ = TUP_INPUT_H;
        use_norm_ = false;
    } else {
        input_w_ = RP_INPUT_W;
        input_h_ = RP_INPUT_H;
        use_norm_ = true;
    }
}

void ArmorInfer::generate_grids_and_stride(
    const int target_w,
    const int target_h,
    const std::vector<int>& strides,
    std::vector<GridAndStride>& grid_strides
) {
    grid_strides.clear();
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

cv::Mat ArmorInfer::letterbox(
    const cv::Mat& img,
    Eigen::Matrix3f& transform_matrix,
    const int new_shape_w,
    const int new_shape_h
) {
    int img_h = img.rows;
    int img_w = img.cols;

    // new_shape expected as {new_w, new_h}
    float scale = std::min(new_shape_h * 1.0f / img_h, new_shape_w * 1.0f / img_w);
    int resize_h = static_cast<int>(round(img_h * scale));
    int resize_w = static_cast<int>(round(img_w * scale));

    int pad_h = new_shape_h - resize_h;
    int pad_w = new_shape_w - resize_w;

    cv::Mat resized_img;
    cv::resize(img, resized_img, cv::Size(resize_w, resize_h));

    float half_h = pad_h * 1.0f / 2;
    float half_w = pad_w * 1.0f / 2;

    int top = static_cast<int>(round(half_h - 0.1f));
    int bottom = static_cast<int>(round(half_h + 0.1f));
    int left = static_cast<int>(round(half_w - 0.1f));
    int right = static_cast<int>(round(half_w + 0.1f));

    transform_matrix << 1.0f / scale, 0, -half_w / scale, 0, 1.0f / scale, -half_h / scale, 0, 0, 1;

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

inline double ArmorInfer::sigmoid(double x) {
    if (x > 0)
        return 1.0 / (1.0 + std::exp(-x));
    else
        return std::exp(x) / (1.0 + std::exp(x));
}

void ArmorInfer::nms_merge_sorted_bboxes(
    std::vector<armor::ArmorObject>& faceobjects,
    std::vector<int>& indices,
    float nms_threshold
) {
    indices.clear();
    const int n = static_cast<int>(faceobjects.size());
    std::vector<float> areas(n);
    for (int i = 0; i < n; ++i)
        areas[i] = faceobjects[i].box.area();

    for (int i = 0; i < n; ++i) {
        armor::ArmorObject& a = faceobjects[i];
        int keep = 1;
        for (int indice: indices) {
            armor::ArmorObject& b = faceobjects[indice];

            // intersection over union
            cv::Rect_<float> inter = a.box & b.box;
            float inter_area = inter.area();
            float union_area = areas[i] + areas[indice] - inter_area;
            float iou = union_area > 0 ? inter_area / union_area : 0.0f;
            if (iou > nms_threshold || std::isnan(iou)) {
                keep = 0;
                // Stored for Merge (follow your original merge rules)
                if (a.number == b.number && a.color == b.color && iou > MERGE_MIN_IOU
                    && std::abs(a.prob - b.prob) < MERGE_CONF_ERROR)
                {
                    for (int k = 0; k < 4; ++k) {
                        b.pts.push_back(a.pts[k]);
                    }
                }
            }
        }
        if (keep)
            indices.push_back(i);
    }
}

std::vector<armor::ArmorObject> ArmorInfer::topKAndNms(
    std::vector<armor::ArmorObject>& output_objs,
    int top_k,
    float nms_threshold
) {
    std::sort(
        output_objs.begin(),
        output_objs.end(),
        [](const armor::ArmorObject& a, const armor::ArmorObject& b) { return a.prob > b.prob; }
    );
    if (output_objs.size() > static_cast<size_t>(top_k)) {
        output_objs.resize(top_k);
    }

    std::vector<int> indices;
    std::vector<armor::ArmorObject> objs_result;
    nms_merge_sorted_bboxes(output_objs, indices, nms_threshold);

    for (size_t idx = 0; idx < indices.size(); ++idx) {
        objs_result.push_back(std::move(output_objs[indices[idx]]));

        if (objs_result[idx].pts.size() >= 8) {
            auto n = objs_result[idx].pts.size();
            cv::Point2f pts_final[4];
            for (size_t j = 0; j < n; ++j) {
                pts_final[j % 4] += objs_result[idx].pts[j];
            }
            objs_result[idx].pts.resize(4);
            for (int j = 0; j < 4; ++j) {
                pts_final[j].x /= static_cast<float>(n) / 4.0f;
                pts_final[j].y /= static_cast<float>(n) / 4.0f;
                objs_result[idx].pts[j] = pts_final[j];
            }
        }
    }
    return objs_result;
}

std::vector<armor::ArmorObject> ArmorInfer::postProcessTUP(
    const cv::Mat& output_buffer,
    const Eigen::Matrix<float, 3, 3>& transform_matrix,
    const std::vector<GridAndStride>& grid_strides
) const {
    std::vector<armor::ArmorObject> output_objs;
    const int num_anchors = static_cast<int>(grid_strides.size());
    for (int anchor_idx = 0; anchor_idx < num_anchors && anchor_idx < output_buffer.rows;
         ++anchor_idx) {
        float confidence = output_buffer.at<float>(anchor_idx, 8);
        if (confidence < conf_threshold_)
            continue;

        const auto& gs = grid_strides[anchor_idx];
        int grid0 = gs.grid0;
        int grid1 = gs.grid1;
        int stride = gs.stride;

        cv::Mat color_scores = output_buffer.row(anchor_idx).colRange(9, 9 + TUP_NUM_COLORS);
        cv::Mat num_scores =
            output_buffer.row(anchor_idx)
                .colRange(9 + TUP_NUM_COLORS, 9 + TUP_NUM_COLORS + TUP_NUM_CLASSES);
        double color_score, num_score;
        cv::Point color_id, num_id;
        cv::minMaxLoc(color_scores, NULL, &color_score, NULL, &color_id);
        cv::minMaxLoc(num_scores, NULL, &num_score, NULL, &num_id);

        float x1 = (output_buffer.at<float>(anchor_idx, 0) + grid0) * stride;
        float y1 = (output_buffer.at<float>(anchor_idx, 1) + grid1) * stride;
        float x2 = (output_buffer.at<float>(anchor_idx, 2) + grid0) * stride;
        float y2 = (output_buffer.at<float>(anchor_idx, 3) + grid1) * stride;
        float x3 = (output_buffer.at<float>(anchor_idx, 4) + grid0) * stride;
        float y3 = (output_buffer.at<float>(anchor_idx, 5) + grid1) * stride;
        float x4 = (output_buffer.at<float>(anchor_idx, 6) + grid0) * stride;
        float y4 = (output_buffer.at<float>(anchor_idx, 7) + grid1) * stride;

        Eigen::Matrix<float, 3, 4> apex_norm;
        apex_norm << x1, x2, x3, x4, y1, y2, y3, y4, 1, 1, 1, 1;
        Eigen::Matrix<float, 3, 4> apex_dst = transform_matrix * apex_norm;

        armor::ArmorObject obj;
        obj.pts.resize(4);
        obj.pts[0] = cv::Point2f(apex_dst(0, 0), apex_dst(1, 0));
        obj.pts[1] = cv::Point2f(apex_dst(0, 1), apex_dst(1, 1));
        obj.pts[2] = cv::Point2f(apex_dst(0, 2), apex_dst(1, 2));
        obj.pts[3] = cv::Point2f(apex_dst(0, 3), apex_dst(1, 3));
        obj.box = cv::boundingRect(obj.pts);
        obj.color = static_cast<armor::ArmorColor>(color_id.x);
        obj.number = static_cast<armor::ArmorNumber>(num_id.x);
        obj.prob = confidence;

        output_objs.push_back(std::move(obj));
    }
    return topKAndNms(output_objs, top_k_, nms_threshold_);
}

std::vector<armor::ArmorObject> ArmorInfer::postProcessRP(
    const cv::Mat& output_buffer,
    const Eigen::Matrix<float, 3, 3>& transform_matrix,
    const std::vector<GridAndStride>& grid_strides
) const {
    std::vector<armor::ArmorObject> output_objs;
    for (int i = 0; i < output_buffer.rows; ++i) {
        float confidence = output_buffer.at<float>(i, 8);
        confidence = static_cast<float>(sigmoid(confidence));
        if (confidence < conf_threshold_)
            continue;

        cv::Mat color_scores = output_buffer.row(i).colRange(9, 9 + RP_NUM_COLORS);
        cv::Mat class_scores =
            output_buffer.row(i).colRange(9 + RP_NUM_COLORS, 9 + RP_NUM_COLORS + RP_NUM_CLASSES);

        double max_color_score, max_class_score;
        cv::Point max_color_id, max_class_id;
        cv::minMaxLoc(color_scores, NULL, &max_color_score, NULL, &max_color_id);
        cv::minMaxLoc(class_scores, NULL, &max_class_score, NULL, &max_class_id);

        float x1 = output_buffer.at<float>(i, 0);
        float y1 = output_buffer.at<float>(i, 1);
        float x2 = output_buffer.at<float>(i, 2);
        float y2 = output_buffer.at<float>(i, 3);
        float x3 = output_buffer.at<float>(i, 4);
        float y3 = output_buffer.at<float>(i, 5);
        float x4 = output_buffer.at<float>(i, 6);
        float y4 = output_buffer.at<float>(i, 7);

        Eigen::Matrix<float, 3, 4> pts_norm;
        pts_norm << x1, x2, x3, x4, y1, y2, y3, y4, 1, 1, 1, 1;

        Eigen::Matrix<float, 3, 4> pts_trans = transform_matrix * pts_norm;

        armor::ArmorObject obj;
        obj.pts.resize(4);
        for (int k = 0; k < 4; ++k) {
            obj.pts[k] = cv::Point2f(pts_trans(0, k), pts_trans(1, k));
        }
        obj.box = cv::boundingRect(obj.pts);

        if (max_color_id.x == 0) {
            obj.color = armor::ArmorColor::RED;
        } else if (max_color_id.x == 1) {
            obj.color = armor::ArmorColor::BLUE;
        } else {
            obj.color = armor::ArmorColor::NONE;
        }

        obj.number = static_cast<armor::ArmorNumber>(max_class_id.x);
        obj.prob = confidence;
        output_objs.push_back(std::move(obj));
    }

    return topKAndNms(output_objs, top_k_, nms_threshold_);
}

std::vector<armor::ArmorObject> ArmorInfer::postProcess(
    const cv::Mat& output_buffer,
    const Eigen::Matrix<float, 3, 3>& transform_matrix,
    const std::vector<GridAndStride>& grid_strides
) const {
    if (mode_ == Mode::TUP) {
        return postProcessTUP(output_buffer, transform_matrix, grid_strides);
    } else {
        return postProcessRP(output_buffer, transform_matrix, grid_strides);
    }
}

} // namespace armor_infer
