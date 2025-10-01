#include "rune_infer.hpp"
namespace rune_infer {
static constexpr int V11_INPUT_W = 416;
static constexpr int V11_INPUT_H = 416;
static constexpr int V11_NUM_CLASSES = 1;
static constexpr int V11_NUM_POINTS = 6;
static constexpr float MERGE_CONF_ERROR = 0.15f;
static constexpr float MERGE_MIN_IOU = 0.9f;
RuneInfer::RuneInfer(Mode mode, float conf_threshold, float nms_threshold, int top_k):
    mode_(mode),
    conf_threshold_(conf_threshold),
    nms_threshold_(nms_threshold),
    top_k_(top_k) {
    if (mode_ == Mode::V11) {
        input_h_ = V11_INPUT_H;
        input_w_ = V11_INPUT_W;
        use_norm_ = true;
    } else {
        input_h_ = V11_INPUT_H;
        input_w_ = V11_INPUT_W;
        use_norm_ = true;
    }
}

// generate grids and stride
void RuneInfer::generateGridsAndStride(
    const int target_w,
    const int target_h,
    const std::vector<int>& strides,
    std::vector<GridAndStride>& grid_strides
) {
    grid_strides.clear();
    for (auto stride: strides) {
        int num_grid_w = target_w / stride;
        int num_grid_h = target_h / stride;
        for (int g1 = 0; g1 < num_grid_h; ++g1) {
            for (int g0 = 0; g0 < num_grid_w; ++g0) {
                grid_strides.emplace_back(GridAndStride { g0, g1, stride });
            }
        }
    }
}

// letterbox -> returns resized+padded image (uint8 CV_8UC3) and fill transform_matrix
cv::Mat RuneInfer::letterbox(
    const cv::Mat& img,
    Eigen::Matrix3f& transform_matrix,
    int new_shape_w,
    int new_shape_h
) const {
    CV_Assert(!img.empty());
    int img_h = img.rows;
    int img_w = img.cols;

    float scale = std::min(new_shape_h * 1.0f / img_h, new_shape_w * 1.0f / img_w);
    int resize_h = static_cast<int>(round(img_h * scale));
    int resize_w = static_cast<int>(round(img_w * scale));

    int pad_h = new_shape_h - resize_h;
    int pad_w = new_shape_w - resize_w;

    cv::Mat resized_img;
    cv::resize(img, resized_img, cv::Size(resize_w, resize_h));

    float half_h = pad_h * 1.0f / 2.0f;
    float half_w = pad_w * 1.0f / 2.0f;

    int top = static_cast<int>(round(half_h));
    int bottom = static_cast<int>(round(half_h));
    int left = static_cast<int>(round(half_w));
    int right = static_cast<int>(round(half_w));

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

// letterbox_into: write final NHWC u8 into dst_data
void RuneInfer::letterbox_into(
    const cv::Mat& img,
    uint8_t* dst_data,
    Eigen::Matrix3f& transform_matrix,
    int dst_w,
    int dst_h
) {
    CV_Assert(!img.empty());
    CV_Assert(img.type() == CV_8UC3);
    int img_h = img.rows;
    int img_w = img.cols;

    float scale = std::min(dst_h * 1.0f / img_h, dst_w * 1.0f / img_w);
    int resize_h = std::max(1, static_cast<int>(round(img_h * scale)));
    int resize_w = std::max(1, static_cast<int>(round(img_w * scale)));

    int pad_h = dst_h - resize_h;
    int pad_w = dst_w - resize_w;

    float half_h = pad_h / 2.0f;
    float half_w = pad_w / 2.0f;

    int top = static_cast<int>(round(half_h - 0.1f));
    int left = static_cast<int>(round(half_w - 0.1f));

    transform_matrix << 1.0f / scale, 0, -half_w / scale, 0, 1.0f / scale, -half_h / scale, 0, 0, 1;

    cv::Mat resized_img;
    cv::resize(img, resized_img, cv::Size(resize_w, resize_h), 0, 0, cv::INTER_LINEAR);

    cv::Mat final_img(dst_h, dst_w, CV_8UC3, cv::Scalar(114, 114, 114));
    resized_img.copyTo(final_img(cv::Rect(left, top, resize_w, resize_h)));

    // memory layout: NHWC contiguous
    size_t total = static_cast<size_t>(dst_h) * dst_w * 3;
    std::memcpy(dst_data, final_img.data, total);
}

float polygonArea(const std::vector<cv::Point2f>& poly) {
    if (poly.size() < 3)
        return 0.0f;
    return std::fabs(cv::contourArea(poly));
}

inline float intersectionArea(const rune::RuneObject& a, const rune::RuneObject& b) {
    cv::Rect_<float> inter = a.box & b.box;
    return inter.area();
}
void RuneInfer::nmsMergeSortedBboxes(
    std::vector<rune::RuneObject>& faceobjects,
    std::vector<int>& indices,
    float nms_threshold
) const {
    indices.clear();

    const int n = faceobjects.size();

    std::vector<float> areas(n);
    for (int i = 0; i < n; i++) {
        areas[i] = faceobjects[i].box.area();
    }

    for (int i = 0; i < n; i++) {
        rune::RuneObject& a = faceobjects[i];

        int keep = 1;
        for (size_t j = 0; j < indices.size(); j++) {
            rune::RuneObject& b = faceobjects[indices[j]];

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
            }
        }

        if (keep) {
            indices.push_back(i);
        }
    }
}
// std::vector<rune::RuneObject> RuneInfer::postProcessTup(
//     const cv::Mat& output_buffer,
//     const Eigen::Matrix<float, 3, 3>& transform_matrix,
//     std::vector<GridAndStride> grid_strides
// ) const {
//     std::vector<rune::RuneObject> output_objs;
//     const int num_anchors = grid_strides.size();

//     for (int anchor_idx = 0; anchor_idx < num_anchors; anchor_idx++) {
//         float confidence = output_buffer.at<float>(anchor_idx, TUP_NUM_POINTS_2);
//         if (confidence < conf_threshold_) {
//             continue;
//         }

//         const int grid0 = grid_strides[anchor_idx].grid0;
//         const int grid1 = grid_strides[anchor_idx].grid1;
//         const int stride = grid_strides[anchor_idx].stride;

//         double color_score, class_score;
//         cv::Point color_id, class_id;
//         cv::Mat color_scores =
//             output_buffer.row(anchor_idx)
//                 .colRange(TUP_NUM_POINTS_2 + 1, TUP_NUM_POINTS_2 + 1 + TUP_NUM_COLORS);
//         cv::Mat num_scores = output_buffer.row(anchor_idx)
//                                  .colRange(
//                                      TUP_NUM_POINTS_2 + 1 + TUP_NUM_COLORS,
//                                      TUP_NUM_POINTS_2 + 1 + TUP_NUM_COLORS + TUP_NUM_CLASSES
//                                  );
//         // Argmax
//         cv::minMaxLoc(color_scores, NULL, &color_score, NULL, &color_id);
//         cv::minMaxLoc(num_scores, NULL, &class_score, NULL, &class_id);

//         float x_1 = (output_buffer.at<float>(anchor_idx, 0) + grid0) * stride;
//         float y_1 = (output_buffer.at<float>(anchor_idx, 1) + grid1) * stride;
//         float x_2 = (output_buffer.at<float>(anchor_idx, 2) + grid0) * stride;
//         float y_2 = (output_buffer.at<float>(anchor_idx, 3) + grid1) * stride;
//         float x_3 = (output_buffer.at<float>(anchor_idx, 4) + grid0) * stride;
//         float y_3 = (output_buffer.at<float>(anchor_idx, 5) + grid1) * stride;
//         float x_4 = (output_buffer.at<float>(anchor_idx, 6) + grid0) * stride;
//         float y_4 = (output_buffer.at<float>(anchor_idx, 7) + grid1) * stride;
//         float x_5 = (output_buffer.at<float>(anchor_idx, 8) + grid0) * stride;
//         float y_5 = (output_buffer.at<float>(anchor_idx, 9) + grid1) * stride;

//         Eigen::Matrix<float, 3, 5> apex_norm;
//         Eigen::Matrix<float, 3, 5> apex_dst;

//         /* clang-format off */
//         /* *INDENT-OFF* */
//         apex_norm << x_1, x_2, x_3, x_4, x_5,
//                     y_1, y_2, y_3, y_4, y_5,
//                     1,   1,   1,   1,   1;
//         /* *INDENT-ON* */
//         /* clang-format on */

//         apex_dst = transform_matrix * apex_norm;

//         rune::RuneObject obj;

//         obj.pts.r_center = cv::Point2f(apex_dst(0, 0), apex_dst(1, 0));
//         obj.pts.bottom_left = cv::Point2f(apex_dst(0, 1), apex_dst(1, 1));
//         obj.pts.top_left = cv::Point2f(apex_dst(0, 2), apex_dst(1, 2));
//         obj.pts.top_right = cv::Point2f(apex_dst(0, 3), apex_dst(1, 3));
//         obj.pts.bottom_right = cv::Point2f(apex_dst(0, 4), apex_dst(1, 4));

//         auto rect = cv::boundingRect(obj.pts.toVector2f());

//         obj.box = rect;
//         obj.color = DNN_COLOR_TO_ENEMY_COLOR[color_id.x];
//         obj.type = static_cast<rune::RuneType>(class_id.x);
//         obj.prob = confidence;

//         output_objs.push_back(std::move(obj));
//     }
//     std::sort(
//         output_objs.begin(),
//         output_objs.end(),
//         [](const rune::RuneObject& a, const rune::RuneObject& b) { return a.prob > b.prob; }
//     );
//     if (output_objs.size() > static_cast<size_t>(top_k_)) {
//         output_objs.resize(top_k_);
//     }
//     std::vector<int> indices;
//     std::vector<rune::RuneObject> objs_result;
//     nmsMergeSortedBboxes(output_objs, indices, nms_threshold_);

//     for (size_t i = 0; i < indices.size(); i++) {
//         objs_result.push_back(std::move(output_objs[indices[i]]));

//         if (objs_result[i].pts.children.size() > 0) {
//             const float N = static_cast<float>(objs_result[i].pts.children.size() + 1);
//             rune::FeaturePoints pts_final = std::accumulate(
//                 objs_result[i].pts.children.begin(),
//                 objs_result[i].pts.children.end(),
//                 objs_result[i].pts
//             );
//             objs_result[i].pts = pts_final / N;
//         }
//     }
//     return objs_result;
// }
std::vector<rune::RuneObject> RuneInfer::postProcess(
    const cv::Mat& output_buffer,
    const Eigen::Matrix<float, 3, 3>& transform_matrix,
    std::vector<GridAndStride> grid_strides
) const {
    switch (mode_) {
        case Mode::V11:
            return postProcessV11(output_buffer, transform_matrix, grid_strides);
        // case Mode::TUP:
        //     return postProcessTup(output_buffer, transform_matrix, grid_strides);
        default:
            return {};
    }
}
std::vector<rune::RuneObject> RuneInfer::postProcessV11(
    const cv::Mat& output_buffer,
    const Eigen::Matrix<float, 3, 3>& transform_matrix,
    const std::vector<GridAndStride>& grid_strides
) const {
    std::vector<rune::RuneObject> output_objs;

    const int num_channels = output_buffer.rows; // 17
    const int num_preds = output_buffer.cols; // 3549

    const int box_offset = 0;
    const int conf_offset = 4;
    const int kpt_offset = 5;
    const int num_kpts = V11_NUM_POINTS;
    const int kpt_step = 2;

    for (int i = 0; i < num_preds; ++i) {
        float score = output_buffer.at<float>(conf_offset, i);

        if (score < conf_threshold_)
            continue;
        rune::RuneObject obj;
        float x_center = (output_buffer.at<float>(0, i));
        float y_center = (output_buffer.at<float>(1, i));
        float w = output_buffer.at<float>(2, i);
        float h = output_buffer.at<float>(3, i);

        float x0 = x_center - w * 0.5f;
        float y0 = y_center - h * 0.5f;
        float x1 = x_center + w * 0.5f;
        float y1 = y_center + h * 0.5f;
        Eigen::Matrix<float, 3, 4> box_corners;
        box_corners << x0, x1, x1, x0, y0, y0, y1, y1, 1.0, 1.0, 1.0, 1.0;

        // 3. 应用 transform_matrix
        Eigen::Matrix<float, 3, 4> box_dst = transform_matrix * box_corners;

        // 4. 转换成 OpenCV 点
        std::vector<cv::Point2f> box_points(4);
        for (int k = 0; k < 4; ++k) {
            box_points[k] = cv::Point2f(box_dst(0, k), box_dst(1, k));
        }

        // 5. 如果需要 OpenCV Rect，可以取外接矩形
        cv::Rect rect = cv::boundingRect(box_points);

        // 保存到 obj
        obj.box = rect;
        obj.prob = score;
        const int grid0 = grid_strides[i].grid0;
        const int grid1 = grid_strides[i].grid1;
        const int stride = grid_strides[i].stride;

        Eigen::Matrix<float, 3, num_kpts> apex_norm;
        bool valid = false;
        for (int k = 0; k < num_kpts; ++k) {
            float kx = output_buffer.at<float>(kpt_offset + k * kpt_step + 0, i);
            float ky = output_buffer.at<float>(kpt_offset + k * kpt_step + 1, i);

            kx = std::max(0.0f, kx);
            ky = std::max(0.0f, ky);

            apex_norm(0, k) = kx;
            apex_norm(1, k) = ky;
            apex_norm(2, k) = 1.0;
        }

        Eigen::Matrix<float, 3, num_kpts> apex_dst = transform_matrix * apex_norm;
        obj.pts.r_tag = cv::Point2f(apex_dst(0, 0), apex_dst(1, 0));
        obj.pts.fan_center = cv::Point2f(apex_dst(0, 1), apex_dst(1, 1));
        obj.pts.fan_top = cv::Point2f(apex_dst(0, 2), apex_dst(1, 2));
        obj.pts.fan_right = cv::Point2f(apex_dst(0, 3), apex_dst(1, 3));
        obj.pts.fan_bottom = cv::Point2f(apex_dst(0, 4), apex_dst(1, 4));
        obj.pts.fan_left = cv::Point2f(apex_dst(0, 5), apex_dst(1, 5));
        obj.type = rune::RuneType::INACTIVATED;
        output_objs.push_back(std::move(obj));
    }

    std::sort(
        output_objs.begin(),
        output_objs.end(),
        [](const rune::RuneObject& a, const rune::RuneObject& b) { return a.prob > b.prob; }
    );

    if (output_objs.size() > static_cast<size_t>(top_k_))
        output_objs.resize(top_k_);

    std::vector<int> indices;
    std::vector<rune::RuneObject> objs_result;
    nmsMergeSortedBboxes(output_objs, indices, nms_threshold_);

    for (int idx: indices)
        objs_result.push_back(std::move(output_objs[idx]));

    return objs_result;
}

} // namespace rune_infer
