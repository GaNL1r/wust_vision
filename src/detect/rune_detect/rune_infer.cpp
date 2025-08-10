#include "detect/rune_detect/rune_infer.hpp"
namespace rune_infer {
cv::Mat
letterbox(const cv::Mat& img, Eigen::Matrix3f& transform_matrix, std::vector<int> new_shape) {
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

void generateGridsAndStride(
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
std::vector<rune::RuneObject> postProcess(
    std::vector<rune::RuneObject>& output_objs,
    const cv::Mat& output_buffer,
    const Eigen::Matrix<float, 3, 3>& transform_matrix,
    std::vector<GridAndStride> grid_strides,
    float conf_threshold,
    float nms_threshold,
    int top_k
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

        rune::RuneObject obj;

        obj.pts.r_center = cv::Point2f(apex_dst(0, 0), apex_dst(1, 0));
        obj.pts.bottom_left = cv::Point2f(apex_dst(0, 1), apex_dst(1, 1));
        obj.pts.top_left = cv::Point2f(apex_dst(0, 2), apex_dst(1, 2));
        obj.pts.top_right = cv::Point2f(apex_dst(0, 3), apex_dst(1, 3));
        obj.pts.bottom_right = cv::Point2f(apex_dst(0, 4), apex_dst(1, 4));

        auto rect = cv::boundingRect(obj.pts.toVector2f());

        obj.box = rect;
        obj.color = DNN_COLOR_TO_ENEMY_COLOR[color_id.x];
        obj.type = static_cast<rune::RuneType>(class_id.x);
        obj.prob = confidence;

        output_objs.push_back(std::move(obj));
    }
    std::sort(
        output_objs.begin(),
        output_objs.end(),
        [](const rune::RuneObject& a, const rune::RuneObject& b) { return a.prob > b.prob; }
    );
    if (output_objs.size() > static_cast<size_t>(top_k)) {
        output_objs.resize(top_k);
    }
    std::vector<int> indices;
    std::vector<rune::RuneObject> objs_result;
    nmsMergeSortedBboxes(output_objs, indices, nms_threshold);

    for (size_t i = 0; i < indices.size(); i++) {
        objs_result.push_back(std::move(output_objs[indices[i]]));

        if (objs_result[i].pts.children.size() > 0) {
            const float N = static_cast<float>(objs_result[i].pts.children.size() + 1);
            rune::FeaturePoints pts_final = std::accumulate(
                objs_result[i].pts.children.begin(),
                objs_result[i].pts.children.end(),
                objs_result[i].pts
            );
            objs_result[i].pts = pts_final / N;
        }
    }

    return objs_result;
}

void nmsMergeSortedBboxes(
    std::vector<rune::RuneObject>& faceobjects,
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
                // cout<<b.pts_x.size()<<endl;
            }
        }

        if (keep) {
            indices.push_back(i);
        }
    }
}
} // namespace rune_infer
