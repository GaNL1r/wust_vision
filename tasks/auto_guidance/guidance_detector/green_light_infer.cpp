#include "green_light_infer.hpp"

namespace auto_guidance {
GreenLightInfer::GreenLightInfer(const Params& params) {
    params_ = params;
}
std::vector<GreenLight> GreenLightInfer::postProcess(
    const cv::Mat& output_buffer,
    const Eigen::Matrix<float, 3, 3>& transform_matrix
) {
    std::vector<GreenLight> Lights;
    std::vector<float> confidences;
    std::vector<cv::Rect> boxes;

    const int num_boxes = output_buffer.rows;
    const int attr = output_buffer.cols;

    for (int i = 0; i < num_boxes; ++i) {
        float confidence = output_buffer.at<float>(i, 4);
        if (confidence < params_.conf_threshold)
            continue;

        cv::Mat class_scores = output_buffer.row(i).colRange(5, 5 + 9);
        cv::Mat color_scores = output_buffer.row(i).colRange(5 + 9, 5 + 9 + 4);

        double maxClassConfidence;
        cv::Point classIdPoint;
        cv::minMaxLoc(class_scores, nullptr, &maxClassConfidence, nullptr, &classIdPoint);
        if (maxClassConfidence < params_.conf_threshold)
            continue;

        if (classIdPoint.x != 8)
            continue;

        float cx = output_buffer.at<float>(i, 0);
        float cy = output_buffer.at<float>(i, 1);
        float w = output_buffer.at<float>(i, 2);
        float h = output_buffer.at<float>(i, 3);

        // === coordinate transform ===
        Eigen::Vector3f pt(cx, cy, 1.0f);
        Eigen::Vector3f pt_trans = transform_matrix * pt;

        float cx_t = pt_trans(0);
        float cy_t = pt_trans(1);

        // compute scale for bbox
        float scale_x = std::sqrt(transform_matrix.row(0).head<2>().squaredNorm());
        float scale_y = std::sqrt(transform_matrix.row(1).head<2>().squaredNorm());

        float w_t = w * scale_x;
        float h_t = h * scale_y;

        cv::Rect2d bbox(cx_t - w_t / 2.0f, cy_t - h_t / 2.0f, w_t, h_t);

        GreenLight light;
        light.id = classIdPoint.x;
        light.score = confidence;
        light.center_point = cv::Point2f(cx_t, cy_t);
        light.box = bbox;

        Lights.emplace_back(light);
        confidences.emplace_back(confidence);
        boxes.emplace_back(bbox);
    }

    std::vector<int> nms_result;
    cv::dnn::NMSBoxes(
        boxes,
        confidences,
        params_.conf_threshold,
        params_.nms_threshold,
        nms_result,
        0.5f,
        params_.top_k
    );
    auto IoU = [](const cv::Rect2d& a, const cv::Rect2d& b) {
        double inter = (a & b).area();
        double uni = a.area() + b.area() - inter;
        return inter / uni;
    };

    std::vector<GreenLight> final_result;
    for (int i = 0; i < nms_result.size(); i++) {
        bool keep = true;
        for (int j = 0; j < final_result.size(); j++) {
            if (IoU(final_result[j].box, Lights[nms_result[i]].box) > 0.3) {
                keep = false;
                break;
            }
        }
        if (keep)
            final_result.push_back(Lights[nms_result[i]]);
    }

    return final_result;
}

} // namespace auto_guidance