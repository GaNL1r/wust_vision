#pragma once

#include "Eigen/Dense"
#include "opencv2/opencv.hpp"
#include "type/type.hpp"
namespace armor_infer {
const int INPUT_W = 416; // Width of input
const int INPUT_H = 416; // Height of input
constexpr int NUM_CLASSES = 8; // Number of classes
constexpr int NUM_COLORS = 4; // Number of color
constexpr float MERGE_CONF_ERROR = 0.15;
constexpr float MERGE_MIN_IOU = 0.9;
void generate_grids_and_stride(
    const int target_w,
    const int target_h,
    std::vector<int>& strides,
    std::vector<GridAndStride>& grid_strides
);
cv::Mat letterbox(
    const cv::Mat& img,
    Eigen::Matrix3f& transform_matrix,
    std::vector<int> new_shape = { INPUT_W, INPUT_H }
);
inline float intersection_area(const armor::ArmorObject& a, const armor::ArmorObject& b) {
    cv::Rect_<float> inter = a.box & b.box;
    return inter.area();
}
void nms_merge_sorted_bboxes(
    std::vector<armor::ArmorObject>& faceobjects,
    std::vector<int>& indices,
    float nms_threshold
);
std::vector<armor::ArmorObject> postProcess(
    std::vector<armor::ArmorObject>& output_objs,
    const cv::Mat& output_buffer,
    const Eigen::Matrix<float, 3, 3>& transform_matrix,
    std::vector<GridAndStride> grid_strides,
    float conf_threshold,
    float nms_threshold,
    int top_k
);
} // namespace armor_infer
