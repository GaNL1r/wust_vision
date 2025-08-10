#pragma once

#include "Eigen/Dense"
#include "opencv2/opencv.hpp"
#include "type/type.hpp"

namespace rune_infer {
constexpr int INPUT_W = 480; // Width of input
constexpr int INPUT_H = 480; // Height of input
constexpr int NUM_CLASSES = 2; // Number of classes
constexpr int NUM_COLORS = 2; // Number of color
constexpr int NUM_POINTS = 5;
constexpr int NUM_POINTS_2 = 2 * NUM_POINTS;
constexpr float MERGE_CONF_ERROR = 0.15;
constexpr float MERGE_MIN_IOU = 0.9;
inline std::unordered_map<int, EnemyColor> DNN_COLOR_TO_ENEMY_COLOR = { { 0, EnemyColor::BLUE },
                                                                        { 1, EnemyColor::RED } };
cv::Mat letterbox(
    const cv::Mat& img,
    Eigen::Matrix3f& transform_matrix,
    std::vector<int> new_shape = { INPUT_W, INPUT_H }
);
void generateGridsAndStride(
    const int target_w,
    const int target_h,
    std::vector<int>& strides,
    std::vector<GridAndStride>& grid_strides
);
std::vector<rune::RuneObject> postProcess(
    std::vector<rune::RuneObject>& output_objs,
    const cv::Mat& output_buffer,
    const Eigen::Matrix<float, 3, 3>& transform_matrix,
    std::vector<GridAndStride> grid_strides,
    float conf_threshold,
    float nms_threshold,
    int top_k
);
inline float intersectionArea(const rune::RuneObject& a, const rune::RuneObject& b) {
    cv::Rect_<float> inter = a.box & b.box;
    return inter.area();
}
void nmsMergeSortedBboxes(
    std::vector<rune::RuneObject>& faceobjects,
    std::vector<int>& indices,
    float nms_threshold
);
} // namespace rune_infer