#pragma once

#include "Eigen/Dense"
#include "opencv2/opencv.hpp"
#include "type/type.hpp"
#include <algorithm>
#include <numeric>
#include <unordered_map>
#include <vector>

namespace rune_infer {
inline std::unordered_map<int, EnemyColor> DNN_COLOR_TO_ENEMY_COLOR = { { 0, EnemyColor::BLUE },
                                                                        { 1, EnemyColor::RED } };
enum class Mode { TUP };
inline Mode modeFromString(const std::string& mode) {
    if (mode == "tup" || mode == "TUP")
        return Mode::TUP;
    else
        return Mode::TUP;
}

class RuneInfer {
public:
    RuneInfer(
        Mode mode = Mode::TUP,
        float conf_threshold = 0.25f,
        float nms_threshold = 0.45f,
        int top_k = 100
    );

    // setters
    void setMode(Mode m) {
        mode_ = m;
    }
    void setConfThreshold(float t) {
        conf_threshold_ = t;
    }
    void setNmsThreshold(float t) {
        nms_threshold_ = t;
    }
    void setTopK(int k) {
        top_k_ = k;
    }
    void setUseNorm(bool v) {
        use_norm_ = v;
    }

    // getters
    int getInputW() const {
        return input_w_;
    }
    int getInputH() const {
        return input_h_;
    }
    bool getUseNorm() const {
        return use_norm_;
    }

    // utilities
    void generateGridsAndStride(
        const int target_w,
        const int target_h,
        const std::vector<int>& strides,
        std::vector<GridAndStride>& grid_strides
    );

    // letterbox that returns cv::Mat (uint8) and produces transform matrix
    cv::Mat letterbox(
        const cv::Mat& img,
        Eigen::Matrix3f& transform_matrix,
        int new_shape_w,
        int new_shape_h
    ) const;

    // faster letterbox that writes into preallocated uint8 buffer (NHWC u8)
    void letterbox_into(
        const cv::Mat& img,
        uint8_t* dst_data,
        Eigen::Matrix3f& transform_matrix,
        int dst_w,
        int dst_h
    );

    // unified postProcess: wraps the original free-function logic
    std::vector<rune::RuneObject> postProcess(
        const cv::Mat& output_buffer,
        const Eigen::Matrix<float, 3, 3>& transform_matrix,
        std::vector<GridAndStride> grid_strides
    ) const;

private:
    // helpers (mirrors your previous functions)
    void nmsMergeSortedBboxes(
        std::vector<rune::RuneObject>& faceobjects,
        std::vector<int>& indices,
        float nms_threshold
    ) const;

private:
    Mode mode_;
    int input_w_;
    int input_h_;
    bool use_norm_;
    float conf_threshold_;
    float nms_threshold_;
    int top_k_;
    // constants are reused from original file (expect these to be visible)
};

} // namespace rune_infer
