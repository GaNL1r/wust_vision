#pragma once
#include "tasks/auto_guidance/type.hpp"
namespace auto_guidance {
class GreenLightInfer {
public:
    using GreenLightInferPtr = std::unique_ptr<GreenLightInfer>;
    struct Params {
        int input_w;
        int input_h;
        float conf_threshold;
        float nms_threshold;
        int top_k;
        bool use_norm;
    } params_;
    GreenLightInfer(const Params& params);
    static inline GreenLightInferPtr makeGreenLightInfer(const Params& params) {
        return std::make_unique<GreenLightInfer>(params);
    }
    cv::Mat letterbox(
        const cv::Mat& img,
        Eigen::Matrix3f& transform_matrix,
        const int new_shape_w,
        const int new_shape_h
    );
    std::vector<GreenLight>
    postProcess(const cv::Mat& output_buffer, const Eigen::Matrix<float, 3, 3>& transform_matrix);
    int getInputW() {
        return params_.input_w;
    }
    int getInputH() {
        return params_.input_h;
    }
    bool getUseNorm() {
        return params_.use_norm;
    }
};
} // namespace auto_guidance
