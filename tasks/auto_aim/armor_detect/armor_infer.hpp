#pragma once

#include "tasks/auto_aim/type.hpp"
#include "tasks/type_common.hpp"
namespace auto_aim {

namespace armor_infer {

    enum class Mode { TUP, RP, SPV8, AT };
    inline Mode modeFromString(const std::string& mode) {
        if (mode == "tup" || mode == "TUP")
            return Mode::TUP;
        else if (mode == "rp" || mode == "RP")
            return Mode::RP;
        else if (mode == "spv8" || mode == "SPV8")
            return Mode::SPV8;
        else if (mode == "at" || mode == "AT")
            return Mode::AT;
        else
            return Mode::TUP;
    }

    class ArmorInfer {
    public:
        ArmorInfer(
            Mode mode = Mode::TUP,
            float conf_threshold = 0.25f,
            float nms_threshold = 0.45f,
            int top_k = 100
        );
        // static cv::Mat letterbox(
        //     const cv::Mat& img,
        //     Eigen::Matrix3f& transform_matrix,
        //     const int new_shape_w,
        //     const int new_shape_h
        // ) {
        //     const int img_h = img.rows;
        //     const int img_w = img.cols;

        //     // scale (keep aspect ratio)
        //     const float scale = std::min(new_shape_h * 1.0f / img_h, new_shape_w * 1.0f / img_w);

        //     const int resize_h = static_cast<int>(img_h * scale + 0.5f);
        //     const int resize_w = static_cast<int>(img_w * scale + 0.5f);

        //     const int pad_h = new_shape_h - resize_h;
        //     const int pad_w = new_shape_w - resize_w;

        //     // YOLO-style symmetric padding
        //     const float half_h = pad_h * 0.5f;
        //     const float half_w = pad_w * 0.5f;

        //     const int top = static_cast<int>(half_h - 0.1f);
        //     const int left = static_cast<int>(half_w - 0.1f);

        //     // Allocate output once, fill with padding color
        //     cv::Mat out(new_shape_h, new_shape_w, img.type(), cv::Scalar(114, 114, 114));

        //     // ROI where resized image will be placed
        //     cv::Rect roi(left, top, resize_w, resize_h);
        //     cv::Mat out_roi = out(roi);

        //     // Resize directly into ROI
        //     cv::resize(img, out_roi, out_roi.size(), 0, 0, cv::INTER_LINEAR);

        //     // Transform matrix: letterbox -> original image
        //     transform_matrix << 1.0f / scale, 0.0f, -half_w / scale, 0.0f, 1.0f / scale,
        //         -half_h / scale, 0.0f, 0.0f, 1.0f;

        //     return out;
        // }

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
        int getInputW() {
            return input_w_;
        }
        int getInputH() {
            return input_h_;
        }
        bool getUseNorm() {
            return use_norm_;
        }

        void generate_grids_and_stride(
            const int target_w,
            const int target_h,
            const std::vector<int>& strides,
            std::vector<GridAndStride>& grid_strides
        );

        // Main unified interface: run post-processing according to current mode
        // output_buffer: rows = anchors / proposals, cols = model outputs (float)
        std::vector<ArmorObject> postProcess(
            const cv::Mat& output_buffer,
            const Eigen::Matrix<float, 3, 3>& transform_matrix,
            const std::vector<GridAndStride>& grid_strides
        ) const;

    private:
        // internal mode-specific processors
        std::vector<ArmorObject> postProcessTUP(
            const cv::Mat& output_buffer,
            const Eigen::Matrix<float, 3, 3>& transform_matrix,
            const std::vector<GridAndStride>& grid_strides
        ) const;

        std::vector<ArmorObject> postProcessRP(
            const cv::Mat& output_buffer,
            const Eigen::Matrix<float, 3, 3>& transform_matrix,
            const std::vector<GridAndStride>& grid_strides
        ) const;
        std::vector<ArmorObject> postProcessSPV8(
            const cv::Mat& output_buffer,
            const Eigen::Matrix<float, 3, 3>& transform_matrix,
            const std::vector<GridAndStride>& grid_strides
        ) const;
        std::vector<ArmorObject> postProcessAT(
            const cv::Mat& output_buffer,
            const Eigen::Matrix<float, 3, 3>& transform_matrix,
            const std::vector<GridAndStride>& grid_strides
        ) const;
        // internal helpers
        static void nms_merge_sorted_bboxes(
            std::vector<ArmorObject>& faceobjects,
            std::vector<int>& indices,
            float nms_threshold
        );

        static std::vector<ArmorObject>
        topKAndNms(std::vector<ArmorObject>& objs, int top_k, float nms_threshold);

        static inline double sigmoid(double x);

    private:
        Mode mode_;
        int input_w_;
        int input_h_;
        float conf_threshold_;
        float nms_threshold_;
        int top_k_;
        bool use_norm_;
    };

} // namespace armor_infer
} // namespace auto_aim