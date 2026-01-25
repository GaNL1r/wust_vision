#pragma once

#include "tasks/auto_aim/type.hpp"
#include "tasks/type_common.hpp"
namespace wust_vision {
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
} // namespace wust_vision