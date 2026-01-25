#include "tasks/auto_aim/armor_detect/armor_infer.hpp"
#include <algorithm>
#include <cmath>
namespace auto_aim {

namespace armor_infer {

    static const int RP_INPUT_W = 640;
    static const int RP_INPUT_H = 640;
    static constexpr int RP_NUM_CLASSES = 9;
    static constexpr int RP_NUM_COLORS = 4;
    static const int TUP_INPUT_W = 416;
    static const int TUP_INPUT_H = 416;
    static constexpr int TUP_NUM_CLASSES = 8;
    static constexpr int TUP_NUM_COLORS = 4;
    static const int SPV8_INPUT_W = 416;
    static const int SPV8_INPUT_H = 416;
    static const int SPV8_NUM_CLASSES = 8;
    static const int SPV8_NUM_COLORS = 2;
    static constexpr float MERGE_CONF_ERROR = 0.15f;
    static constexpr float MERGE_MIN_IOU = 0.9f;

    static constexpr int AT_INPUT_W = 640;
    static constexpr int AT_INPUT_H = 640;
    static constexpr int AT_NUM_COLORS = 4;
    static constexpr int AT_NUM_CLASSES = 13;
    static constexpr int AT_NUM_KPTS = 4;
    ArmorInfer::ArmorInfer(Mode mode, float conf_threshold, float nms_threshold, int top_k):
        mode_(mode),
        conf_threshold_(conf_threshold),
        nms_threshold_(nms_threshold),
        top_k_(top_k) {
        if (mode_ == Mode::TUP) {
            input_w_ = TUP_INPUT_W;
            input_h_ = TUP_INPUT_H;
            use_norm_ = false;
        } else if (mode_ == Mode::RP) {
            input_w_ = RP_INPUT_W;
            input_h_ = RP_INPUT_H;
            use_norm_ = true;
        } else if (mode_ == Mode::SPV8) {
            input_w_ = SPV8_INPUT_W;
            input_h_ = SPV8_INPUT_H;
            use_norm_ = true;
        } else if (mode == Mode::AT) {
            input_w_ = AT_INPUT_W;
            input_h_ = AT_INPUT_H;
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

    inline double ArmorInfer::sigmoid(double x) {
        if (x > 0)
            return 1.0 / (1.0 + std::exp(-x));
        else
            return std::exp(x) / (1.0 + std::exp(x));
    }

    void ArmorInfer::nms_merge_sorted_bboxes(
        std::vector<ArmorObject>& faceobjects,
        std::vector<int>& indices,
        float nms_threshold
    ) {
        indices.clear();
        const int n = static_cast<int>(faceobjects.size());
        std::vector<float> areas(n);
        for (int i = 0; i < n; ++i)
            areas[i] = faceobjects[i].box.area();

        for (int i = 0; i < n; ++i) {
            ArmorObject& a = faceobjects[i];
            int keep = 1;
            for (int indice: indices) {
                ArmorObject& b = faceobjects[indice];

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

    std::vector<ArmorObject>
    ArmorInfer::topKAndNms(std::vector<ArmorObject>& output_objs, int top_k, float nms_threshold) {
        std::sort(
            output_objs.begin(),
            output_objs.end(),
            [](const ArmorObject& a, const ArmorObject& b) { return a.prob > b.prob; }
        );
        if (output_objs.size() > static_cast<size_t>(top_k)) {
            output_objs.resize(top_k);
        }

        std::vector<int> indices;
        std::vector<ArmorObject> objs_result;
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

    std::vector<ArmorObject> ArmorInfer::postProcessTUP(
        const cv::Mat& output_buffer,
        const Eigen::Matrix<float, 3, 3>& transform_matrix,
        const std::vector<GridAndStride>& grid_strides
    ) const {
        std::vector<ArmorObject> output_objs;
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
            Eigen::Matrix<float, 3, 4> apex_dst = apex_norm;
            // Eigen::Matrix<float, 3, 4> apex_dst =transform_matrix* apex_norm;
            ArmorObject obj;
            obj.pts.resize(4);
            obj.pts[0] = cv::Point2f(apex_dst(0, 0), apex_dst(1, 0));
            obj.pts[1] = cv::Point2f(apex_dst(0, 1), apex_dst(1, 1));
            obj.pts[2] = cv::Point2f(apex_dst(0, 2), apex_dst(1, 2));
            obj.pts[3] = cv::Point2f(apex_dst(0, 3), apex_dst(1, 3));
            obj.box = cv::boundingRect(obj.pts);
            obj.color = static_cast<ArmorColor>(color_id.x);
            obj.number = static_cast<ArmorNumber>(num_id.x);
            obj.prob = confidence;

            output_objs.push_back(std::move(obj));
        }
        return topKAndNms(output_objs, top_k_, nms_threshold_);
    }

    std::vector<ArmorObject> ArmorInfer::postProcessRP(
        const cv::Mat& output_buffer,
        const Eigen::Matrix<float, 3, 3>& transform_matrix,
        const std::vector<GridAndStride>& grid_strides
    ) const {
        std::vector<ArmorObject> output_objs;
        for (int i = 0; i < output_buffer.rows; ++i) {
            float confidence = output_buffer.at<float>(i, 8);
            confidence = static_cast<float>(sigmoid(confidence));
            if (confidence < conf_threshold_)
                continue;

            cv::Mat color_scores = output_buffer.row(i).colRange(9, 9 + RP_NUM_COLORS);
            cv::Mat class_scores = output_buffer.row(i).colRange(
                9 + RP_NUM_COLORS,
                9 + RP_NUM_COLORS + RP_NUM_CLASSES
            );

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

            // Eigen::Matrix<float, 3, 4> pts_trans = transform_matrix * pts_norm;
            Eigen::Matrix<float, 3, 4> pts_trans = pts_norm;
            ArmorObject obj;
            obj.pts.resize(4);
            for (int k = 0; k < 4; ++k) {
                obj.pts[k] = cv::Point2f(pts_trans(0, k), pts_trans(1, k));
            }
            obj.box = cv::boundingRect(obj.pts);

            if (max_color_id.x == 0) {
                obj.color = ArmorColor::RED;
            } else if (max_color_id.x == 1) {
                obj.color = ArmorColor::BLUE;
            } else {
                obj.color = ArmorColor::NONE;
            }

            obj.number = static_cast<ArmorNumber>(max_class_id.x);
            obj.prob = confidence;
            output_objs.push_back(std::move(obj));
        }

        return topKAndNms(output_objs, top_k_, nms_threshold_);
    }
    std::vector<ArmorObject> ArmorInfer::postProcessSPV8(
        const cv::Mat& output_buffer,
        const Eigen::Matrix<float, 3, 3>& transform_matrix,
        const std::vector<GridAndStride>& grid_strides
    ) const {
        std::vector<ArmorObject> output_objs;

        for (int r = 0; r < output_buffer.rows; r++) {
            // bbox xywh 或四个角点
            auto xywh = output_buffer.row(r).colRange(0, 4);

            // 类别 scores
            auto scores = output_buffer.row(r).colRange(4, 4 + SPV8_NUM_COLORS);

            // 关键点 4*2
            auto key_points =
                output_buffer.row(r).colRange(4 + SPV8_NUM_COLORS, 12 + SPV8_NUM_COLORS);

            double max_score;
            cv::Point max_idx;
            cv::minMaxLoc(scores, nullptr, &max_score, nullptr, &max_idx);
            if (max_score < conf_threshold_)
                continue;

            // bbox 左上角 + 宽高
            float cx = xywh.at<float>(0);
            float cy = xywh.at<float>(1);
            float w = xywh.at<float>(2);
            float h = xywh.at<float>(3);

            float x1 = key_points.at<float>(0, 0 * 2 + 0);
            float y1 = key_points.at<float>(0, 0 * 2 + 1);
            float x2 = key_points.at<float>(0, 1 * 2 + 0);
            float y2 = key_points.at<float>(0, 1 * 2 + 1);
            float x3 = key_points.at<float>(0, 2 * 2 + 0);
            float y3 = key_points.at<float>(0, 2 * 2 + 1);
            float x4 = key_points.at<float>(0, 3 * 2 + 0);
            float y4 = key_points.at<float>(0, 3 * 2 + 1);

            Eigen::Matrix<float, 3, 4> pts_norm;
            pts_norm << x1, x2, x3, x4, y1, y2, y3, y4, 1, 1, 1, 1;

            // Eigen::Matrix<float, 3, 4> pts_trans = transform_matrix * pts_norm;
            Eigen::Matrix<float, 3, 4> pts_trans = pts_norm;
            ArmorObject obj;
            obj.pts.resize(4);
            for (int i = 0; i < 4; i++) {
                obj.pts[i] = cv::Point2f(pts_trans(0, i), pts_trans(1, i));
            }
            obj.box = cv::boundingRect(obj.pts);
            obj.color = ArmorColor::RED;
            obj.number = static_cast<ArmorNumber>(max_idx.x);
            obj.prob = max_score;

            output_objs.emplace_back(std::move(obj));
        }

        return topKAndNms(output_objs, top_k_, nms_threshold_);
    }
    std::vector<ArmorObject> ArmorInfer::postProcessAT(
        const cv::Mat& output_buffer,
        const Eigen::Matrix<float, 3, 3>& transform_matrix,
        const std::vector<GridAndStride>& grid_strides
    ) const {
        std::vector<ArmorObject> output_objs;

        if (output_buffer.empty() || output_buffer.rows < 4) {
            return output_objs;
        }

        // 正确：dims = 行，每条预测维度；anchors = 列，anchor 数量
        const int dims = output_buffer.rows;
        const int anchors = output_buffer.cols;

        constexpr int nkpt = 4; // 四点装甲板
        constexpr int kpt_dim = 2;
        constexpr int nk = nkpt * kpt_dim; // 8

        const int num_cls = dims - 4 - nk;
        if (num_cls <= 0) {
            std::cerr << "Invalid dims=" << dims << " (nk=" << nk << "), num_cls<=0\n";
            return output_objs;
        }

        auto map_point = [&](float x, float y) -> cv::Point2f {
            Eigen::Vector3f pt(x, y, 1.f);
            // Eigen::Vector3f tr = transform_matrix * pt;
            Eigen::Vector3f tr = pt;
            return { tr(0), tr(1) };
        };

        // 每个 anchor 对应一列
        for (int a = 0; a < anchors; ++a) {
            // 使用 (row, col) 访问，第 col=a 是当前 anchor
            float cx = output_buffer.at<float>(0, a);
            float cy = output_buffer.at<float>(1, a);
            float w = output_buffer.at<float>(2, a);
            float h = output_buffer.at<float>(3, a);

            if (!(std::isfinite(cx) && std::isfinite(cy) && std::isfinite(w) && std::isfinite(h)))
                continue;
            if (w <= 0 || h <= 0)
                continue;

            // 取得最大类别
            float best_score = 0.f;
            int best_idx = -1;
            for (int c = 0; c < num_cls; ++c) {
                float sc = output_buffer.at<float>(4 + c, a);
                if (sc > best_score) {
                    best_score = sc;
                    best_idx = c;
                }
            }
            if (best_idx < 0 || best_score < conf_threshold_)
                continue;

            // 映射成 color + id
            int combined = best_idx;
            int color = combined / AT_NUM_CLASSES;
            int cls = combined % AT_NUM_CLASSES;

            ArmorObject obj {};
            obj.number = ArmorNumber::NO3;
            obj.color = ArmorColor::BLUE;
            obj.prob = best_score;

            // 关键点起始位置
            const int kp_base = 4 + num_cls;

            obj.pts.reserve(nkpt);
            for (int k = 0; k < nkpt; ++k) {
                float kx = output_buffer.at<float>(kp_base + 2 * k, a);
                float ky = output_buffer.at<float>(kp_base + 2 * k + 1, a);
                obj.pts.push_back(map_point(kx, ky));
            }

            obj.box = cv::boundingRect(obj.pts);

            output_objs.emplace_back(std::move(obj));
        }

        return topKAndNms(output_objs, top_k_, nms_threshold_);
    }

    std::vector<ArmorObject> ArmorInfer::postProcess(
        const cv::Mat& output_buffer,
        const Eigen::Matrix<float, 3, 3>& transform_matrix,
        const std::vector<GridAndStride>& grid_strides
    ) const {
        if (mode_ == Mode::TUP) {
            return postProcessTUP(output_buffer, transform_matrix, grid_strides);
        } else if (mode_ == Mode::SPV8) {
            return postProcessSPV8(output_buffer, transform_matrix, grid_strides);
        } else if (mode_ == Mode::AT) {
            return postProcessAT(output_buffer, transform_matrix, grid_strides);
        } else if (mode_ == Mode::RP) {
            return postProcessRP(output_buffer, transform_matrix, grid_strides);
        }
    }

} // namespace armor_infer
} // namespace auto_aim