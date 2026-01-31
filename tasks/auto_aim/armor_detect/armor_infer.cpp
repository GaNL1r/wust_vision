#include "armor_infer.hpp"
namespace wust_vision::auto_aim::armor_infer {
[[nodiscard]] static inline std::vector<GridAndStride>
generate_grids_and_stride(int target_w, int target_h, const std::vector<int>& strides) noexcept {
    std::vector<GridAndStride> grid_strides;
    for (int stride: strides) {
        const int num_w = target_w / stride;
        const int num_h = target_h / stride;
        grid_strides.reserve(grid_strides.size() + num_w * num_h);
        for (int gy = 0; gy < num_h; ++gy) {
            for (int gx = 0; gx < num_w; ++gx) {
                grid_strides.push_back(GridAndStride { gx, gy, stride });
            }
        }
    }
    return grid_strides;
}
std::vector<ArmorObject> ArmorInfer::postProcessTUP_impl(const cv::Mat& out) const {
    static std::optional<std::vector<GridAndStride>> _grid_strides;
    if (!_grid_strides) {
        _grid_strides = generate_grids_and_stride(inputW(), inputH(), { 8, 16, 32 });
    }
    const auto& grid_strides = _grid_strides.value();
    std::vector<ArmorObject> out_objs;
    const int num_anchors =
        static_cast<int>(std::min<size_t>(grid_strides.size(), static_cast<size_t>(out.rows)));
    for (int a = 0; a < num_anchors; ++a) {
        const float confidence = out.at<float>(a, 8);
        if (confidence < conf_threshold_)
            continue;

        const auto& gs = grid_strides[a];
        const int gx = gs.grid0, gy = gs.grid1, stride = gs.stride;

        // color & class
        const int color_offset = 9;
        const int num_colors = ModelTraits<Mode::TUP>::NUM_COLORS;
        const int num_classes = ModelTraits<Mode::TUP>::NUM_CLASSES;

        cv::Mat color_scores = out.row(a).colRange(color_offset, color_offset + num_colors);
        cv::Mat class_scores =
            out.row(a).colRange(color_offset + num_colors, color_offset + num_colors + num_classes);

        double max_color, max_class;
        cv::Point color_id, class_id;
        cv::minMaxLoc(color_scores, nullptr, &max_color, nullptr, &color_id);
        cv::minMaxLoc(class_scores, nullptr, &max_class, nullptr, &class_id);

        const float x1 = (out.at<float>(a, 0) + gx) * stride;
        const float y1 = (out.at<float>(a, 1) + gy) * stride;
        const float x2 = (out.at<float>(a, 2) + gx) * stride;
        const float y2 = (out.at<float>(a, 3) + gy) * stride;
        const float x3 = (out.at<float>(a, 4) + gx) * stride;
        const float y3 = (out.at<float>(a, 5) + gy) * stride;
        const float x4 = (out.at<float>(a, 6) + gx) * stride;
        const float y4 = (out.at<float>(a, 7) + gy) * stride;

        ArmorObject obj;
        obj.pts = { cv::Point2f(x1, y1),
                    cv::Point2f(x2, y2),
                    cv::Point2f(x3, y3),
                    cv::Point2f(x4, y4) };
        obj.box = cv::boundingRect(obj.pts);
        obj.color = static_cast<ArmorColor>(color_id.x);
        obj.number = static_cast<ArmorNumber>(class_id.x);
        obj.prob = confidence;
        out_objs.push_back(std::move(obj));
    }
    return topKAndNms(out_objs, top_k_, nms_threshold_);
}

std::vector<ArmorObject> ArmorInfer::postProcessRP_impl(const cv::Mat& out) const {
    std::vector<ArmorObject> out_objs;
    const int rows = out.rows;
    const int color_offset = 9;
    const int num_colors = ModelTraits<Mode::RP>::NUM_COLORS;
    const int num_classes = ModelTraits<Mode::RP>::NUM_CLASSES;

    for (int r = 0; r < rows; ++r) {
        float conf_raw = out.at<float>(r, 8);
        const float confidence = static_cast<float>(sigmoid(conf_raw));
        if (confidence < conf_threshold_)
            continue;

        cv::Mat color_scores = out.row(r).colRange(color_offset, color_offset + num_colors);
        cv::Mat class_scores =
            out.row(r).colRange(color_offset + num_colors, color_offset + num_colors + num_classes);

        double max_color_score, max_class_score;
        cv::Point color_id, class_id;
        cv::minMaxLoc(color_scores, nullptr, &max_color_score, nullptr, &color_id);
        cv::minMaxLoc(class_scores, nullptr, &max_class_score, nullptr, &class_id);

        ArmorObject obj;
        obj.pts.resize(4);
        for (int k = 0; k < 4; ++k) {
            const float x = out.at<float>(r, 0 + k * 2);
            const float y = out.at<float>(r, 1 + k * 2);
            obj.pts[k] = cv::Point2f(x, y);
        }
        obj.box = cv::boundingRect(obj.pts);

        if (color_id.x == 0)
            obj.color = ArmorColor::RED;
        else if (color_id.x == 1)
            obj.color = ArmorColor::BLUE;
        else
            obj.color = ArmorColor::NONE;

        obj.number = static_cast<ArmorNumber>(class_id.x);
        obj.prob = confidence;
        out_objs.push_back(std::move(obj));
    }

    return topKAndNms(out_objs, top_k_, nms_threshold_);
}

std::vector<ArmorObject> ArmorInfer::postProcessAT_impl(const cv::Mat& out) const {
    std::vector<ArmorObject> out_objs;
    if (out.empty())
        return out_objs;

    // AT expects dims x anchors (rows = dims, cols = anchors)
    const int dims = out.rows;
    const int anchors = out.cols;
    constexpr int nkpt = ModelTraits<Mode::AT>::NUM_KPTS;
    constexpr int kpt_dim = 2;
    constexpr int nk = nkpt * kpt_dim;
    const int num_cls = dims - 4 - nk;
    if (num_cls <= 0) {
        // invalid layout
        return out_objs;
    }

    for (int a = 0; a < anchors; ++a) {
        auto read = [&](int r) -> float { return out.at<float>(r, a); };
        const float cx = read(0);
        const float cy = read(1);
        const float w = read(2);
        const float h = read(3);
        if (!std::isfinite(cx) || !std::isfinite(cy) || w <= 0.f || h <= 0.f)
            continue;

        float best_score = -std::numeric_limits<float>::infinity();
        int best_idx = -1;
        for (int c = 0; c < num_cls; ++c) {
            const float sc = read(4 + c);
            if (sc > best_score) {
                best_score = sc;
                best_idx = c;
            }
        }
        if (best_idx < 0 || best_score < conf_threshold_)
            continue;

        const int combined = best_idx;
        const int color = combined / ModelTraits<Mode::AT>::NUM_CLASSES;
        const int cls = combined % ModelTraits<Mode::AT>::NUM_CLASSES;

        ArmorObject obj;
        obj.number = ArmorNumber::NO3; // placeholder — original set NO3
        obj.color = ArmorColor::BLUE; // original default
        obj.prob = best_score;

        // keypoints start at row (4 + num_cls)
        const int kp_base = 4 + num_cls;
        obj.pts.reserve(nkpt);
        for (int k = 0; k < nkpt; ++k) {
            const float kx = read(kp_base + 2 * k + 0);
            const float ky = read(kp_base + 2 * k + 1);
            obj.pts.emplace_back(kx, ky);
        }
        obj.box = cv::boundingRect(obj.pts);
        out_objs.push_back(std::move(obj));
    }

    return topKAndNms(out_objs, top_k_, nms_threshold_);
}
std::vector<ArmorObject> ArmorInfer::postProcessAT2_impl(const cv::Mat& out) const {
    std::vector<ArmorObject> out_objs;

    constexpr int nkpt = ModelTraits<Mode::AT2>::NUM_KPTS;
    constexpr int nk = nkpt * 2; // keypoints flattened
    auto max_det = out.rows;
    auto det_dim = out.cols;
    auto output_ptr = out.ptr<float>();
    for (int i = 0; i < max_det; ++i) {
        const float* row = output_ptr + i * det_dim;
        float conf = row[4];
        if (!std::isfinite(conf) || conf < conf_threshold_)
            continue;
        float x = row[0];
        float y = row[1];
        float w = row[2];
        float h = row[3];
        int cls = static_cast<int>(row[5]);

        if (!std::isfinite(x) || !std::isfinite(y) || w <= 0.f || h <= 0.f)
            continue;

        ArmorObject obj;
        obj.prob = conf;
        auto color_num = ModelTraits<Mode::AT2>::CLASSES[cls];
        obj.color = color_num.first;
        obj.number = color_num.second;

        obj.pts.reserve(nkpt);
        for (int k = 0; k < nkpt; ++k) {
            float kx = row[6 + 2 * k];
            float ky = row[6 + 2 * k + 1];
            obj.pts.emplace_back(kx, ky);
        }

        obj.box = cv::boundingRect(obj.pts);
        out_objs.emplace_back(std::move(obj));
    }

    return out_objs;
}
} // namespace wust_vision::auto_aim::armor_infer