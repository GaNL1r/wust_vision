// armor_infer_modern.hpp
#pragma once

#include "tasks/auto_aim/type.hpp"
#include "tasks/type_common.hpp"

#include <Eigen/Core>
#include <opencv2/opencv.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <functional>
#include <limits>
#include <numeric>
#include <optional>
#include <span>
#include <string>
#include <vector>

namespace wust_vision::auto_aim::armor_infer {

static constexpr float MERGE_CONF_ERROR = 0.15f;
static constexpr float MERGE_MIN_IOU = 0.9f;

enum class Mode { TUP, RP, AT };

inline Mode modeFromString(const std::string& m) {
    if (m == "tup" || m == "TUP")
        return Mode::TUP;
    if (m == "rp" || m == "RP")
        return Mode::RP;
    if (m == "at" || m == "AT")
        return Mode::AT;
    return Mode::TUP;
}

// ------------------------- model traits -------------------------
template<Mode M>
struct ModelTraits; // declare
// TUP
template<>
struct ModelTraits<Mode::TUP> {
    static constexpr int INPUT_W = 416;
    static constexpr int INPUT_H = 416;
    static constexpr int NUM_CLASSES = 8;
    static constexpr int NUM_COLORS = 4;
    static constexpr bool USE_NORM = false;
};

// RP
template<>
struct ModelTraits<Mode::RP> {
    static constexpr int INPUT_W = 640;
    static constexpr int INPUT_H = 640;
    static constexpr int NUM_CLASSES = 9;
    static constexpr int NUM_COLORS = 4;
    static constexpr bool USE_NORM = true;
};

// AT
template<>
struct ModelTraits<Mode::AT> {
    static constexpr int INPUT_W = 640;
    static constexpr int INPUT_H = 640;
    static constexpr int NUM_CLASSES = 13; // per-color classes
    static constexpr int NUM_COLORS = 4;
    static constexpr int NUM_KPTS = 4;
    static constexpr bool USE_NORM = true;
};

// ------------------------- utilities -------------------------
[[nodiscard]] inline double sigmoid(double x) noexcept {
    return x >= 0 ? 1.0 / (1.0 + std::exp(-x)) : std::exp(x) / (1.0 + std::exp(x));
}

inline float rectIoU(const cv::Rect2f& a, const cv::Rect2f& b) noexcept {
    const cv::Rect2f inter = a & b;
    const float inter_area = inter.area();
    const float union_area = a.area() + b.area() - inter_area;
    if (union_area <= 0.f || std::isnan(union_area))
        return 0.f;
    return inter_area / union_area;
}

// Merge / NMS helpers that mimic original intent but clearer
inline void nms_merge_sorted_bboxes(
    std::vector<ArmorObject>& objs,
    std::vector<int>& out_indices,
    float nms_threshold
) {
    out_indices.clear();
    const size_t n = objs.size();
    std::vector<float> areas(n);
    for (size_t i = 0; i < n; ++i)
        areas[i] = objs[i].box.area();

    for (size_t i = 0; i < n; ++i) {
        ArmorObject& a = objs[i];
        bool keep = true;
        for (int idx: out_indices) {
            ArmorObject& b = objs[idx];
            const float iou = rectIoU(a.box, b.box);
            if (std::isnan(iou) || iou > nms_threshold) {
                keep = false;
                if (a.number == b.number && a.color == b.color && iou > MERGE_MIN_IOU
                    && std::abs(a.prob - b.prob) < MERGE_CONF_ERROR)
                {
                    // accumulate points for later averaging
                    for (const auto& pt: a.pts)
                        b.pts.push_back(pt);
                }
                break;
            }
        }
        if (keep)
            out_indices.push_back(static_cast<int>(i));
    }
}

inline std::vector<ArmorObject>
topKAndNms(std::vector<ArmorObject>& objs, int top_k, float nms_threshold) {
    std::sort(objs.begin(), objs.end(), [](const ArmorObject& a, const ArmorObject& b) {
        return a.prob > b.prob;
    });
    if (static_cast<int>(objs.size()) > top_k)
        objs.resize(static_cast<size_t>(top_k));

    std::vector<int> indices;
    nms_merge_sorted_bboxes(objs, indices, nms_threshold);

    std::vector<ArmorObject> result;
    result.reserve(indices.size());
    for (size_t i = 0; i < indices.size(); ++i) {
        result.push_back(std::move(objs[indices[i]]));
        // average merged extra points if any
        auto& ro = result.back();
        if (ro.pts.size() >= 8) {
            const size_t npts = ro.pts.size();
            std::array<cv::Point2f, 4> accum { { { 0, 0 }, { 0, 0 }, { 0, 0 }, { 0, 0 } } };
            for (size_t j = 0; j < npts; ++j)
                accum[j % 4] += ro.pts[j];
            ro.pts.resize(4);
            for (int k = 0; k < 4; ++k) {
                float denom = static_cast<float>(npts) / 4.0f;
                ro.pts[k].x = accum[k].x / denom;
                ro.pts[k].y = accum[k].y / denom;
            }
        }
    }
    return result;
}

// ------------------------- per-mode implementations -------------------------

// Generic accessor helpers to read a row or column safely
template<typename Fn>
static void for_each_anchor_row(const cv::Mat& mat, Fn fn) {
    // interpret rows as anchors (each row contains elements)
    const int rows = mat.rows;
    for (int r = 0; r < rows; ++r) {
        const float* rowptr = mat.ptr<float>(r);
        fn(r, rowptr);
    }
}

template<typename Fn>
static void for_each_anchor_col(const cv::Mat& mat, Fn fn) {
    // interpret columns as anchors (mat is dims x anchors)
    const int cols = mat.cols;
    for (int c = 0; c < cols; ++c) {
        fn(c, [&mat, c](int r) -> float { return mat.at<float>(r, c); });
    }
}

// However providing full modern unified class below that delegates using templated helpers.

class ArmorInfer {
public:
    ArmorInfer(
        Mode mode = Mode::TUP,
        float conf_threshold = 0.25f,
        float nms_threshold = 0.45f,
        int top_k = 100
    ) noexcept:
        mode_(mode),
        conf_threshold_(conf_threshold),
        nms_threshold_(nms_threshold),
        top_k_(top_k) {
        setMode(mode_);
    }

    void setMode(Mode m) noexcept {
        mode_ = m;
        switch (mode_) {
            case Mode::TUP: {
                input_w_ = ModelTraits<Mode::TUP>::INPUT_W;
                input_h_ = ModelTraits<Mode::TUP>::INPUT_H;
                use_norm_ = ModelTraits<Mode::TUP>::USE_NORM;
                break;
            }

            case Mode::RP: {
                input_w_ = ModelTraits<Mode::RP>::INPUT_W;
                input_h_ = ModelTraits<Mode::RP>::INPUT_H;
                use_norm_ = ModelTraits<Mode::RP>::USE_NORM;
                break;
            }

            case Mode::AT: {
                input_w_ = ModelTraits<Mode::AT>::INPUT_W;
                input_h_ = ModelTraits<Mode::AT>::INPUT_H;
                use_norm_ = ModelTraits<Mode::AT>::USE_NORM;
            }

            break;
        }
    }

    void setConfThreshold(float t) noexcept {
        conf_threshold_ = t;
    }
    void setNmsThreshold(float t) noexcept {
        nms_threshold_ = t;
    }
    void setTopK(int k) noexcept {
        top_k_ = k;
    }

    int inputW() const noexcept {
        return input_w_;
    }
    int inputH() const noexcept {
        return input_h_;
    }
    bool useNorm() const noexcept {
        return use_norm_;
    }

    // main dispatching entry (keeps original signature)
    [[nodiscard]] std::vector<ArmorObject> postProcess(const cv::Mat& output_buffer) const {
        switch (mode_) {
            case Mode::TUP:
                return postProcessTUP_impl(output_buffer);
            case Mode::RP:
                return postProcessRP_impl(output_buffer);
            case Mode::AT:
                return postProcessAT_impl(output_buffer);
        }
        return {};
    }

private:
    std::vector<ArmorObject> postProcessTUP_impl(const cv::Mat& out) const;

    std::vector<ArmorObject> postProcessRP_impl(const cv::Mat& out) const;

    std::vector<ArmorObject> postProcessAT_impl(const cv::Mat& out) const;

private:
    Mode mode_;
    int input_w_ { 0 };
    int input_h_ { 0 };
    float conf_threshold_ { 0.25f };
    float nms_threshold_ { 0.45f };
    int top_k_ { 100 };
    bool use_norm_ { false };
};

} // namespace wust_vision::auto_aim::armor_infer