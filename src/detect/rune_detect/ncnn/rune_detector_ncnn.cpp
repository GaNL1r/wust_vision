// Copyright 2025 Xiaojian Wu
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "detect/rune_detect/ncnn/rune_detector_ncnn.hpp"
// std
#include <algorithm>
#include <numeric>
#include <unordered_map>
// third party
#include <opencv2/imgproc.hpp>
// project
#include "common/gobal.hpp"
#include "type/type.hpp"

static constexpr int INPUT_W = 480; // Width of input
static constexpr int INPUT_H = 480; // Height of input
static constexpr int NUM_CLASSES = 2; // Number of classes
static constexpr int NUM_COLORS = 2; // Number of color
static constexpr int NUM_POINTS = 5;
static constexpr int NUM_POINTS_2 = 2 * NUM_POINTS;
static constexpr float MERGE_CONF_ERROR = 0.15;
static constexpr float MERGE_MIN_IOU = 0.9;
// 由于训练失误，网络的颜色是反的
static std::unordered_map<int, EnemyColor> DNN_COLOR_TO_ENEMY_COLOR = { { 0, EnemyColor::BLUE },
                                                                        { 1, EnemyColor::RED } };

static cv::Mat letterbox(
    const cv::Mat& img,
    Eigen::Matrix3f& transform_matrix,
    std::vector<int> new_shape = { INPUT_W, INPUT_H }
) {
    // Get current image shape [height, width]

    int img_h = img.rows;
    int img_w = img.cols;

    // Compute scale ratio(new / old) and target resized shape
    float scale = std::min(new_shape[1] * 1.0 / img_h, new_shape[0] * 1.0 / img_w);
    int resize_h = static_cast<int>(round(img_h * scale));
    int resize_w = static_cast<int>(round(img_w * scale));

    // Compute padding
    int pad_h = new_shape[1] - resize_h;
    int pad_w = new_shape[0] - resize_w;

    // Resize and pad image while meeting stride-multiple constraints
    cv::Mat resized_img;
    cv::resize(img, resized_img, cv::Size(resize_w, resize_h));

    // divide padding into 2 sides
    float half_h = pad_h * 1.0 / 2;
    float half_w = pad_w * 1.0 / 2;

    // Compute padding boarder
    int top = static_cast<int>(round(half_h - 0.1));
    int bottom = static_cast<int>(round(half_h + 0.1));
    int left = static_cast<int>(round(half_w - 0.1));
    int right = static_cast<int>(round(half_w + 0.1));

    /* clang-format off */
    /* *INDENT-OFF* */

    // Compute point transform_matrix
    transform_matrix << 1.0 / scale, 0, -half_w / scale,
                        0, 1.0 / scale, -half_h / scale,
                        0, 0, 1;

    /* *INDENT-ON* */
    /* clang-format on */

    // Add border
    cv::copyMakeBorder(
        resized_img,
        resized_img,
        top,
        bottom,
        left,
        right,
        cv::BORDER_CONSTANT,
        cv::Scalar(114, 114, 114)
    );

    return resized_img;
}
static ncnn::Mat letterbox_to_ncnn(
    const cv::Mat& img,
    Eigen::Matrix3f& transform_matrix,
    int out_w = INPUT_W,
    int out_h = INPUT_H
) {
    const int img_w = img.cols;
    const int img_h = img.rows;

    float scale = std::min(out_w * 1.0f / img_w, out_h * 1.0f / img_h);
    int resize_w = static_cast<int>(round(img_w * scale));
    int resize_h = static_cast<int>(round(img_h * scale));

    int pad_w = out_w - resize_w;
    int pad_h = out_h - resize_h;
    int pad_left = static_cast<int>(round(pad_w / 2.0f - 0.1f));
    int pad_top = static_cast<int>(round(pad_h / 2.0f - 0.1f));

    transform_matrix << 1.0f / scale, 0, -pad_left / scale, 0, 1.0f / scale, -pad_top / scale, 0, 0,
        1;

    ncnn::Mat out = ncnn::Mat::from_pixels_resize(
        img.data,
        ncnn::Mat::PIXEL_BGR2RGB,
        img_w,
        img_h,
        resize_w,
        resize_h
    );

    int pad_right = out_w - resize_w - pad_left;
    int pad_bottom = out_h - resize_h - pad_top;

    ncnn::Mat padded;
    ncnn::copy_make_border(
        out,
        padded,
        pad_top,
        pad_bottom,
        pad_left,
        pad_right,
        ncnn::BORDER_CONSTANT,
        114.f
    );

    return padded;
}

// Generate grids and stride for post processing
// target_w: Width of input.
// target_h: Height of input.
// strides A vector of stride.
// grid_strides Grid stride generated in this function
static void generateGridsAndStride(
    const int target_w,
    const int target_h,
    std::vector<int>& strides,
    std::vector<GridAndStride>& grid_strides
) {
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

// Decode output tensor
static void generateProposals(
    std::vector<rune::RuneObject>& output_objs,
    std::vector<float>& scores,
    std::vector<cv::Rect>& rects,
    const cv::Mat& output_buffer,
    const Eigen::Matrix<float, 3, 3>& transform_matrix,
    float conf_threshold,
    std::vector<GridAndStride> grid_strides
) {
    const int num_anchors = grid_strides.size();

    for (int anchor_idx = 0; anchor_idx < num_anchors; anchor_idx++) {
        float confidence = output_buffer.at<float>(anchor_idx, NUM_POINTS_2);
        if (confidence < conf_threshold) {
            continue;
        }

        const int grid0 = grid_strides[anchor_idx].grid0;
        const int grid1 = grid_strides[anchor_idx].grid1;
        const int stride = grid_strides[anchor_idx].stride;

        double color_score, class_score;
        cv::Point color_id, class_id;
        cv::Mat color_scores =
            output_buffer.row(anchor_idx).colRange(NUM_POINTS_2 + 1, NUM_POINTS_2 + 1 + NUM_COLORS);
        cv::Mat num_scores = output_buffer.row(anchor_idx)
                                 .colRange(
                                     NUM_POINTS_2 + 1 + NUM_COLORS,
                                     NUM_POINTS_2 + 1 + NUM_COLORS + NUM_CLASSES
                                 );
        // Argmax
        cv::minMaxLoc(color_scores, NULL, &color_score, NULL, &color_id);
        cv::minMaxLoc(num_scores, NULL, &class_score, NULL, &class_id);

        float x_1 = (output_buffer.at<float>(anchor_idx, 0) + grid0) * stride;
        float y_1 = (output_buffer.at<float>(anchor_idx, 1) + grid1) * stride;
        float x_2 = (output_buffer.at<float>(anchor_idx, 2) + grid0) * stride;
        float y_2 = (output_buffer.at<float>(anchor_idx, 3) + grid1) * stride;
        float x_3 = (output_buffer.at<float>(anchor_idx, 4) + grid0) * stride;
        float y_3 = (output_buffer.at<float>(anchor_idx, 5) + grid1) * stride;
        float x_4 = (output_buffer.at<float>(anchor_idx, 6) + grid0) * stride;
        float y_4 = (output_buffer.at<float>(anchor_idx, 7) + grid1) * stride;
        float x_5 = (output_buffer.at<float>(anchor_idx, 8) + grid0) * stride;
        float y_5 = (output_buffer.at<float>(anchor_idx, 9) + grid1) * stride;

        Eigen::Matrix<float, 3, 5> apex_norm;
        Eigen::Matrix<float, 3, 5> apex_dst;

        /* clang-format off */
        /* *INDENT-OFF* */
        apex_norm << x_1, x_2, x_3, x_4, x_5,
                    y_1, y_2, y_3, y_4, y_5,
                    1,   1,   1,   1,   1;
        /* *INDENT-ON* */
        /* clang-format on */

        apex_dst = transform_matrix * apex_norm;

        rune::RuneObject obj;

        obj.pts.r_center = cv::Point2f(apex_dst(0, 0), apex_dst(1, 0));
        obj.pts.bottom_left = cv::Point2f(apex_dst(0, 1), apex_dst(1, 1));
        obj.pts.top_left = cv::Point2f(apex_dst(0, 2), apex_dst(1, 2));
        obj.pts.top_right = cv::Point2f(apex_dst(0, 3), apex_dst(1, 3));
        obj.pts.bottom_right = cv::Point2f(apex_dst(0, 4), apex_dst(1, 4));

        auto rect = cv::boundingRect(obj.pts.toVector2f());

        obj.box = rect;
        obj.color = DNN_COLOR_TO_ENEMY_COLOR[color_id.x];
        obj.type = static_cast<rune::RuneType>(class_id.x);
        obj.prob = confidence;

        rects.push_back(rect);
        scores.push_back(confidence);
        output_objs.push_back(std::move(obj));
    }
}

// Calculate intersection area between Object a and Object b.
static inline float intersectionArea(const rune::RuneObject& a, const rune::RuneObject& b) {
    cv::Rect_<float> inter = a.box & b.box;
    return inter.area();
}

static void nmsMergeSortedBboxes(
    std::vector<rune::RuneObject>& faceobjects,
    std::vector<int>& indices,
    float nms_threshold
) {
    indices.clear();

    const int n = faceobjects.size();

    std::vector<float> areas(n);
    for (int i = 0; i < n; i++) {
        areas[i] = faceobjects[i].box.area();
    }

    for (int i = 0; i < n; i++) {
        rune::RuneObject& a = faceobjects[i];

        int keep = 1;
        for (size_t j = 0; j < indices.size(); j++) {
            rune::RuneObject& b = faceobjects[indices[j]];

            // intersection over union
            float inter_area = intersectionArea(a, b);
            float union_area = areas[i] + areas[indices[j]] - inter_area;
            float iou = inter_area / union_area;
            if (iou > nms_threshold || isnan(iou)) {
                keep = 0;
                // Stored for Merge
                if (a.type == b.type && a.color == b.color && iou > MERGE_MIN_IOU
                    && abs(a.prob - b.prob) < MERGE_CONF_ERROR)
                {
                    a.pts.children.push_back(b.pts);
                }
                // cout<<b.pts_x.size()<<endl;
            }
        }

        if (keep) {
            indices.push_back(i);
        }
    }
}

RuneDetectorNCNN::RuneDetectorNCNN(
    std::string input_name_,
    std::string output_name_,
    const std::string& model_path_param,
    const std::string& model_path_bin,
    float conf_threshold,
    int top_k,
    float nms_threshold,
    bool use_gpu,
    int cpu_threads,
    bool use_lightmode,
    int device_id
):
    input_name_(input_name_),
    output_name_(output_name_),
    model_path_param_(model_path_param),
    model_path_bin_(model_path_bin),
    conf_threshold_(conf_threshold),
    top_k_(top_k),
    nms_threshold_(nms_threshold),
    use_gpu_(use_gpu),
    cpu_threads_(cpu_threads),
    use_lightmode_(use_lightmode) {
    init(device_id);
}
RuneDetectorNCNN::~RuneDetectorNCNN() {
    net_.clear();
}

void RuneDetectorNCNN::init(int device_id) {
    if (use_gpu_) {
        ncnn::create_gpu_instance();
        opt_.use_vulkan_compute = true;
        ncnn::VulkanDevice* vkdev = ncnn::get_gpu_device(device_id);
        if (vkdev) {
            net_.set_vulkan_device(vkdev);
        }
        WUST_INFO("armor_ncnn") << "ncnn: use gpu";
    } else {
        opt_.use_vulkan_compute = false;
        WUST_INFO("rune_ncnn") << "ncnn: use cpu";
    }
    if (use_lightmode_) {
        opt_.lightmode = true;
    }

    opt_.num_threads = cpu_threads_;
    net_.opt = opt_;
    WUST_INFO("rune_ncnn") << "ncnn: using " << cpu_threads_ << " threads";

    if (net_.load_param(model_path_param_.c_str()) != 0) {
        WUST_ERROR("rune_ncnn") << "Failed to load param";
        return;
    }
    if (net_.load_model(model_path_bin_.c_str()) != 0) {
        WUST_ERROR("rune_ncnn") << "Failed to load model";
        return;
    }

    int ret = net_.load_param((model_path_param_).c_str());
    if (ret != 0) {
        WUST_ERROR("rune_ncnn") << "Failed to load param file: " << model_path_param_;
        return;
    }

    ret = net_.load_model((model_path_bin_).c_str());
    if (ret != 0) {
        WUST_ERROR("rune_ncnn") << "Failed to load bin file: " << model_path_bin_;
        return;
    }
    // input_name_ = "images";
    // output_name_ = "output";
    strides_ = { 8, 16, 32 };
    grid_strides_.clear();
    generateGridsAndStride(INPUT_W, INPUT_H, strides_, grid_strides_);

    WUST_INFO("rune_ncnn") << "ncnn: model loaded successfully";
}

void RuneDetectorNCNN::pushInput(const CommonFrame& frame) {
    // Reprocess

    processCallback(frame);
}

void RuneDetectorNCNN::setCallback(CallbackType callback) {
    infer_callback_ = callback;
}

bool RuneDetectorNCNN::processCallback(const CommonFrame& frame) {
    Eigen::Matrix3f transform_matrix; // transform matrix from resized image to source image.
    // cv::Mat resized_img = letterbox(frame.src_img, transform_matrix);
    // ncnn::Mat in = ncnn::Mat::from_pixels(
    //     resized_img.data,
    //     ncnn::Mat::PIXEL_BGR2RGB, // OpenCV 默认 BGR，转为 RGB
    //     INPUT_W,
    //     INPUT_H
    // );
    ncnn::Mat in = letterbox_to_ncnn(frame.src_img, transform_matrix);

    ncnn::Extractor ex = net_.create_extractor();

    ex.input(input_name_.c_str(), in);

    ncnn::Mat out;
    ex.extract(output_name_.c_str(), out);

    cv::Mat output_buffer(out.h, out.w, CV_32F, out.data);
    // Parsed variable
    std::vector<rune::RuneObject> objs_tmp, objs_result;
    std::vector<cv::Rect> rects;
    std::vector<float> scores;
    std::vector<int> indices;

    // Parse YOLO output
    generateProposals(
        objs_tmp,
        scores,
        rects,
        output_buffer,
        transform_matrix,
        this->conf_threshold_,
        this->grid_strides_
    );

    // TopK
    std::sort(
        objs_tmp.begin(),
        objs_tmp.end(),
        [](const rune::RuneObject& a, const rune::RuneObject& b) { return a.prob > b.prob; }
    );
    if (objs_tmp.size() > static_cast<size_t>(this->top_k_)) {
        objs_tmp.resize(this->top_k_);
    }

    nmsMergeSortedBboxes(objs_tmp, indices, this->nms_threshold_);

    for (size_t i = 0; i < indices.size(); i++) {
        objs_result.push_back(std::move(objs_tmp[indices[i]]));

        if (objs_result[i].pts.children.size() > 0) {
            const float N = static_cast<float>(objs_result[i].pts.children.size() + 1);
            rune::FeaturePoints pts_final = std::accumulate(
                objs_result[i].pts.children.begin(),
                objs_result[i].pts.children.end(),
                objs_result[i].pts
            );
            objs_result[i].pts = pts_final / N;
        }
    }
    objs_result.erase(
        std::remove_if(
            objs_result.begin(),
            objs_result.end(),
            [c = static_cast<EnemyColor>(gobal::detect_color)](const auto& objs_result) {
                return objs_result.color != c;
            }
        ),
        objs_result.end()
    );

    // NMS & TopK
    // cv::dnn::NMSBoxes(
    //   rects, scores, this->conf_threshold_, this->nms_threshold_, indices, 1.0,
    //   this->top_k_);
    // for (size_t i = 0; i < indices.size(); ++i) {
    //   objs_result.push_back(std::move(objs_tmp[i]));
    // }

    // Call callback function
    if (this->infer_callback_) {
        this->infer_callback_(objs_result, frame);
        return true;
    }

    return false;
}

std::tuple<cv::Point2f, cv::Mat> RuneDetectorNCNN::detectRTag(
    const cv::Mat& img,
    int binary_thresh,
    const cv::Point2f& prior,
    bool precise
) {
    if (!img.data || img.cols <= 0 || img.rows <= 0) {
        std::cerr << "[detectRTag] Invalid input image." << std::endl;
        return {};
    }

    if (prior.x < 0 || prior.x >= img.cols || prior.y < 0 || prior.y >= img.rows) {
        std::cerr << "[detectRTag] Prior out of bounds: " << prior
                  << " for image size: " << img.cols << "x" << img.rows << std::endl;
        return { prior, cv::Mat::zeros(cv::Size(200, 200), CV_8UC3) };
    }
    int px = static_cast<int>(std::floor(prior.x));
    int py = static_cast<int>(std::floor(prior.y));
    if (px < 0 || px >= img.cols || py < 0 || py >= img.rows) {
        std::cerr << "[detectRTag] Prior out of bounds: " << prior
                  << " for image size: " << img.cols << "x" << img.rows << std::endl;
        return { prior, cv::Mat::zeros(cv::Size(200, 200), CV_8UC3) };
    }

    // ROI calculation
    cv::Rect roi;
    if (precise) {
        roi = cv::Rect(prior.x - 30, prior.y - 30, 60, 60) & cv::Rect(0, 0, img.cols, img.rows);
    } else {
        roi = cv::Rect(prior.x - 100, prior.y - 100, 200, 200) & cv::Rect(0, 0, img.cols, img.rows);
    }

    if (roi.width == 0 || roi.height == 0) {
        std::cerr << "[detectRTag] ROI is zero-sized: " << roi << std::endl;
        return { prior, cv::Mat::zeros(200, 200, CV_8UC3) };
    }

    // Create ROI

    const cv::Point2f prior_in_roi = prior - cv::Point2f(roi.tl());

    cv::Mat img_roi = img(roi);

    // Gray -> Binary -> Dilate
    cv::Mat gray_img;
    cv::cvtColor(img_roi, gray_img, cv::COLOR_BGR2GRAY);
    cv::Mat binary_img;
    cv::threshold(gray_img, binary_img, 0, 255, cv::THRESH_BINARY | cv::THRESH_OTSU);
    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3));
    cv::dilate(binary_img, binary_img, kernel);

    // Find contours
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(binary_img, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_NONE);

    auto it = std::find_if(
        contours.begin(),
        contours.end(),
        [p = prior_in_roi](const std::vector<cv::Point>& contour) -> bool {
            return cv::boundingRect(contour).contains(p);
        }
    );

    // For visualization
    cv::cvtColor(binary_img, binary_img, cv::COLOR_GRAY2BGR);

    if (it == contours.end()) {
        return { prior, binary_img };
    }

    cv::drawContours(binary_img, contours, it - contours.begin(), cv::Scalar(0, 255, 0), 2);

    cv::Point2f center = std::accumulate(it->begin(), it->end(), cv::Point(0, 0));
    center /= static_cast<float>(it->size());
    center += cv::Point2f(roi.tl());

    return { center, binary_img };
}
