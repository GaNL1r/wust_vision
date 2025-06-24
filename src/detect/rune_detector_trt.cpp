// Copyright 2023 Yunlong Feng
//
// Additional modifications and features by Chengfu Zou, 2024.
//
// Copyright (C) FYT Vision Group. All rights reserved.
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

#include "detect/rune_detector_trt.hpp"
// std
#include <algorithm>
#include <fmt/color.h>
#include <numeric>
#include <unordered_map>
// third party
#include <opencv2/imgproc.hpp>
// project
#include "common/common.hpp"
#include "type/type.hpp"
#include "common/logger.hpp"
#include "NvOnnxParser.h"
#include "fmt/core.h"

#define TRT_ASSERT(expr)                                                       \
  do {                                                                         \
    if (!(expr)) {                                                             \
      fmt::print(fmt::fg(fmt::color::red), "assert fail: '" #expr "'");        \
      exit(-1);                                                                \
    }                                                                          \
  } while (0)


static constexpr int INPUT_W = 480;   // Width of input
static constexpr int INPUT_H = 480;   // Height of input
static constexpr int NUM_CLASSES = 2; // Number of classes
static constexpr int NUM_COLORS = 2;  // Number of color
static constexpr int NUM_POINTS = 5;
static constexpr int NUM_POINTS_2 = 2 * NUM_POINTS;
static constexpr float MERGE_CONF_ERROR = 0.15;
static constexpr float MERGE_MIN_IOU = 0.9;
// 由于训练失误，网络的颜色是反的
static std::unordered_map<int, EnemyColor> DNN_COLOR_TO_ENEMY_COLOR = {
    {0, EnemyColor::BLUE}, {1, EnemyColor::RED}};

static cv::Mat letterbox(const cv::Mat &img, Eigen::Matrix3f &transform_matrix,
                         std::vector<int> new_shape = {INPUT_W, INPUT_H}) {
  // Get current image shape [height, width]

  int img_h = img.rows;
  int img_w = img.cols;

  // Compute scale ratio(new / old) and target resized shape
  float scale =
      std::min(new_shape[1] * 1.0 / img_h, new_shape[0] * 1.0 / img_w);
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
  cv::copyMakeBorder(resized_img, resized_img, top, bottom, left, right,
                     cv::BORDER_CONSTANT, cv::Scalar(114, 114, 114));

  return resized_img;
}

// Generate grids and stride for post processing
// target_w: Width of input.
// target_h: Height of input.
// strides A vector of stride.
// grid_strides Grid stride generated in this function
static void generateGridsAndStride(const int target_w, const int target_h,
                                   std::vector<int> &strides,
                                   std::vector<GridAndStride> &grid_strides) {
  for (auto stride : strides) {
    int num_grid_w = target_w / stride;
    int num_grid_h = target_h / stride;

    for (int g1 = 0; g1 < num_grid_h; g1++) {
      for (int g0 = 0; g0 < num_grid_w; g0++) {
        grid_strides.emplace_back(GridAndStride{g0, g1, stride});
      }
    }
  }
}

// Decode output tensor
static void generateProposals(
    std::vector<RuneObject> &output_objs, std::vector<float> &scores,
    std::vector<cv::Rect> &rects, const cv::Mat &output_buffer,
    const Eigen::Matrix<float, 3, 3> &transform_matrix, float conf_threshold,
    std::vector<GridAndStride> grid_strides) {
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
        output_buffer.row(anchor_idx)
            .colRange(NUM_POINTS_2 + 1, NUM_POINTS_2 + 1 + NUM_COLORS);
    cv::Mat num_scores =
        output_buffer.row(anchor_idx)
            .colRange(NUM_POINTS_2 + 1 + NUM_COLORS,
                      NUM_POINTS_2 + 1 + NUM_COLORS + NUM_CLASSES);
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

    RuneObject obj;

    obj.pts.r_center = cv::Point2f(apex_dst(0, 0), apex_dst(1, 0));
    obj.pts.bottom_left = cv::Point2f(apex_dst(0, 1), apex_dst(1, 1));
    obj.pts.top_left = cv::Point2f(apex_dst(0, 2), apex_dst(1, 2));
    obj.pts.top_right = cv::Point2f(apex_dst(0, 3), apex_dst(1, 3));
    obj.pts.bottom_right = cv::Point2f(apex_dst(0, 4), apex_dst(1, 4));

    auto rect = cv::boundingRect(obj.pts.toVector2f());

    obj.box = rect;
    obj.color = DNN_COLOR_TO_ENEMY_COLOR[color_id.x];
    obj.type = static_cast<RuneType>(class_id.x);
    obj.prob = confidence;

    rects.push_back(rect);
    scores.push_back(confidence);
    output_objs.push_back(std::move(obj));
  }
}
static bool hasNanInApexDst(const Eigen::Matrix<float, 3, 5> &mat) {
    for (int col = 0; col < 5; ++col) {
        if (std::isnan(mat(0, col)) || std::isnan(mat(1, col))) {
            return true;
        }
    }
    return false;
}

// Calculate intersection area between Object a and Object b.
static inline float intersectionArea(const RuneObject &a, const RuneObject &b) {
  cv::Rect_<float> inter = a.box & b.box;
  return inter.area();
}

static void nmsMergeSortedBboxes(std::vector<RuneObject> &faceobjects,
                                 std::vector<int> &indices,
                                 float nms_threshold) {
  indices.clear();

  const int n = faceobjects.size();

  std::vector<float> areas(n);
  for (int i = 0; i < n; i++) {
    areas[i] = faceobjects[i].box.area();
  }

  for (int i = 0; i < n; i++) {
    RuneObject &a = faceobjects[i];

    int keep = 1;
    for (size_t j = 0; j < indices.size(); j++) {
      RuneObject &b = faceobjects[indices[j]];

      // intersection over union
      float inter_area = intersectionArea(a, b);
      float union_area = areas[i] + areas[indices[j]] - inter_area;
      float iou = inter_area / union_area;
      if (iou > nms_threshold || isnan(iou)) {
        keep = 0;
        // Stored for Merge
        if (a.type == b.type && a.color == b.color && iou > MERGE_MIN_IOU &&
            abs(a.prob - b.prob) < MERGE_CONF_ERROR) {
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

RuneDetectorTrt::RuneDetectorTrt(const std::filesystem::path &model_path,
    const Params &params
                           )
    : model_path_(model_path) {
        buildEngine(model_path);
        TRT_ASSERT(context_ = engine_->createExecutionContext());
        TRT_ASSERT((input_idx_ = engine_->getBindingIndex("images")) == 0);
        TRT_ASSERT((output_idx_ = engine_->getBindingIndex("output")) == 1);
      
        auto input_dims = engine_->getBindingDimensions(input_idx_);
        auto output_dims = engine_->getBindingDimensions(output_idx_);
        input_sz_ = input_dims.d[1] * input_dims.d[2] * input_dims.d[3];
        output_sz_ = output_dims.d[1] * output_dims.d[2];
        TRT_ASSERT(
            cudaMalloc(&device_buffers_[input_idx_], input_sz_ * sizeof(float)) == 0);
        TRT_ASSERT(cudaMalloc(&device_buffers_[output_idx_],
                              output_sz_ * sizeof(float)) == 0);
        output_buffer_ = new float[output_sz_];
        TRT_ASSERT(cudaStreamCreate(&stream_) == 0);
        strides_ = {8, 16, 32};
        generateGridsAndStride(INPUT_W, INPUT_H, strides_, grid_strides_);
        thread_pool_ =
      std::make_unique<ThreadPool>(std::thread::hardware_concurrency(), 100);
}
void RuneDetectorTrt::buildEngine(const std::string &onnx_path) {
    std::string engine_path =
        onnx_path.substr(0, onnx_path.find_last_of('.')) + ".engine";
    std::ifstream engine_file(engine_path, std::ios::binary);
    if (engine_file.good()) {
      engine_file.seekg(0, std::ios::end);
      size_t size = engine_file.tellg();
      engine_file.seekg(0, std::ios::beg);
      std::vector<char> engine_data(size);
      engine_file.read(engine_data.data(), size);
      engine_file.close();
  
      runtime_ = nvinfer1::createInferRuntime(g_logger_); // ✅ 作为成员变量保存
      engine_ = runtime_->deserializeCudaEngine(engine_data.data(), size);
      if (engine_ != nullptr) {
        WUST_INFO("TRT") << "Load engine from " << engine_path
                         << " successfully.";
        return;
      }
    }
  
    // 构建新引擎
    auto builder = nvinfer1::createInferBuilder(g_logger_);
    const auto explicit_batch =
        1U << static_cast<uint32_t>(
            nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
    auto network = builder->createNetworkV2(explicit_batch);
    auto parser = nvonnxparser::createParser(*network, g_logger_);
    parser->parseFromFile(onnx_path.c_str(),
                          static_cast<int>(nvinfer1::ILogger::Severity::kINFO));
  
    auto config = builder->createBuilderConfig();
    if (builder->platformHasFastFp16()) {
      config->setFlag(nvinfer1::BuilderFlag::kFP16);
    }
    engine_ = builder->buildEngineWithConfig(*network, *config);
  
    // 保存引擎
    auto serialized_engine = engine_->serialize();
    std::ofstream out_file(engine_path, std::ios::binary);
    out_file.write(reinterpret_cast<const char *>(serialized_engine->data()),
                   serialized_engine->size());
    out_file.close();
    serialized_engine->destroy();
  
    // 反序列化仍然需要 runtime_
    if (!runtime_) {
      runtime_ = nvinfer1::createInferRuntime(g_logger_);
    }
  
    // 清理
    parser->destroy();
    network->destroy();
    config->destroy();
    builder->destroy();
  
    WUST_INFO("TRT") << "Build engine from " << onnx_path << " successfully.";
  }

  RuneDetectorTrt::~RuneDetectorTrt() {
    delete[] output_buffer_;
    cudaStreamDestroy(stream_);
    cudaFree(device_buffers_[output_idx_]);
    cudaFree(device_buffers_[input_idx_]);
    if (context_)
      context_->destroy();
    if (engine_)
      engine_->destroy();
    if (runtime_)
      runtime_->destroy();
    thread_pool_.reset();
    if (thread_pool_) {
      thread_pool_->waitUntilEmpty();
    }
  }
void RuneDetectorTrt::pushInput(const cv::Mat &rgb_img,
                             std::chrono::steady_clock::time_point timestamp,Eigen::Matrix4d T_camera_to_odom) {
  if (rgb_img.empty()) {
    std::cerr << "[RuneDetectorTrt::pushInput] Warning: input rgb_img is empty!" << std::endl;
    return;
  }

  Eigen::Matrix3f transform_matrix;
  cv::Mat resized_img = letterbox(rgb_img, transform_matrix);

  if (resized_img.empty()) {
    std::cerr << "[RuneDetectorTrt::pushInput] Warning: resized_img is empty!" << std::endl;
    return;
  }

  thread_pool_->enqueue([this, resized_img, transform_matrix, timestamp, rgb_img,T_camera_to_odom]() {
    this->processCallback(resized_img, transform_matrix, timestamp, rgb_img,T_camera_to_odom);
  });
}

void RuneDetectorTrt::setCallback(CallbackType callback) {
  infer_callback_ = callback;
}

bool RuneDetectorTrt::processCallback(
    const cv::Mat resized_img, Eigen::Matrix3f transform_matrix,
    std::chrono::steady_clock::time_point timestamp, const cv::Mat &src_img,Eigen::Matrix4d T_camera_to_odom) {
  // BGR->RGB, u8(0-255)->f32(0.0-1.0), HWC->NCHW
  // note: TUP's model no need to normalize
  cv::Mat blob = cv::dnn::blobFromImage(
      resized_img, 1., cv::Size(INPUT_W, INPUT_H), cv::Scalar(0, 0, 0), true);

cudaMemcpyAsync(device_buffers_[input_idx_], blob.ptr<float>(),
      input_sz_ * sizeof(float), cudaMemcpyHostToDevice, stream_);
context_->enqueueV2(device_buffers_, stream_, nullptr);
cudaMemcpyAsync(output_buffer_, device_buffers_[output_idx_],
      output_sz_ * sizeof(float), cudaMemcpyDeviceToHost, stream_);
cudaStreamSynchronize(stream_);

std::vector<RuneObject> objs_tmp, objs_result;
std::vector<cv::Rect> rects;
std::vector<float> scores;


  // Parse YOLO output
//   generateProposals(objs_tmp, scores, rects, output_buffer, transform_matrix,
//                     this->conf_threshold_, this->grid_strides_);
    objs_result = postprocess(objs_tmp, scores, rects, output_buffer_,
                            output_sz_ / 15, transform_matrix);
  // TopK
//   std::sort(
//       objs_tmp.begin(), objs_tmp.end(),
//       [](const RuneObject &a, const RuneObject &b) { return a.prob > b.prob; });
//   if (objs_tmp.size() > static_cast<size_t>(this->top_k_)) {
//     objs_tmp.resize(this->top_k_);
//   }

//   nmsMergeSortedBboxes(objs_tmp, indices, this->nms_threshold_);

//   for (size_t i = 0; i < indices.size(); i++) {
//     objs_result.push_back(std::move(objs_tmp[indices[i]]));

//     if (objs_result[i].pts.children.size() > 0) {
//       const float N =
//           static_cast<float>(objs_result[i].pts.children.size() + 1);
//       FeaturePoints pts_final = std::accumulate(
//           objs_result[i].pts.children.begin(),
//           objs_result[i].pts.children.end(), objs_result[i].pts);
//       objs_result[i].pts = pts_final / N;
//     }
//   }

  // NMS & TopK
//   cv::dnn::NMSBoxes(
//     rects, scores, this->conf_threshold_, this->nms_threshold_, indices, 1.0,
//     this->top_k_);
//   for (size_t i = 0; i < indices.size(); ++i) {
//     objs_result.push_back(std::move(objs_tmp[i]));
//   }

  // Call callback function
  if (this->infer_callback_) {
    this->infer_callback_(objs_result, timestamp, src_img,T_camera_to_odom);
    return true;
  }

  return false;
}
std::vector<RuneObject> RuneDetectorTrt::postprocess(
    std::vector<RuneObject> &output_objs, std::vector<float> &scores,
    std::vector<cv::Rect> &rects, const float *output, int num_detections,
    const Eigen::Matrix<float, 3, 3> &transform_matrix) {


  for (int anchor_idx = 0; anchor_idx < num_detections; anchor_idx++) {
    //float confidence = output_buffer.at<float>(anchor_idx, NUM_POINTS_2);
    const float *det = output + anchor_idx * 15;
    float confidence = det[NUM_POINTS_2];

    if (confidence < params_.conf_threshold) {
    
      continue;
    }
    
   // std::cout<<"confidence: "<<confidence<<std::endl;
    const int grid0 = grid_strides_[anchor_idx].grid0;
    const int grid1 = grid_strides_[anchor_idx].grid1;
    const int stride = grid_strides_[anchor_idx].stride;

    double color_score, class_score;
    cv::Point color_id, class_id;
    cv::Mat color_scores(1, NUM_COLORS, CV_32F);
    for (int i = 0; i < NUM_COLORS; ++i) {
        color_scores.at<float>(0, i) = det[NUM_POINTS_2 + 1 + i];
    }

    cv::Mat num_scores(1, NUM_CLASSES, CV_32F);
    for (int i = 0; i < NUM_CLASSES; ++i) {
        num_scores.at<float>(0, i) = det[NUM_POINTS_2 + 1 + NUM_COLORS + i];
    }


    // Argmax
    cv::minMaxLoc(color_scores, NULL, &color_score, NULL, &color_id);
    cv::minMaxLoc(num_scores, NULL, &class_score, NULL, &class_id);

    float x_1 = (det[0] + grid0) * stride;
    float y_1 = (det[1] + grid1) * stride;
    float x_2 = (det[2] + grid0) * stride;
    float y_2 = (det[3] + grid1) * stride;
    float x_3 = (det[4] + grid0) * stride;
    float y_3 = (det[5] + grid1) * stride;
    float x_4 = (det[6] + grid0) * stride;
    float y_4 = (det[7] + grid1) * stride;
    float x_5 = (det[8] + grid0) * stride;
    float y_5 = (det[9] + grid1) * stride;

    auto has_nan = [](const std::initializer_list<float> &vals) {
        for (auto v : vals) {
            if (std::isnan(v)) return true;
        }
        return false;
    };

    if (has_nan({x_1, y_1, x_2, y_2, x_3, y_3, x_4, y_4, x_5, y_5})) {
        continue;
    }

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
    if (hasNanInApexDst(apex_dst)) {
        // 发现 NaN，跳过当前检测
        continue;
    }

    RuneObject obj;

    obj.pts.r_center = cv::Point2f(apex_dst(0, 0), apex_dst(1, 0));
    obj.pts.bottom_left = cv::Point2f(apex_dst(0, 1), apex_dst(1, 1));
    obj.pts.top_left = cv::Point2f(apex_dst(0, 2), apex_dst(1, 2));
    obj.pts.top_right = cv::Point2f(apex_dst(0, 3), apex_dst(1, 3));
    obj.pts.bottom_right = cv::Point2f(apex_dst(0, 4), apex_dst(1, 4));



    auto rect = cv::boundingRect(obj.pts.toVector2f());

    obj.box = rect;
    obj.color = DNN_COLOR_TO_ENEMY_COLOR[color_id.x];
    obj.type = static_cast<RuneType>(class_id.x);
    obj.prob = confidence;

    rects.push_back(rect);
    scores.push_back(confidence);
    output_objs.push_back(std::move(obj));
  }
  std::sort(
    output_objs.begin(), output_objs.end(),
      [](const RuneObject &a, const RuneObject &b) { return a.prob > b.prob; });
  if (output_objs.size() > static_cast<size_t>(this->top_k_)) {
    output_objs.resize(this->top_k_);
  }
  std::vector<int> indices;
  std::vector<RuneObject> objs_result;

  nmsMergeSortedBboxes(output_objs, indices, params_.nms_threshold);

  for (size_t i = 0; i < indices.size(); i++) {
    objs_result.push_back(std::move(output_objs[indices[i]]));

    if (objs_result[i].pts.children.size() > 0) {
      const float N =
          static_cast<float>(objs_result[i].pts.children.size() + 1);
      FeaturePoints pts_final = std::accumulate(
          objs_result[i].pts.children.begin(),
          objs_result[i].pts.children.end(), objs_result[i].pts);
      objs_result[i].pts = pts_final / N;
    }
  }
//     cv::dnn::NMSBoxes(
//     rects, scores, this->conf_threshold_, this->nms_threshold_, indices, 1.0,
//     this->top_k_);
//   for (size_t i = 0; i < indices.size(); ++i) {
//     objs_result.push_back(std::move(output_objs[i]));
//   }

  return objs_result;
}

std::tuple<cv::Point2f, cv::Mat>
RuneDetectorTrt::detectRTag(const cv::Mat &img, int binary_thresh,
                         const cv::Point2f &prior) {
 if (!img.data || img.cols <= 0 || img.rows <= 0) {
    std::cerr << "[detectRTag] Invalid input image." << std::endl;
    return {};
}

if (prior.x < 0 || prior.x >= img.cols || prior.y < 0 || prior.y >= img.rows) {
  std::cerr << "[detectRTag] Prior out of bounds: " << prior << " for image size: "
            << img.cols << "x" << img.rows << std::endl;
  return {prior, cv::Mat::zeros(cv::Size(200, 200), CV_8UC3)};
}
int px = static_cast<int>(std::floor(prior.x));
int py = static_cast<int>(std::floor(prior.y));
if (px < 0 || px >= img.cols || py < 0 || py >= img.rows) {
    std::cerr << "[detectRTag] Prior out of bounds: " << prior << " for image size: "
              << img.cols << "x" << img.rows << std::endl;
    return {prior, cv::Mat::zeros(cv::Size(200, 200), CV_8UC3)};
}




// ROI calculation
cv::Rect roi = cv::Rect(prior.x - 100, prior.y - 100, 200, 200) &
               cv::Rect(0, 0, img.cols, img.rows);

if (roi.width == 0 || roi.height == 0) {
    std::cerr << "[detectRTag] ROI is zero-sized: " << roi << std::endl;
    return {prior, cv::Mat::zeros(200, 200, CV_8UC3)};
}

  // Create ROI

  const cv::Point2f prior_in_roi = prior - cv::Point2f(roi.tl());

  cv::Mat img_roi = img(roi);

  // Gray -> Binary -> Dilate
  cv::Mat gray_img;
  cv::cvtColor(img_roi, gray_img, cv::COLOR_BGR2GRAY);
  cv::Mat binary_img;
  cv::threshold(gray_img, binary_img, 0, 255,
                cv::THRESH_BINARY | cv::THRESH_OTSU);
  cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3));
  cv::dilate(binary_img, binary_img, kernel);

  // Find contours
  std::vector<std::vector<cv::Point>> contours;
  cv::findContours(binary_img, contours, cv::RETR_EXTERNAL,
                   cv::CHAIN_APPROX_NONE);

  auto it = std::find_if(
      contours.begin(), contours.end(),
      [p = prior_in_roi](const std::vector<cv::Point> &contour) -> bool {
        return cv::boundingRect(contour).contains(p);
      });

  // For visualization
  cv::cvtColor(binary_img, binary_img, cv::COLOR_GRAY2BGR);

  if (it == contours.end()) {
    return {prior, binary_img};
  }

  cv::drawContours(binary_img, contours, it - contours.begin(),
                   cv::Scalar(0, 255, 0), 2);

  cv::Point2f center = std::accumulate(it->begin(), it->end(), cv::Point(0, 0));
  center /= static_cast<float>(it->size());
  center += cv::Point2f(roi.tl());

  return {center, binary_img};
}
