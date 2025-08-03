// Copyright 2025 Zikang Xie
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

#include "detect/armor_detect/tensorrt/armor_detector_trt.hpp"
#include "NvOnnxParser.h"
#include "common/gobal.hpp"
#include "common/logger.hpp"
#include "common/timer.hpp"
#include "cuda_runtime_api.h"
#include <cuda.h>
#include <device_launch_parameters.h>
#include <fstream>
// #include <logger.h>
#define TRT_ASSERT(expr) \
    do { \
        if (!(expr)) { \
            fmt::print(fmt::fg(fmt::color::red), "assert fail: '" #expr "'"); \
            exit(-1); \
        } \
    } while (0)
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            std::cerr << "CUDA error: " << cudaGetErrorString(err) << " at " << __FILE__ << ":" \
                      << __LINE__ << std::endl; \
            return nullptr; \
        } \
    } while (0)

static const int INPUT_W = 416; // Width of input
static const int INPUT_H = 416; // Height of input
static constexpr int NUM_CLASSES = 8; // Number of classes
static constexpr int NUM_COLORS = 4; // Number of color
static constexpr float MERGE_CONF_ERROR = 0.15;
static constexpr float MERGE_MIN_IOU = 0.9;
// 辅助函数：生成网格和步长

static void
generate_grids_and_stride(std::vector<int>& strides, std::vector<GridAndStride>& grid_strides) {
    for (auto stride: strides) {
        int num_grid_w = 416 / stride;
        int num_grid_h = 416 / stride;
        for (int g1 = 0; g1 < num_grid_h; g1++) {
            for (int g0 = 0; g0 < num_grid_w; g0++) {
                grid_strides.push_back(GridAndStride { g0, g1, stride });
            }
        }
    }
}
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
/**
 * @brief Calculate intersection area between two objects.
 * @param a Object a.
 * @param b Object b.
 * @return Area of intersection.
 */
static inline float intersection_area(const armor::ArmorObject& a, const armor::ArmorObject& b) {
    cv::Rect_<float> inter = a.box & b.box;
    return inter.area();
}

static void nms_merge_sorted_bboxes(
    std::vector<armor::ArmorObject>& faceobjects,
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
        armor::ArmorObject& a = faceobjects[i];

        int keep = 1;
        for (int indice: indices) {
            armor::ArmorObject& b = faceobjects[indice];

            // intersection over union
            float inter_area = intersection_area(a, b);
            float union_area = areas[i] + areas[indice] - inter_area;
            float iou = inter_area / union_area;
            if (iou > nms_threshold || isnan(iou)) {
                keep = 0;
                // Stored for Merge
                if (a.number == b.number && a.color == b.color && iou > MERGE_MIN_IOU
                    && abs(a.prob - b.prob) < MERGE_CONF_ERROR)
                {
                    for (int i = 0; i < 4; i++) {
                        b.pts.push_back(a.pts[i]);
                    }
                }
                // cout<<b.pts_x.size()<<endl;
            }
        }

        if (keep) {
            indices.push_back(i);
        }
    }
}

// 构造函数：初始化参数并构建引擎
ArmorDetectTrt::ArmorDetectTrt(
    const std::string& onnx_path,
    const Params& params,
    const ArmorDetectCommonParams& armor_detect_common_params,
    bool use_armor_detect_common

):
    params_(params),
    engine_(nullptr),
    context_(nullptr),
    output_buffer_(nullptr),
    runtime_(nullptr),
    use_armor_detect_common_(use_armor_detect_common) {
    int device_id = params_.device_id;
    cudaSetDevice(device_id);
    buildEngine(onnx_path);
    TRT_ASSERT(context_ = engine_->createExecutionContext());
    TRT_ASSERT((input_idx_ = engine_->getBindingIndex("images")) == 0);
    TRT_ASSERT((output_idx_ = engine_->getBindingIndex("output")) == 1);

    auto input_dims = engine_->getBindingDimensions(input_idx_);
    auto output_dims = engine_->getBindingDimensions(output_idx_);
    input_sz_ = input_dims.d[1] * input_dims.d[2] * input_dims.d[3];
    output_sz_ = output_dims.d[1] * output_dims.d[2];
    TRT_ASSERT(cudaMalloc(&device_buffers_[input_idx_], input_sz_ * sizeof(float)) == 0);
    TRT_ASSERT(cudaMalloc(&device_buffers_[output_idx_], output_sz_ * sizeof(float)) == 0);
    output_buffer_ = new float[output_sz_];
    TRT_ASSERT(cudaStreamCreate(&stream_) == 0);
    std::vector<int> strides = { 8, 16, 32 };
    AdaptiveResourcePool<Infer>::Params pool_params;
    pool_params.resource_initializer = [=]() {
        std::vector<std::unique_ptr<Infer>> infers;
        for (int i = 0; i < params.max_infer_running; ++i) {
            auto infer = std::make_unique<Infer>();
            auto ctx = engine_->createExecutionContext();
            infer->context = std::unique_ptr<nvinfer1::IExecutionContext>(ctx);
            infer->cuda_infer = std::make_unique<armor_cuda_infer::CudaInfer>();
            size_t max_input_img = 4096 * 2160 * 3;
            size_t num_grid_strides = 0;
            armor_cuda_infer::GPUGridAndStride* device_grid_strides =
                armor_cuda_infer::init_grid_strides_on_gpu(
                    INPUT_W,
                    INPUT_W,
                    strides,
                    num_grid_strides
                );
            infer->cuda_infer
                ->init(device_grid_strides, max_input_img, output_sz_ / 21, num_grid_strides);
            if (!infer->context || !infer->cuda_infer) {
                WUST_ERROR("TRT") << "create infer failed"
                                  << "index:" << i;
                continue;
            }
            size_t free_mem, total_mem;
            cudaMemGetInfo(&free_mem, &total_mem);
            WUST_DEBUG("TRT") << "Free GPU memory:" << free_mem / 1024.0 / 1024.0 << "MB"
                              << "Total GPU memory:" << total_mem / 1024.0 / 1024.0 << "MB";
            double free_mem_ratio = static_cast<double>(free_mem) / static_cast<double>(total_mem);
            if (free_mem_ratio < params.min_free_mem_ratio) {
                WUST_WARN("TRT") << "GPU memory is not enough!"
                                 << "Free GPU memory:" << free_mem_ratio * 100 << "%";
                WUST_INFO("TRT") << "Cut remaining infer";
                break;
            }
            infers.emplace_back(std::move(infer));
            WUST_INFO("TRT") << "create execution context success"
                             << "index:" << i;
        }
        return infers;
    };
    auto release_func = [](std::unique_ptr<Infer>& resource) {
        if (resource) {
        }
    };
    auto restore_func = [=](size_t idx) -> std::unique_ptr<Infer> {
        auto infer = std::make_unique<Infer>();
        auto ctx = engine_->createExecutionContext();
        infer->context = std::unique_ptr<nvinfer1::IExecutionContext>(ctx);
        infer->cuda_infer = std::make_unique<armor_cuda_infer::CudaInfer>();
        size_t max_input_img = 4096 * 2160 * 3;
        size_t num_grid_strides = 0;
        armor_cuda_infer::GPUGridAndStride* device_grid_strides =
            armor_cuda_infer::init_grid_strides_on_gpu(INPUT_W, INPUT_W, strides, num_grid_strides);
        infer->cuda_infer
            ->init(device_grid_strides, max_input_img, output_sz_ / 21, num_grid_strides);
        if (!infer->context || !infer->cuda_infer) {
            WUST_ERROR("TRT") << "create infer failed";
            return nullptr;
        }
        return infer;
    };
    pool_params.restore_func = restore_func;

    pool_params.release_func = release_func;

    pool_params.can_restore = [=](size_t active_count) {
        size_t free_mem = 0, total_mem = 0;
        cudaMemGetInfo(&free_mem, &total_mem);

        double free_ratio = static_cast<double>(free_mem) / total_mem;
        size_t used_mem = total_mem - free_mem;
        size_t avg_used_per_resource = active_count > 0 ? used_mem / active_count : 1;
        size_t safe_margin = avg_used_per_resource;

        bool enough_for_one_more = free_mem > (avg_used_per_resource + safe_margin);

        return free_ratio > params_.min_free_mem_ratio * 1.2 && enough_for_one_more;
    };

    pool_params.should_release = [=](size_t active_count) {
        size_t free_mem = 0, total_mem = 0;
        cudaMemGetInfo(&free_mem, &total_mem);
        double free_ratio = static_cast<double>(free_mem) / total_mem;
        return free_ratio < params_.min_free_mem_ratio && active_count > 1;
    };

    thread_pool_ = std::make_shared<ThreadPool>(params.max_infer_running);
    pool_params.thread_pool = thread_pool_;
    pool_params.logger = [](const std::string& msg) {
        WUST_INFO("ArmorDetectTrt:infer pool") << msg;
    };
    infer_pool_ = std::make_unique<AdaptiveResourcePool<Infer>>(pool_params);
    if (use_armor_detect_common_) {
        armor_detect_common_ = std::make_unique<ArmorDetectCommon>(armor_detect_common_params);
    }
}

ArmorDetectTrt::~ArmorDetectTrt() {
    delete[] output_buffer_;
    cudaStreamDestroy(stream_);
    cudaFree(device_buffers_[output_idx_]);
    cudaFree(device_buffers_[input_idx_]);
    if (infer_pool_) {
        infer_pool_.reset();
    }
    if (context_)
        context_->destroy();

    if (engine_)
        engine_->destroy();
    if (runtime_)
        runtime_->destroy();
    if (thread_pool_) {
        thread_pool_->waitUntilEmpty();
        thread_pool_.reset();
    }
}

void ArmorDetectTrt::buildEngine(const std::string& onnx_path) {
    std::string engine_path = onnx_path.substr(0, onnx_path.find_last_of('.')) + ".engine";
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
            WUST_INFO("TRT") << "Load engine from " << engine_path << " successfully.";
            return;
        }
    }
    WUST_INFO("TRT") << "building new engine...";
    // 构建新引擎
    auto builder = nvinfer1::createInferBuilder(g_logger_);
    const auto explicit_batch = 1U
        << static_cast<uint32_t>(nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
    auto network = builder->createNetworkV2(explicit_batch);
    auto parser = nvonnxparser::createParser(*network, g_logger_);
    parser->parseFromFile(onnx_path.c_str(), static_cast<int>(nvinfer1::ILogger::Severity::kINFO));

    auto config = builder->createBuilderConfig();
    if (builder->platformHasFastFp16()) {
        config->setFlag(nvinfer1::BuilderFlag::kFP16);
    }
    engine_ = builder->buildEngineWithConfig(*network, *config);

    // 保存引擎
    auto serialized_engine = engine_->serialize();
    std::ofstream out_file(engine_path, std::ios::binary);
    out_file.write(
        reinterpret_cast<const char*>(serialized_engine->data()),
        serialized_engine->size()
    );
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

void ArmorDetectTrt::setCallback(DetectorCallback callback) {
    infer_callback_ = callback;
}
static void buildCpuResult(
    const std::vector<armor_cuda_infer::GPUArmorObject>& host_results,
    std::vector<armor::ArmorObject>& objs_result
) {
    for (const auto& gobj: host_results) {
        if (gobj.valid == 0)
            continue;

        armor::ArmorObject obj;
        obj.prob = gobj.confidence;
        obj.color = static_cast<armor::ArmorColor>(gobj.color_id);
        obj.number = static_cast<armor::ArmorNumber>(gobj.number_id);

        int n = std::max(4, gobj.num_pts); // 防止 num_pts 没填（兼容性）
        cv::Point2f avg_pts[4] = {};

        for (int i = 0; i < n; ++i) {
            int idx = i % 4;
            avg_pts[idx].x += gobj.x[i];
            avg_pts[idx].y += gobj.y[i];
        }

        obj.pts.resize(4);
        for (int i = 0; i < 4; ++i) {
            obj.pts[i].x = avg_pts[i].x / (n / 4.0f);
            obj.pts[i].y = avg_pts[i].y / (n / 4.0f);
        }

        obj.box = cv::boundingRect(obj.pts);
        objs_result.push_back(std::move(obj));
    }
}
// 推理函数
bool ArmorDetectTrt::processCallback(const CommonFrame& frame, Infer* infer) {
    auto t0 = time_utils::now();
    Eigen::Matrix3f transform_matrix;
    std::vector<armor::ArmorObject> objs_tmp, objs_result;
    std::vector<cv::Rect> rects;
    std::vector<float> scores;
    void* input_tensor_ptr;
    if (infer->cuda_infer && params_.use_cuda_pre) {
        input_tensor_ptr = infer->cuda_infer->preprocess(
            frame.src_img.data,
            frame.src_img.cols,
            frame.src_img.rows,
            transform_matrix,
            stream_
        );
    } else {
        cv::Mat resized_img = letterbox(frame.src_img, transform_matrix);
        cv::Mat blob = cv::dnn::blobFromImage(
            resized_img,
            1.,
            cv::Size(INPUT_W, INPUT_H),
            cv::Scalar(0, 0, 0),
            true
        );
        // 拷贝数据到显存
        cudaMemcpyAsync(
            device_buffers_[input_idx_],
            blob.ptr<float>(),
            input_sz_ * sizeof(float),
            cudaMemcpyHostToDevice,
            stream_
        );
        input_tensor_ptr = device_buffers_[input_idx_];
    }
    auto t1 = time_utils::now();
    if (infer->context && input_tensor_ptr) {
        infer->context->setTensorAddress("images", input_tensor_ptr);
        infer->context->setTensorAddress("output", device_buffers_[output_idx_]);

        if (!infer->context->enqueueV3(stream_)) {
            std::cerr << "enqueueV3 failed!";
            return {};
        }
    }
    auto t2 = time_utils::now();
    if (infer->cuda_infer && params_.use_cuda_post) {
        auto host_results = infer->cuda_infer->postprocess(
            (float*)device_buffers_[output_idx_],
            output_sz_ / 21,
            transform_matrix,
            params_.conf_threshold,
            params_.nms_threshold,
            params_.top_k
        );
        buildCpuResult(host_results, objs_result);
    } else {
        cudaMemcpyAsync(
            output_buffer_,
            device_buffers_[output_idx_],
            output_sz_ * sizeof(float),
            cudaMemcpyDeviceToHost,
            stream_
        );
        cudaStreamSynchronize(stream_);
        objs_result =
            postProcess(objs_tmp, scores, rects, output_buffer_, output_sz_ / 21, transform_matrix);
    }
    auto t3 = time_utils::now();
    if (params_.log_time) {
        WUST_INFO("TRT") << std::fixed << std::setprecision(3) << "pre "
                         << time_utils::durationMs(t0, t1) << " "
                         << "infer " << time_utils::durationMs(t1, t2) << " "
                         << "post " << time_utils::durationMs(t2, t3) << " "
                         << "total " << time_utils::durationMs(t0, t3);
    }

    std::vector<armor::ArmorObject> armors;
    if (use_armor_detect_common_) {
        armors = armor_detect_common_->detectNet(frame.src_img, objs_result);
        // Call callback function
        if (this->infer_callback_) {
            this->infer_callback_(armors, frame);

            return true;
        }
    } else {
        for (auto obj: objs_result) {
            if (gobal::detect_color == 0 && obj.color == armor::ArmorColor::BLUE) {
                continue;
            } else if (gobal::detect_color == 1 && obj.color == armor::ArmorColor::RED) {
                continue;
            }
        }
        if (this->infer_callback_) {
            this->infer_callback_(armors, frame);
            return true;
        }
    }

    return true;
}

std::vector<armor::ArmorObject> ArmorDetectTrt::postProcess(
    std::vector<armor::ArmorObject>& output_objs,
    std::vector<float>& scores,
    std::vector<cv::Rect>& rects,
    const float* output,
    int num_detections,
    const Eigen::Matrix<float, 3, 3>& transform_matrix
) {
    std::vector<int> strides = { 8, 16, 32 };
    std::vector<GridAndStride> grid_strides;
    generate_grids_and_stride(strides, grid_strides);

    for (int i = 0; i < num_detections; ++i) {
        const float* det = output + i * 21;
        float conf = det[8];

        if (conf < params_.conf_threshold)
            continue;

        // 解析坐标
        int grid0 = grid_strides[i].grid0;
        int grid1 = grid_strides[i].grid1;
        int stride = grid_strides[i].stride;

        cv::Point color_id, num_id;

        float x_1 = (det[0] + grid0) * stride;
        float y_1 = (det[1] + grid1) * stride;
        float x_2 = (det[2] + grid0) * stride;
        float y_2 = (det[3] + grid1) * stride;
        float x_3 = (det[4] + grid0) * stride;
        float y_3 = (det[5] + grid1) * stride;
        float x_4 = (det[6] + grid0) * stride;
        float y_4 = (det[7] + grid1) * stride;

        Eigen::Matrix<float, 3, 4> apex_norm;
        Eigen::Matrix<float, 3, 4> apex_dst;

        apex_norm << x_1, x_2, x_3, x_4, y_1, y_2, y_3, y_4, 1, 1, 1, 1;

        apex_dst = transform_matrix * apex_norm;

        armor::ArmorObject obj;

        obj.pts.resize(4);

        obj.pts[0] = cv::Point2f(apex_dst(0, 0), apex_dst(1, 0));
        obj.pts[1] = cv::Point2f(apex_dst(0, 1), apex_dst(1, 1));
        obj.pts[2] = cv::Point2f(apex_dst(0, 2), apex_dst(1, 2));
        obj.pts[3] = cv::Point2f(apex_dst(0, 3), apex_dst(1, 3));

        auto rect = cv::boundingRect(obj.pts);

        obj.box = rect;
        // obj.color = static_cast<ArmorColor>(color_id.x);
        // obj.number = static_cast<ArmorNumber>(num_id.x);
        obj.prob = conf;

        // 解析颜色和类别
        obj.color = static_cast<armor::ArmorColor>(
            std::max_element(det + 9, det + 9 + NUM_COLORS) - (det + 9)
        );
        obj.number = static_cast<armor::ArmorNumber>(
            std::max_element(det + 9 + NUM_COLORS, det + 9 + NUM_COLORS + NUM_CLASSES)
            - (det + 9 + NUM_COLORS)
        );
        // box.confidence = conf;

        rects.push_back(rect);
        scores.push_back(conf);
        output_objs.push_back(std::move(obj));
    }

    // TopK
    std::sort(
        output_objs.begin(),
        output_objs.end(),
        [](const armor::ArmorObject& a, const armor::ArmorObject& b) { return a.prob > b.prob; }
    );
    if (output_objs.size() > static_cast<size_t>(params_.top_k)) {
        output_objs.resize(params_.top_k);
    }
    std::vector<int> indices;
    std::vector<armor::ArmorObject> objs_result;
    // cv::dnn::NMSBoxes(rects, scores, params_.conf_threshold,
    // params_.nms_threshold, indices);
    nms_merge_sorted_bboxes(output_objs, indices, params_.nms_threshold);

    for (size_t i = 0; i < indices.size(); i++) {
        objs_result.push_back(std::move(output_objs[indices[i]]));

        if (objs_result[i].pts.size() >= 8) {
            auto n = objs_result[i].pts.size();
            cv::Point2f pts_final[4];

            for (size_t j = 0; j < n; j++) {
                pts_final[j % 4] += objs_result[i].pts[j];
            }

            objs_result[i].pts.resize(4);
            for (int j = 0; j < 4; j++) {
                pts_final[j].x /= static_cast<float>(n) / 4.0;
                pts_final[j].y /= static_cast<float>(n) / 4.0;
                objs_result[i].pts[j] = pts_final[j];
            }
        }
    }

    return objs_result;
}

void ArmorDetectTrt::pushInput(const CommonFrame& frame) {
    if (infer_pool_) {
        infer_pool_->enqueue([this, frame = std::move(frame)](Infer& infer) {
            this->processCallback(frame, &infer);
        });
    }
}
