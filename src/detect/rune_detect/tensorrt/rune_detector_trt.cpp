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

#include "detect/rune_detect/tensorrt/rune_detector_trt.hpp"
// std
#include <algorithm>
#include <fmt/color.h>
#include <numeric>
#include <unordered_map>
// third party
#include <opencv2/imgproc.hpp>
// project
#include "NvOnnxParser.h"
#include "common/gobal.hpp"
#include "common/logger.hpp"
#include "common/timer.hpp"
#include "detect/rune_detect/rune_infer.hpp"
#include "fmt/core.h"
#include "type/type.hpp"
#define TRT_ASSERT(expr) \
    do { \
        if (!(expr)) { \
            fmt::print(fmt::fg(fmt::color::red), "assert fail: '" #expr "'"); \
            exit(-1); \
        } \
    } while (0)

RuneDetectorTrt::RuneDetectorTrt(
    std::string model_type,
    const std::filesystem::path& model_path,
    const Params& params
):
    params_(params),
    model_path_(model_path) {
    int device_id = params_.device_id;
    cudaSetDevice(device_id);
    buildEngine(model_path);
    TRT_ASSERT(context_ = engine_->createExecutionContext());
    TRT_ASSERT((input_idx_ = engine_->getBindingIndex("images")) == 0);
    TRT_ASSERT((output_idx_ = engine_->getBindingIndex("output")) == 1);
    auto model = rune_infer::modeFromString(model_type);
    rune_infer_ = std::make_unique<rune_infer::RuneInfer>(
        model,
        params.conf_threshold,
        params.nms_threshold,
        params.top_k
    );
    input_dims_ = engine_->getBindingDimensions(input_idx_);
    output_dims_ = engine_->getBindingDimensions(output_idx_);
    input_sz_ = input_dims_.d[1] * input_dims_.d[2] * input_dims_.d[3];
    output_sz_ = output_dims_.d[1] * output_dims_.d[2];
    TRT_ASSERT(cudaMalloc(&device_buffers_[input_idx_], input_sz_ * sizeof(float)) == 0);
    TRT_ASSERT(cudaMalloc(&device_buffers_[output_idx_], output_sz_ * sizeof(float)) == 0);
    output_buffer_ = new float[output_sz_];
    TRT_ASSERT(cudaStreamCreate(&stream_) == 0);
    strides_ = { 8, 16, 32 };
    rune_infer_->generateGridsAndStride(
        rune_infer_->getInputW(),
        rune_infer_->getInputH(),
        strides_,
        grid_strides_
    );
    AdaptiveResourcePool<Infer>::Params pool_params;
    pool_params.resource_initializer = [=]() {
        std::vector<std::unique_ptr<Infer>> infers;
        for (int i = 0; i < params.max_infer_running; ++i) {
            auto infer = std::make_unique<Infer>();
            auto ctx = engine_->createExecutionContext();
            infer->context = std::unique_ptr<nvinfer1::IExecutionContext>(ctx);
            if (params_.use_cuda_pre || params_.use_cuda_post) {
                infer->cuda_infer = std::make_unique<rune_cuda_infer::CudaInfer>();
                size_t max_input_img = 4096 * 2160 * 3;
                size_t num_grid_strides = 0;
                rune_cuda_infer::GPUGridAndStride* device_grid_strides =
                    rune_cuda_infer::init_grid_strides_on_gpu(
                        rune_infer_->getInputW(),
                        rune_infer_->getInputH(),
                        strides_,
                        num_grid_strides
                    );
                infer->cuda_infer->init(
                    device_grid_strides,
                    max_input_img,
                    grid_strides_.size(),
                    num_grid_strides
                );
            }
            if (!infer->context) {
                WUST_ERROR("TRT") << "create infer failed, missing context"
                                  << " index:" << i;
                continue;
            }
            if ((params_.use_cuda_pre || params_.use_cuda_post) && !infer->cuda_infer) {
                WUST_ERROR("TRT") << "create infer failed, missing cuda_infer"
                                  << " index:" << i;
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
            if (resource->cuda_infer) {
                resource->cuda_infer.reset();
            }
        }
    };
    auto restore_func = [=](size_t idx) -> std::unique_ptr<Infer> {
        auto infer = std::make_unique<Infer>();
        auto ctx = engine_->createExecutionContext();
        infer->context = std::unique_ptr<nvinfer1::IExecutionContext>(ctx);
        if (params_.use_cuda_pre || params_.use_cuda_post) {
            infer->cuda_infer = std::make_unique<rune_cuda_infer::CudaInfer>();
            size_t max_input_img = 4096 * 2160 * 3;
            size_t num_grid_strides = 0;
            rune_cuda_infer::GPUGridAndStride* device_grid_strides =
                rune_cuda_infer::init_grid_strides_on_gpu(
                    rune_infer_->getInputW(),
                    rune_infer_->getInputH(),
                    strides_,
                    num_grid_strides
                );
            infer->cuda_infer
                ->init(device_grid_strides, max_input_img, grid_strides_.size(), num_grid_strides);
        }
        if (!infer->context) {
            WUST_ERROR("TRT") << "create infer failed, missing context";
            return nullptr;
        }
        if ((params_.use_cuda_pre || params_.use_cuda_post) && !infer->cuda_infer) {
            WUST_ERROR("TRT") << "create infer failed, missing cuda_infer";
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

    pool_params.logger = [](const std::string& msg) {
        WUST_INFO("RuneDetectorTrt:infer pool") << msg;
    };
    infer_pool_ = std::make_unique<AdaptiveResourcePool<Infer>>(pool_params);
}
void RuneDetectorTrt::buildEngine(const std::string& onnx_path) {
    std::string engine_path = onnx_path.substr(0, onnx_path.find_last_of('.')) + ".engine";
    std::ifstream engine_file(engine_path, std::ios::binary);
    if (engine_file.good()) {
        engine_file.seekg(0, std::ios::end);
        size_t size = engine_file.tellg();
        engine_file.seekg(0, std::ios::beg);
        std::vector<char> engine_data(size);
        engine_file.read(engine_data.data(), size);
        engine_file.close();

        runtime_ = nvinfer1::createInferRuntime(g_logger_);
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

RuneDetectorTrt::~RuneDetectorTrt() {
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
}

void RuneDetectorTrt::setCallback(CallbackType callback) {
    infer_callback_ = callback;
}
static void buildCpuResult(
    const std::vector<rune_cuda_infer::GPURuneObject>& host_results,
    std::vector<rune::RuneObject>& objs_result
) {
    for (const auto& gobj: host_results) {
        if (gobj.valid == 0)
            continue; // 过滤无效目标

        rune::RuneObject robj;
        robj.prob = gobj.confidence;
        robj.color = rune_infer::DNN_COLOR_TO_ENEMY_COLOR[gobj.color_id];
        robj.type = static_cast<rune::RuneType>(gobj.type_id);

        // 填充 FeaturePoints
        rune::FeaturePoints pts;
        pts.r_center = cv::Point2f(gobj.x[0], gobj.y[0]);
        pts.bottom_left = cv::Point2f(gobj.x[1], gobj.y[1]);
        pts.top_left = cv::Point2f(gobj.x[2], gobj.y[2]);
        pts.top_right = cv::Point2f(gobj.x[3], gobj.y[3]);
        pts.bottom_right = cv::Point2f(gobj.x[4], gobj.y[4]);

        robj.pts = pts;
        robj.box = cv::boundingRect(pts.toVector2f());

        objs_result.push_back(std::move(robj));
    }
}
bool RuneDetectorTrt::processCallback(const CommonFrame& frame, Infer* infer) {
    auto t0 = time_utils::now();
    Eigen::Matrix3f transform_matrix;
    std::vector<rune::RuneObject> objs_tmp, objs_result;
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
        cv::Mat resized_img = rune_infer_->letterbox(
            frame.src_img,
            transform_matrix,
            rune_infer_->getInputW(),
            rune_infer_->getInputH()
        );
        float scale = rune_infer_->getUseNorm() ? 255.0f : 1.0f;
        cv::Mat blob = cv::dnn::blobFromImage(
            resized_img,
            scale,
            cv::Size(rune_infer_->getInputW(), rune_infer_->getInputH()),
            cv::Scalar(0, 0, 0),
            true
        );
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
            grid_strides_.size(),
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
        cv::Mat output_mat(output_dims_.d[1], output_dims_.d[2], CV_32F, output_buffer_);
        objs_result = rune_infer_->postProcess(output_mat, transform_matrix, grid_strides_);
    }

    auto t3 = time_utils::now();
    if (params_.log_time) {
        WUST_INFO("TRT") << std::fixed << std::setprecision(3) << "pre "
                         << time_utils::durationMs(t0, t1) << " "
                         << "infer " << time_utils::durationMs(t1, t2) << " "
                         << "post " << time_utils::durationMs(t2, t3) << " "
                         << "total " << time_utils::durationMs(t0, t3);
    }

    // objs_result.erase(
    //     std::remove_if(
    //         objs_result.begin(),
    //         objs_result.end(),
    //         [c = static_cast<EnemyColor>(gobal::detect_color)](const auto& objs_result) {
    //             return objs_result.color != c;
    //         }
    //     ),
    //     objs_result.end()
    // );

    // Call callback function
    if (this->infer_callback_) {
        this->infer_callback_(objs_result, frame);
        return true;
    }

    return false;
}

std::tuple<cv::Point2f, cv::Mat> RuneDetectorTrt::detectRTag(
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
void RuneDetectorTrt::pushInput(CommonFrame& frame) {
    if (infer_pool_) {
        auto infer_ptr = infer_pool_->acquire();
        if (infer_ptr != nullptr) {
            frame.id = current_id_++;
            this->processCallback(frame, infer_ptr);
            infer_pool_->release(infer_ptr);
        }
    }
}