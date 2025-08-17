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
#include "cuda_runtime_api.h"
#include "detect/armor_detect/armor_infer.hpp"
#include "wust_vl/common/utils/logger.hpp"
#include "wust_vl/common/utils/timer.hpp"
#include <cuda.h>
#include <device_launch_parameters.h>
#include <fstream>
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

// 构造函数：初始化参数并构建引擎
ArmorDetectTrt::ArmorDetectTrt(
    std::string model_type,
    const std::string& onnx_path,
    const Params& params,
    const ArmorDetectCommonParams& armor_detect_common_params,
    bool use_armor_detect_common
):
    params_(params),
    use_armor_detect_common_(use_armor_detect_common) {
    int device_id = params_.device_id;
    cudaSetDevice(device_id);
    auto model = armor_infer::modeFromString(model_type);
    armor_infer_ = std::make_unique<armor_infer::ArmorInfer>(
        model,
        params.conf_threshold,
        params.nms_threshold,
        params.top_k
    );
    trt_net_ = std::make_unique<ml_net::TensorRTNet>();
    ml_net::TensorRTNet::Params trt_params;
    trt_params.model_path = onnx_path;
    trt_net_->init(trt_params);
    auto input_output_dims = trt_net_->getInputOutputDims();
    input_dims_ = std::get<0>(input_output_dims);
    output_dims_ = std::get<1>(input_output_dims);
    strides_ = { 8, 16, 32 };
    armor_infer_->generate_grids_and_stride(
        armor_infer_->getInputW(),
        armor_infer_->getInputH(),
        strides_,
        grid_strides_
    );
    AdaptiveResourcePool<Infer>::Params pool_params;
    pool_params.resource_initializer = [=]() {
        std::vector<std::unique_ptr<Infer>> infers;
        for (int i = 0; i < params.max_infer_running; ++i) {
            auto infer = std::make_unique<Infer>();
            auto ctx = trt_net_->getAContext();
            infer->context = std::unique_ptr<nvinfer1::IExecutionContext>(ctx);
            // 初始化 CUDA 推理
            if (params_.use_cuda_pre || params_.use_cuda_post) {
                infer->cuda_infer = std::make_unique<armor_cuda_infer::CudaInfer>();
                size_t max_input_img = 4096 * 2160 * 3;
                size_t num_grid_strides = 0;
                auto* device_grid_strides = armor_cuda_infer::init_grid_strides_on_gpu(
                    armor_infer_->getInputW(),
                    armor_infer_->getInputH(),
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
        auto ctx = trt_net_->getAContext();
        infer->context = std::unique_ptr<nvinfer1::IExecutionContext>(ctx);
        if (params_.use_cuda_pre || params_.use_cuda_post) {
            infer->cuda_infer = std::make_unique<armor_cuda_infer::CudaInfer>();
            size_t max_input_img = 4096 * 2160 * 3;
            size_t num_grid_strides = 0;
            auto* device_grid_strides = armor_cuda_infer::init_grid_strides_on_gpu(
                armor_infer_->getInputW(),
                armor_infer_->getInputH(),
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

    //pool_params.thread_pool = thread_pool_;
    pool_params.logger = [](const std::string& msg) {
        WUST_INFO("ArmorDetectTrt:infer pool") << msg;
    };
    infer_pool_ = std::make_unique<AdaptiveResourcePool<Infer>>(pool_params);
    if (use_armor_detect_common_) {
        armor_detect_common_ = std::make_unique<ArmorDetectCommon>(armor_detect_common_params);
    }
}

ArmorDetectTrt::~ArmorDetectTrt() {
    if (infer_pool_) {
        infer_pool_.reset();
    }
    trt_net_.reset();
    armor_detect_common_.reset();
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
    void* input_tensor_ptr;
    if (infer->cuda_infer && params_.use_cuda_pre) {
        input_tensor_ptr = infer->cuda_infer->preprocess(
            frame.src_img.data,
            frame.src_img.cols,
            frame.src_img.rows,
            transform_matrix,
            trt_net_->getStream()
        );
    } else {
        cv::Mat resized_img = armor_infer_->letterbox(
            frame.src_img,
            transform_matrix,
            armor_infer_->getInputW(),
            armor_infer_->getInputH()
        );
        float scale = armor_infer_->getUseNorm() ? 255.0f : 1.0f;
        cv::Mat blob = cv::dnn::blobFromImage(
            resized_img,
            scale,
            cv::Size(armor_infer_->getInputW(), armor_infer_->getInputH()),
            cv::Scalar(0, 0, 0),
            true
        );
        trt_net_->input2Device(blob.ptr<float>());
        input_tensor_ptr = trt_net_->getInputTensorPtr();
    }
    auto t1 = time_utils::now();
    if (infer->context && input_tensor_ptr) {
        trt_net_->infer(input_tensor_ptr, infer->context.get());
    }
    auto t2 = time_utils::now();
    if (infer->cuda_infer && params_.use_cuda_post) {
        auto host_results = infer->cuda_infer->postprocess(
            (float*)trt_net_->getDeviceOutput(),
            grid_strides_.size(),
            transform_matrix,
            params_.conf_threshold,
            params_.nms_threshold,
            params_.top_k
        );
        buildCpuResult(host_results, objs_result);
    } else {
        cv::Mat output_mat(output_dims_.d[1], output_dims_.d[2], CV_32F, trt_net_->output2Host());
        objs_result = armor_infer_->postProcess(output_mat, transform_matrix, grid_strides_);
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
            auto detect_color = gobal::stringanything.get_value<int>("detect_color");
            if (detect_color == 0 && obj.color == armor::ArmorColor::BLUE) {
                continue;
            } else if (detect_color == 1 && obj.color == armor::ArmorColor::RED) {
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

void ArmorDetectTrt::pushInput(CommonFrame& frame) {
    if (infer_pool_) {
        auto infer_ptr = infer_pool_->acquire();
        if (infer_ptr != nullptr) {
            frame.id = current_id_++;
            this->processCallback(frame, infer_ptr);
            infer_pool_->release(infer_ptr);
        }
    }
}
