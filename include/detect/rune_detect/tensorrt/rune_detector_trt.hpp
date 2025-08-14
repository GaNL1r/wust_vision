// Copyright 2025 XiaoJian Wu
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
#pragma once

// std
#include <filesystem>
#include <functional>
#include <future>
#include <memory>
#include <string>
#include <tuple>
#include <vector>
// third party
#include <Eigen/Dense>
#include <opencv2/opencv.hpp>
#include <openvino/openvino.hpp>
// project
#include "NvInfer.h"
#include "NvInferRuntime.h"
#include "common/ThreadPool.h"
#include "common/adaptive_resource_pool.hpp"
#include "detect/rune_detect/rune_infer.hpp"
#include "rune_infer.hpp"
#include "type/type.hpp"

class RuneDetectorTrt {
public:
    using CallbackType = std::function<void(std::vector<rune::RuneObject>&, const CommonFrame&)>;
    struct Params {
        float conf_threshold = 0.3; // 置信度阈值
        float nms_threshold = 0.5; // NMS阈值
        int top_k = 128; // 最大检测框数
        int device_id = 0;
        int max_infer_running;
        double min_free_mem_ratio;
        bool use_cuda_pre = false;
        bool use_cuda_post = false;
        bool log_time = false;
    };
    struct Infer {
        std::unique_ptr<nvinfer1::IExecutionContext> context;
        std::unique_ptr<rune_cuda_infer::CudaInfer> cuda_infer;
    };

public:
    // Construct a new Trt Detector object
    explicit RuneDetectorTrt(
        std::string model_type,
        const std::filesystem::path& model_path,
        const Params& params
    );
    ~RuneDetectorTrt();

    // Push an inference request to the detector
    void pushInput(CommonFrame& frame);

    void setCallback(CallbackType callback);

    // Detect R tag using traditional method
    // Return the center of the R tag and binary roi image (for debug)
    std::tuple<cv::Point2f, cv::Mat>
    detectRTag(const cv::Mat& img, int binary_thresh, const cv::Point2f& prior, bool precise);

private:
    // Do inference and call the infer_callback_ after inference
    bool processCallback(const CommonFrame& frame, Infer* infer);
    void buildEngine(const std::string& onnx_path);

private:
    std::string model_path_;
    std::string device_name_;
    float conf_threshold_;
    int top_k_;
    float nms_threshold_;
    std::vector<int> strides_;
    std::vector<GridAndStride> grid_strides_;
    CallbackType infer_callback_;
    Params params_;
    nvinfer1::ICudaEngine* engine_;
    nvinfer1::IExecutionContext* context_;
    void* device_buffers_[2]; // 输入输出显存指针
    float* output_buffer_; // 输出数据主机内存
    cudaStream_t stream_; // CUDA流
    int input_idx_, output_idx_;
    size_t input_sz_, output_sz_;
    nvinfer1::Dims input_dims_;
    nvinfer1::Dims output_dims_;
    TRTLogger g_logger_;
    nvinfer1::IRuntime* runtime_ = nullptr;
    std::unique_ptr<AdaptiveResourcePool<Infer>> infer_pool_;
    std::unique_ptr<rune_infer::RuneInfer> rune_infer_;
    int current_id_ = 0;
};
