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
#include "type/type.hpp"

class RuneDetectorTrt {
public:
    using CallbackType = std::function<void(
        std::vector<RuneObject>&,
        std::chrono::steady_clock::time_point,
        const cv::Mat&,
        Eigen::Matrix4d T_camera_to_odom
    )>;
    struct Params {
        int input_w = 416; // 模型输入宽度
        int input_h = 416; // 模型输入高度
        int num_classes = 8; // 类别数 (0-7)
        int num_colors = 4; // 颜色数 (0-3)
        float conf_threshold = 0.3; // 置信度阈值
        float nms_threshold = 0.5; // NMS阈值
        int top_k = 128; // 最大检测框数
    };

public:
    // Construct a new Trt Detector object
    explicit RuneDetectorTrt(const std::filesystem::path& model_path, const Params& params);
    ~RuneDetectorTrt();

    // Push an inference request to the detector
    void pushInput(
        const cv::Mat& rgb_img,
        std::chrono::steady_clock::time_point timestamp,
        Eigen::Matrix4d T_camera_to_odom
    );

    void setCallback(CallbackType callback);

    // Detect R tag using traditional method
    // Return the center of the R tag and binary roi image (for debug)
    std::tuple<cv::Point2f, cv::Mat>
    detectRTag(const cv::Mat& img, int binary_thresh, const cv::Point2f& prior, bool precise);

private:
    // Do inference and call the infer_callback_ after inference
    bool processCallback(
        const cv::Mat resized_img,
        Eigen::Matrix3f transform_matrix,
        std::chrono::steady_clock::time_point timestamp,
        const cv::Mat& src_img,
        Eigen::Matrix4d T_camera_to_odom
    );
    void buildEngine(const std::string& onnx_path);
    std::vector<RuneObject> postprocess(
        std::vector<RuneObject>& output_objs,
        std::vector<float>& scores,
        std::vector<cv::Rect>& rects,
        const float* output,
        int num_detections,
        const Eigen::Matrix<float, 3, 3>& transform_matrix
    );
    bool extractImage(const cv::Mat& src, RuneObject& rune);
    void detectTarget(RuneObject& rune);

private:
    std::string model_path_;
    std::string device_name_;
    float conf_threshold_;
    int top_k_;
    float nms_threshold_;
    std::mutex mtx_;
    std::vector<int> strides_;
    std::vector<GridAndStride> grid_strides_;

    CallbackType infer_callback_;
    std::unique_ptr<ThreadPool> thread_pool_;

    Params params_;
    nvinfer1::ICudaEngine* engine_;
    nvinfer1::IExecutionContext* context_;
    void* device_buffers_[2]; // 输入输出显存指针
    float* output_buffer_; // 输出数据主机内存
    cudaStream_t stream_; // CUDA流
    int input_idx_, output_idx_;
    size_t input_sz_, output_sz_;
    // Eigen::Matrix3f transform_matrix; // 变换矩阵
    TRTLogger g_logger_;

    nvinfer1::IRuntime* runtime_ = nullptr;
};
