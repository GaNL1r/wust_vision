#include "common/ThreadPool.h"
#include "common/logger.hpp"
#include "detect/armor_detect/armor_detect_common.hpp"
#include "eigen3/Eigen/Dense"
#include "fmt/color.h"
#include "fmt/core.h"
#include "fmt/printf.h"
#include "opencv2/opencv.hpp"
#include <filesystem>
#include <onnxruntime_cxx_api.h>

class ArmorDetectOnnxRuntime {
public:
    using DetectorCallback =
        std::function<void(const std::vector<armor::ArmorObject>&, const CommonFrame&)>;

    explicit ArmorDetectOnnxRuntime(
        const std::filesystem::path& model_path,
        const std::string& classify_model_path,
        const std::string& classify_label_path,
        const armor::LightParams& l,
        const armor::ArmorParams& a,
        double classifier_threshold,
        float conf_threshold = 0.25,
        int top_k = 128,
        float nms_threshold = 0.3,
        float expand_ratio_w = 1.1f,
        float expand_ratio_h = 1.1f,
        int binary_thres_ = 85,
        bool use_gpu_ = false,
        int device_id_ = 0,
        bool use_armor_detect_common_ = true
    );

    ~ArmorDetectOnnxRuntime();

    void init();
    bool processCallback(
        const cv::Mat resized_img,
        Eigen::Matrix3f transform_matrix,
        const CommonFrame& frame
    );

    void drawResult(const cv::Mat& src_img, std::vector<armor::ArmorObject>& armor_objects);

    void pushInput(const CommonFrame& frame);

    void setCallback(DetectorCallback callback);

private:
    std::string model_path_;
    float conf_threshold_;
    int top_k_;
    float nms_threshold_;
    std::unique_ptr<Ort::Env> env_;
    std::unique_ptr<Ort::Session> session_;
    Ort::SessionOptions session_options_;

    std::vector<int64_t> input_dims_;
    std::string input_name_;
    std::string output_name_;

    std::vector<int> strides_;
    std::vector<GridAndStride> grid_strides_;
    DetectorCallback infer_callback_;
    std::unique_ptr<ArmorDetectCommon> armor_detect_common_;
    bool use_gpu_ = false;
    int device_id_ = 0;
    bool use_armor_detect_common_ = true;
};