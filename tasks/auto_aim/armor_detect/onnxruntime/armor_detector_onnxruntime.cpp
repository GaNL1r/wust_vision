#include "armor_detector_onnxruntime.hpp"
#include "Eigen/Dense"
#include "tasks/auto_aim/armor_detect/armor_infer.hpp"
#include <algorithm>
#include <array>
#include <cmath>
#include <opencv2/opencv.hpp>

ArmorDetectOnnxRuntime::ArmorDetectOnnxRuntime(
    std::string provider,
    std::string model_type,
    const std::filesystem::path& model_path,
    const ArmorDetectCommonParams& armor_detect_common_params,
    float conf_threshold,
    int top_k,
    float nms_threshold,
    bool use_armor_detect_common
):

    model_path_(model_path),
    conf_threshold_(conf_threshold),
    top_k_(top_k),
    nms_threshold_(nms_threshold),
    use_armor_detect_common_(use_armor_detect_common) {
    if (use_armor_detect_common_) {
        armor_detect_common_ = std::make_unique<ArmorDetectCommon>(armor_detect_common_params);
    }
    auto model = armor_infer::modeFromString(model_type);
    armor_infer_ =
        std::make_unique<armor_infer::ArmorInfer>(model, conf_threshold, nms_threshold, top_k);
    provider_ = string2OrtProvider(provider);
    init();
}

void ArmorDetectOnnxRuntime::init() {
    onnxruntime_net_ = std::make_unique<ml_net::OnnxRuntimeNet>();
    ml_net::OnnxRuntimeNet::Params params;
    params.model_path = model_path_;
    params.provider = provider_;
    onnxruntime_net_->init(params);

    // 9. 准备 YOLO 网格和步幅
    strides_ = { 8, 16, 32 };
    armor_infer_->generate_grids_and_stride(
        armor_infer_->getInputW(),
        armor_infer_->getInputH(),
        strides_,
        grid_strides_
    );
}

ArmorDetectOnnxRuntime::~ArmorDetectOnnxRuntime() {
    onnxruntime_net_.reset();
    armor_detect_common_.reset();
}

void ArmorDetectOnnxRuntime::setCallback(DetectorCallback callback) {
    infer_callback_ = callback;
}
bool ArmorDetectOnnxRuntime::processCallback(const CommonFrame& frame) {
    Eigen::Matrix3f transform_matrix;
    auto roi = frame.src_img(frame.expanded);
    cv::Mat resized_img = armor_infer_->letterbox(
        roi,
        transform_matrix,
        armor_infer_->getInputW(),
        armor_infer_->getInputH()
    );
    cv::imshow("resized_img", resized_img);
    cv::waitKey(1);
    float scale = armor_infer_->getUseNorm() ? 1.0f / 255.0f : 1.0f;
    cv::Mat blob = cv::dnn::blobFromImage(
        resized_img,
        scale,
        cv::Size(armor_infer_->getInputW(), armor_infer_->getInputH()),
        cv::Scalar(0, 0, 0),
        true
    );

    auto output_data = onnxruntime_net_->infer(blob.ptr<float>(), blob.total());

    auto output_shape = onnxruntime_net_->getOutputShape();
    int rows = static_cast<int>(output_shape[1]);
    int cols = static_cast<int>(output_shape[2]);
    cv::Mat output_buffer(rows, cols, CV_32F, output_data);

    // Parsed variable
    std::vector<armor::ArmorObject> objs_result;
    objs_result = armor_infer_->postProcess(output_buffer, transform_matrix, grid_strides_);
    std::vector<armor::ArmorObject> armors;
    if (use_armor_detect_common_) {
        armors = armor_detect_common_
                     ->detectNet(resized_img, objs_result, transform_matrix, frame.detect_color);
        // Call callback function
        if (this->infer_callback_) {
            this->infer_callback_(armors, frame);
            return true;
        }
    } else {
        for (auto obj: objs_result) {
            auto detect_color = frame.detect_color;
            if (detect_color == 0 && obj.color == armor::ArmorColor::BLUE) {
                continue;
            } else if (detect_color == 1 && obj.color == armor::ArmorColor::RED) {
                continue;
            }
            obj.transform(transform_matrix);
            armors.push_back(obj);
        }
        if (this->infer_callback_) {
            this->infer_callback_(armors, frame);
            return true;
        }
    }

    return false;
}
void ArmorDetectOnnxRuntime::pushInput(CommonFrame& frame) {
    frame.id = current_id_++;
    processCallback(frame);
}