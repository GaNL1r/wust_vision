#include "detect/armor_detect/onnxruntime/armor_detector_ort.hpp"
#include "common/gobal.hpp"
#include "detect/armor_detect/armor_infer.hpp"
static void letterbox_into_CHW_uint8_rgb(
    const cv::Mat& img,
    float* output_ptr,
    Eigen::Matrix3f& transform_matrix,
    int dst_width,
    int dst_height
) {
    int img_h = img.rows;
    int img_w = img.cols;

    float scale = std::min(dst_height * 1.0f / img_h, dst_width * 1.0f / img_w);
    int resize_h = static_cast<int>(round(img_h * scale));
    int resize_w = static_cast<int>(round(img_w * scale));

    int pad_h = dst_height - resize_h;
    int pad_w = dst_width - resize_w;
    int top = static_cast<int>(round(pad_h / 2.0f - 0.1f));
    int left = static_cast<int>(round(pad_w / 2.0f - 0.1f));

    // Compute inverse affine transform matrix
    transform_matrix << 1.0f / scale, 0, -left / scale, 0, 1.0f / scale, -top / scale, 0, 0, 1;

    // Resize input image first
    cv::Mat resized;
    cv::resize(img, resized, cv::Size(resize_w, resize_h), 0, 0, cv::INTER_LINEAR);

    // Fill CHW uint8 RGB into output_ptr with padding (but still stored in float)
    for (int c = 0; c < 3; ++c) {
        for (int y = 0; y < dst_height; ++y) {
            for (int x = 0; x < dst_width; ++x) {
                int out_index = c * dst_height * dst_width + y * dst_width + x;
                if (y >= top && y < top + resize_h && x >= left && x < left + resize_w) {
                    int ry = y - top;
                    int rx = x - left;
                    cv::Vec3b pixel = resized.at<cv::Vec3b>(ry, rx); // BGR
                    output_ptr[out_index] = static_cast<float>(pixel[2 - c]); // R,G,B → CHW
                } else {
                    output_ptr[out_index] = 114.0f; // padding value (not normalized)
                }
            }
        }
    }
}
ArmorDetectOnnxRuntime::ArmorDetectOnnxRuntime(
    const std::filesystem::path& model_path,
    const ArmorDetectCommonParams& armor_detect_common_params,
    float conf_threshold,
    int top_k,
    float nms_threshold,
    bool use_gpu_,
    int device_id_,
    bool use_armor_detect_common
):

    model_path_(model_path),
    conf_threshold_(conf_threshold),
    top_k_(top_k),
    nms_threshold_(nms_threshold),
    use_gpu_(use_gpu_),
    device_id_(device_id_),
    use_armor_detect_common_(use_armor_detect_common) {
    if (use_armor_detect_common_) {
        armor_detect_common_ = std::make_unique<ArmorDetectCommon>(armor_detect_common_params);
    }

    init();
}

void ArmorDetectOnnxRuntime::init() {
    env_ = std::make_unique<Ort::Env>(ORT_LOGGING_LEVEL_WARNING, "ArmorDetectONNX");
    Ort::SessionOptions session_options;
    session_options.SetIntraOpNumThreads(4);

    session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);

    if (use_gpu_) {
        OrtSessionOptionsAppendExecutionProvider_CUDA(session_options, device_id_);
    }

    session_ = std::make_unique<Ort::Session>(*env_, model_path_.c_str(), session_options);

    Ort::AllocatorWithDefaultOptions allocator;

    Ort::AllocatedStringPtr input_name_ptr = session_->GetInputNameAllocated(0, allocator);
    input_name_ = std::string(input_name_ptr.get());

    auto input_type_info = session_->GetInputTypeInfo(0);
    auto input_tensor_info = input_type_info.GetTensorTypeAndShapeInfo();
    input_dims_ = input_tensor_info.GetShape();

    Ort::AllocatedStringPtr output_name_ptr = session_->GetOutputNameAllocated(0, allocator);
    output_name_ = std::string(output_name_ptr.get());

    strides_ = { 8, 16, 32 };
    armor_infer::generate_grids_and_stride(
        armor_infer::INPUT_W,
        armor_infer::INPUT_H,
        strides_,
        grid_strides_
    );
}

ArmorDetectOnnxRuntime::~ArmorDetectOnnxRuntime() {
    env_.reset();
    session_.reset();
    armor_detect_common_.reset();
}

void ArmorDetectOnnxRuntime::setCallback(DetectorCallback callback) {
    infer_callback_ = callback;
}
bool ArmorDetectOnnxRuntime::processCallback(const CommonFrame& frame) {
    Eigen::Matrix3f transform_matrix;

    std::vector<float> input_tensor_values(armor_infer::INPUT_W * armor_infer::INPUT_H * 3);
    letterbox_into_CHW_uint8_rgb(
        frame.src_img,
        input_tensor_values.data(),
        transform_matrix,
        armor_infer::INPUT_W,
        armor_infer::INPUT_H
    );

    Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
        memory_info,
        input_tensor_values.data(),
        input_tensor_values.size(),
        input_dims_.data(),
        input_dims_.size()
    );

    const char* input_names[] = { input_name_.c_str() };
    const char* output_names[] = { output_name_.c_str() };
    auto output_tensors =
        session_->Run(Ort::RunOptions { nullptr }, input_names, &input_tensor, 1, output_names, 1);

    float* output_data = output_tensors.front().GetTensorMutableData<float>();
    auto output_shape = output_tensors.front().GetTensorTypeAndShapeInfo().GetShape();
    int rows = static_cast<int>(output_shape[1]);
    int cols = static_cast<int>(output_shape[2]);
    cv::Mat output_buffer(rows, cols, CV_32F, output_data);

    // Parsed variable
    std::vector<armor::ArmorObject> objs_tmp, objs_result;
    std::vector<int> indices;

    objs_result = armor_infer::postProcess(
        objs_tmp,
        output_buffer,
        transform_matrix,
        grid_strides_,
        this->conf_threshold_,
        this->nms_threshold_,
        this->top_k_
    );
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

    return false;
}
void ArmorDetectOnnxRuntime::pushInput(const CommonFrame& frame) {
    processCallback(frame);
}