#include "detect/armor_detect/onnxruntime/armor_detector_ort.hpp"
#include "Eigen/Dense"
#include "common/gobal.hpp"
#include "detect/armor_detect/armor_infer.hpp"
#include <algorithm>
#include <array>
#include <cmath>
#include <opencv2/opencv.hpp>

// output_ptr must have size >= 3 * dst_width * dst_height
static void letterbox_into_CHW_float_rgb(
    const cv::Mat& img,
    float* output_ptr,
    Eigen::Matrix3f& transform_matrix,
    int dst_width,
    int dst_height,
    bool normalize = true, // 是否做归一化
    bool use_imagenet = true, // normalize=true 时：是否使用 ImageNet mean/std
    const std::array<float, 3>& mean = { 0.485f, 0.456f, 0.406f }, // in RGB order
    const std::array<float, 3>& stdv = { 0.229f, 0.224f, 0.225f }, // in RGB order
    float pad_value = 114.0f // pad 的像素值（原始 0..255 量级）
) {
    // 校验
    CV_Assert(!img.empty());
    CV_Assert(img.type() == CV_8UC3);

    int img_h = img.rows;
    int img_w = img.cols;

    // 计算缩放与填充（与你原始实现一致）
    float scale = std::min(dst_height * 1.0f / img_h, dst_width * 1.0f / img_w);
    int resize_h = std::max(1, static_cast<int>(round(img_h * scale)));
    int resize_w = std::max(1, static_cast<int>(round(img_w * scale)));

    int pad_h = dst_height - resize_h;
    int pad_w = dst_width - resize_w;
    int top = static_cast<int>(round(pad_h / 2.0f - 0.1f));
    int left = static_cast<int>(round(pad_w / 2.0f - 0.1f));

    // 逆仿射变换矩阵（从网络坐标 -> 原图坐标）
    transform_matrix << 1.0f / scale, 0, -left / scale, 0, 1.0f / scale, -top / scale, 0, 0, 1;

    // 先 resize
    cv::Mat resized;
    cv::resize(img, resized, cv::Size(resize_w, resize_h), 0, 0, cv::INTER_LINEAR);

    const int H = dst_height;
    const int W = dst_width;
    const int HW = H * W;
    // precompute pad_value per channel according to normalize mode
    std::array<float, 3> pad_chan {};
    if (!normalize) {
        // 直接使用 0..255 的 pad value
        pad_chan = { pad_value, pad_value, pad_value }; // RGB order
    } else {
        if (use_imagenet) {
            // pad: (pad/255 - mean)/std
            for (int c = 0; c < 3; ++c) {
                float pv = pad_value / 255.0f;
                pad_chan[c] = (pv - mean[c]) / stdv[c];
            }
        } else {
            // 仅缩放到 [0,1]
            float pv = pad_value / 255.0f;
            pad_chan = { pv, pv, pv };
        }
    }

    // 1) fill whole buffer with pad values (CHW: R plane, G, B)
    // output_ptr layout: [R_plane (H*W), G_plane (H*W), B_plane (H*W)]
    for (int c = 0; c < 3; ++c) {
        float* plane_ptr = output_ptr + c * HW;
        for (int i = 0; i < HW; ++i)
            plane_ptr[i] = pad_chan[c];
    }

    // 2) copy resized image into region [top:top+resize_h, left:left+resize_w]
    //    注意：img/resized 是 BGR，且我们要输出 RGB 且放在 CHW
    for (int ry = 0; ry < resize_h; ++ry) {
        int dst_y = top + ry;
        if (dst_y < 0 || dst_y >= H)
            continue;
        const cv::Vec3b* src_row = resized.ptr<cv::Vec3b>(ry);
        // pointers to start of each plane row
        float* r_plane_row = output_ptr + 0 * HW + dst_y * W;
        float* g_plane_row = output_ptr + 1 * HW + dst_y * W;
        float* b_plane_row = output_ptr + 2 * HW + dst_y * W;

        for (int rx = 0; rx < resize_w; ++rx) {
            int dst_x = left + rx;
            if (dst_x < 0 || dst_x >= W)
                continue;

            cv::Vec3b pix = src_row[rx]; // B G R

            // get per-channel raw value
            float R = static_cast<float>(pix[2]);
            float G = static_cast<float>(pix[1]);
            float B = static_cast<float>(pix[0]);

            if (!normalize) {
                r_plane_row[dst_x] = R;
                g_plane_row[dst_x] = G;
                b_plane_row[dst_x] = B;
            } else {
                // scale to [0,1]
                float r = R / 255.0f;
                float g = G / 255.0f;
                float b = B / 255.0f;
                if (use_imagenet) {
                    r_plane_row[dst_x] = (r - mean[0]) / stdv[0]; // mean/std in RGB order
                    g_plane_row[dst_x] = (g - mean[1]) / stdv[1];
                    b_plane_row[dst_x] = (b - mean[2]) / stdv[2];
                } else {
                    r_plane_row[dst_x] = r;
                    g_plane_row[dst_x] = g;
                    b_plane_row[dst_x] = b;
                }
            }
        }
    }
}

ArmorDetectOnnxRuntime::ArmorDetectOnnxRuntime(
    std::string model_type,
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
    auto model = armor_infer::modeFromString(model_type);
    armor_infer_ =
        std::make_unique<armor_infer::ArmorInfer>(model, conf_threshold, nms_threshold, top_k);
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
    armor_infer_->generate_grids_and_stride(
        armor_infer_->getInputW(),
        armor_infer_->getInputH(),
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

    std::vector<float> input_tensor_values(
        armor_infer_->getInputW() * armor_infer_->getInputH() * 3
    );
    letterbox_into_CHW_float_rgb(
        frame.src_img,
        input_tensor_values.data(),
        transform_matrix,
        armor_infer_->getInputW(),
        armor_infer_->getInputH(),
        armor_infer_->getUseNorm(),
        false
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
    std::vector<armor::ArmorObject> objs_result;
    objs_result = armor_infer_->postProcess(output_buffer, transform_matrix, grid_strides_);
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
void ArmorDetectOnnxRuntime::pushInput(CommonFrame& frame) {
    frame.id = current_id_++;
    processCallback(frame);
}