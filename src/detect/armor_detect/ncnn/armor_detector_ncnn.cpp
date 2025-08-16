#include "detect/armor_detect/ncnn/armor_detector_ncnn.hpp"
#include "common/gobal.hpp"
#include "common/logger.hpp"
#include "detect/armor_detect/armor_infer.hpp"
#include <ncnn/simpleomp.h>
static ncnn::Mat letterbox_to_ncnn(
    const cv::Mat& img,
    Eigen::Matrix3f& transform_matrix,
    int out_w,
    int out_h,
    bool use_norm = true,
    bool use_imagenet = true
) {
    const int img_w = img.cols;
    const int img_h = img.rows;

    float scale = std::min(out_w * 1.0f / img_w, out_h * 1.0f / img_h);
    int resize_w = static_cast<int>(round(img_w * scale));
    int resize_h = static_cast<int>(round(img_h * scale));

    int pad_w = out_w - resize_w;
    int pad_h = out_h - resize_h;
    int pad_left = static_cast<int>(round(pad_w / 2.0f - 0.1f));
    int pad_top = static_cast<int>(round(pad_h / 2.0f - 0.1f));

    transform_matrix << 1.0f / scale, 0, -pad_left / scale, 0, 1.0f / scale, -pad_top / scale, 0, 0,
        1;

    ncnn::Mat out = ncnn::Mat::from_pixels_resize(
        img.data,
        ncnn::Mat::PIXEL_BGR2RGB,
        img_w,
        img_h,
        resize_w,
        resize_h
    );

    int pad_right = out_w - resize_w - pad_left;
    int pad_bottom = out_h - resize_h - pad_top;

    ncnn::Mat padded;
    ncnn::copy_make_border(
        out,
        padded,
        pad_top,
        pad_bottom,
        pad_left,
        pad_right,
        ncnn::BORDER_CONSTANT,
        114.f
    );
    if (use_norm) {
        // 两种常用策略：
        // A) 仅 scale 到 [0,1] -> mean = {0,0,0}, norm = {1/255,1/255,1/255}
        // B) ImageNet (x/255 - mean)/std:
        //    mean_vals = mean * 255, norm_vals = 1/(std * 255)
        std::array<float, 3> mean_vals;
        std::array<float, 3> norm_vals;

        if (use_imagenet) {
            // 注意：这里顺序为 RGB（因为 from_pixels_resize 用的是 PIXEL_BGR2RGB）
            const std::array<float, 3> mean = { 0.485f, 0.456f, 0.406f }; // R,G,B
            const std::array<float, 3> stdv = { 0.229f, 0.224f, 0.225f }; // R,G,B

            for (int c = 0; c < 3; ++c) {
                mean_vals[c] = mean[c] * 255.0f; // mean * 255
                norm_vals[c] = 1.0f / (stdv[c] * 255.0f); // 1 / (std * 255)
            }
        } else {
            // 只做 /255 -> [0,1]
            mean_vals = { 0.f, 0.f, 0.f };
            norm_vals = { 1.0f / 255.0f, 1.0f / 255.0f, 1.0f / 255.0f };
        }

        // 执行归一化（会把数据转为 float 并按通道处理）
        padded.substract_mean_normalize(mean_vals.data(), norm_vals.data());
    }

    return padded;
}

ArmorDetectNCNN::ArmorDetectNCNN(
    std::string model_type,
    std::string input_name_,
    std::string output_name_,
    const std::string& model_path_param,
    const std::string& model_path_bin,
    const ArmorDetectCommonParams& armor_detect_common_params,
    float conf_threshold,
    int top_k,
    float nms_threshold,
    bool use_gpu,
    int cpu_threads,
    bool use_lightmode,
    bool use_armor_detect_common,
    int device_id
):
    input_name_(input_name_),
    output_name_(output_name_),
    model_path_bin_(model_path_bin),
    model_path_param_(model_path_param),
    conf_threshold_(conf_threshold),
    top_k_(top_k),
    nms_threshold_(nms_threshold),
    use_gpu_(use_gpu),
    cpu_threads_(cpu_threads),
    use_lightmode_(use_lightmode),
    use_armor_detect_common_(use_armor_detect_common) {
    if (use_armor_detect_common_) {
        armor_detect_common_ = std::make_unique<ArmorDetectCommon>(armor_detect_common_params);
    }
    auto model = armor_infer::modeFromString(model_type);
    armor_infer_ =
        std::make_unique<armor_infer::ArmorInfer>(model, conf_threshold, nms_threshold, top_k);
    init(device_id);
}
ArmorDetectNCNN::~ArmorDetectNCNN() {
    net_.clear();
    armor_detect_common_.reset();
}

void ArmorDetectNCNN::init(int device_id) {
    if (use_gpu_) {
        ncnn::create_gpu_instance();
        opt_.use_vulkan_compute = true;
        ncnn::VulkanDevice* vkdev = ncnn::get_gpu_device(device_id);
        if (vkdev) {
            net_.set_vulkan_device(vkdev);
        }
        WUST_INFO("armor_ncnn") << "ncnn: use gpu";
    } else {
        opt_.use_vulkan_compute = false;
        WUST_INFO("armor_ncnn") << "ncnn: use cpu";
    }
    if (use_lightmode_) {
        opt_.lightmode = true;
    }

    opt_.num_threads = cpu_threads_;
    net_.opt = opt_;

    if (net_.load_param(model_path_param_.c_str()) != 0) {
        WUST_ERROR("armor_ncnn") << "Failed to load param";
        return;
    }
    if (net_.load_model(model_path_bin_.c_str()) != 0) {
        WUST_ERROR("armor_ncnn") << "Failed to load model";
        return;
    }

    int ret = net_.load_param((model_path_param_).c_str());
    if (ret != 0) {
        WUST_ERROR("armor_ncnn") << "Failed to load param file: " << model_path_param_;
        return;
    }

    ret = net_.load_model((model_path_bin_).c_str());
    if (ret != 0) {
        WUST_ERROR("armor_ncnn") << "Failed to load bin file: " << model_path_bin_;
        return;
    }

    // input_name_ = "images";
    // output_name_ = "output";

    strides_ = { 8, 16, 32 };
    armor_infer_->generate_grids_and_stride(
        armor_infer_->getInputW(),
        armor_infer_->getInputH(),
        strides_,
        grid_strides_
    );
}

void ArmorDetectNCNN::setCallback(DetectorCallback callback) {
    infer_callback_ = callback;
}
bool ArmorDetectNCNN::processCallback(const CommonFrame& frame) {
    // Eigen::Matrix3f transform_matrix;
    // cv::Mat resized_img = letterbox(frame.src_img, transform_matrix);
    // ncnn::Mat in =
    //     ncnn::Mat::from_pixels(resized_img.data, ncnn::Mat::PIXEL_BGR2RGB, INPUT_W, INPUT_H);
    Eigen::Matrix3f transform_matrix;
    ncnn::Mat in = letterbox_to_ncnn(
        frame.src_img,
        transform_matrix,
        armor_infer_->getInputW(),
        armor_infer_->getInputH(),
        armor_infer_->getUseNorm()
    );

    ncnn::Extractor ex = net_.create_extractor();
    ex.input(input_name_.c_str(), in);

    ncnn::Mat out;
    ex.extract(output_name_.c_str(), out);

    cv::Mat output_buffer(out.h, out.w, CV_32F, out.data);

    // Parse YOLO output
    auto objs_result = armor_infer_->postProcess(output_buffer, transform_matrix, grid_strides_);
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
            auto detect_color = gobal::stringanyting.get_value<int>("detect_color");
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

    return false;
}
void ArmorDetectNCNN::pushInput(CommonFrame& frame) {
    frame.id = current_id_++;
    processCallback(frame);
}
