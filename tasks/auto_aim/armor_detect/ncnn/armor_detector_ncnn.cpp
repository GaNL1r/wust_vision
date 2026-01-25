#include "tasks/auto_aim/armor_detect/ncnn/armor_detector_ncnn.hpp"
#include "tasks/auto_aim/armor_detect/armor_detector_common.hpp"
#include "tasks/auto_aim/armor_detect/armor_infer.hpp"
#include "wust_vl/ml_net/ncnn/ncnn_net.hpp"
#include <ncnn/simpleomp.h>
namespace auto_aim {

struct ArmorDetectorNCNN::Impl {
public:
    Impl(const YAML::Node& config, bool use_armor_detect_common) {
        if (use_armor_detect_common) {
            armor_detect_common_ = ArmorDetectorCommon::create(config);
        }
        std::string model_type = config["ncnn"]["model_type"].as<std::string>();
        auto model = armor_infer::modeFromString(model_type);
        float conf_threshold = config["ncnn"]["conf_threshold"].as<float>();
        int top_k = config["ncnn"]["top_k"].as<int>();
        float nms_threshold = config["ncnn"]["nms_threshold"].as<float>();
        armor_infer_ =
            std::make_unique<armor_infer::ArmorInfer>(model, conf_threshold, nms_threshold, top_k);
        std::string model_path_param =
            utils::expandEnv(config["ncnn"]["model_path_param"].as<std::string>());
        std::string model_path_bin =
            utils::expandEnv(config["ncnn"]["model_path_bin"].as<std::string>());
        bool use_gpu = config["ncnn"]["use_gpu"].as<bool>();
        int cpu_threads = config["ncnn"]["cpu_threads"].as<int>();
        bool use_lightmode = config["ncnn"]["use_lightmode"].as<bool>();
        auto input_name = config["ncnn"]["input_name"].as<std::string>();
        auto output_name = config["ncnn"]["output_name"].as<std::string>();
        int device_id = config["ncnn"]["device_id"].as<int>();
        ml_net::NCNNNet::Params params;
        params.model_path_param = model_path_param;
        params.model_path_bin = model_path_bin;
        params.input_name = input_name;
        params.output_name = output_name;
        params.use_vulkan = use_gpu;
        params.device_id = device_id;
        params.use_light_mode = use_lightmode;
        params.cpu_threads = cpu_threads;
        ncnn_net_ = std::make_unique<ml_net::NCNNNet>();
        ncnn_net_->init(params);

        strides_ = { 8, 16, 32 };
        armor_infer_->generate_grids_and_stride(
            armor_infer_->getInputW(),
            armor_infer_->getInputH(),
            strides_,
            grid_strides_
        );
    }
    static Ptr create(const YAML::Node& config, bool use_armor_detect_common) {
        return std::make_unique<ArmorDetectorNCNN>(config, use_armor_detect_common);
    }
    ~Impl() {
        armor_detect_common_.reset();
        ncnn_net_.reset();
    }
    cv::Mat ncnnMatToCvMat(const ncnn::Mat& m) {
        cv::Mat img(m.h, m.w, CV_8UC3);
        m.to_pixels(img.data, ncnn::Mat::PIXEL_RGB2BGR);

        return img;
    }
    void setCallback(DetectorCallback callback) {
        infer_callback_ = callback;
    }
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

        transform_matrix << 1.0f / scale, 0, -pad_left / scale, 0, 1.0f / scale, -pad_top / scale,
            0, 0, 1;

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
    void
    processCallback(const CommonFrame& frame, const std::optional<ArmorNumber>& target_number) {
        // Eigen::Matrix3f transform_matrix;
        // cv::Mat resized_img = letterbox(frame.src_img, transform_matrix);
        // ncnn::Mat in =
        //     ncnn::Mat::from_pixels(resized_img.data, ncnn::Mat::PIXEL_BGR2RGB, INPUT_W, INPUT_H);
        Eigen::Matrix3f transform_matrix;
        auto roi = frame.src_img(frame.expanded);
        ncnn::Mat in = letterbox_to_ncnn(
            roi.clone(),
            transform_matrix,
            armor_infer_->getInputW(),
            armor_infer_->getInputH(),
            armor_infer_->getUseNorm()
        );
        cv::Mat resized_img = ncnnMatToCvMat(in);

        auto out = ncnn_net_->infer(in);

        cv::Mat output_buffer(out.h, out.w, CV_32F, out.data);

        // Parse YOLO output
        auto objs_result =
            armor_infer_->postProcess(output_buffer, transform_matrix, grid_strides_);
        std::vector<ArmorObject> armors;
        if (armor_detect_common_) {
            armors = armor_detect_common_->detectNet(
                resized_img,
                objs_result,
                transform_matrix,
                frame.detect_color,
                target_number
            );
            // Call callback function
            if (this->infer_callback_) {
                this->infer_callback_(armors, frame);
                return;
            }
        } else {
            for (auto obj: objs_result) {
                auto detect_color = frame.detect_color;
                if (detect_color == 0 && obj.color == ArmorColor::BLUE) {
                    continue;
                } else if (detect_color == 1 && obj.color == ArmorColor::RED) {
                    continue;
                }
                obj.transform(transform_matrix);
                armors.push_back(obj);
            }
            if (this->infer_callback_) {
                this->infer_callback_(armors, frame);
                return;
            }
        }

        return;
    }
    void pushInput(CommonFrame& frame, const std::optional<ArmorNumber>& target_number) {
        frame.id = current_id_++;
        processCallback(frame, target_number);
    }

private:
    DetectorCallback infer_callback_;
    std::vector<int> strides_;
    std::vector<GridAndStride> grid_strides_;
    std::unique_ptr<ArmorDetectorCommon> armor_detect_common_;
    std::unique_ptr<armor_infer::ArmorInfer> armor_infer_;
    int current_id_ = 0;
    std::unique_ptr<ml_net::NCNNNet> ncnn_net_;
};
ArmorDetectorNCNN::ArmorDetectorNCNN(const YAML::Node& config, bool use_armor_detect_common) {
    _impl = std::make_unique<Impl>(config, use_armor_detect_common);
}
ArmorDetectorNCNN::~ArmorDetectorNCNN() {
    _impl.reset();
}
void ArmorDetectorNCNN::setCallback(DetectorCallback callback) {
    _impl->setCallback(callback);
}
void ArmorDetectorNCNN::pushInput(
    CommonFrame& frame,
    const std::optional<ArmorNumber>& target_number
) {
    _impl->pushInput(frame, target_number);
}
} // namespace auto_aim