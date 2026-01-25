#include "guidance_detector_openvino.hpp"
#include "tasks/utils.hpp"
namespace auto_guidance {
struct GuidanceDetectorOpenVino::Impl {
public:
    Impl(const YAML::Node& config_gobal) {
        auto config = config_gobal["openvino"];
        std::string model_path = utils::expandEnv(config["model_path"].as<std::string>());
        std::string device_name = config["device_type"].as<std::string>();
        int top_k = config["top_k"].as<int>();
        float nms_threshold = config["nms_threshold"].as<float>();
        float conf_threshold = config["conf_threshold"].as<float>();
        green_light_infer_ = GreenLightInfer::makeGreenLightInfer(GreenLightInfer::Params {
            .conf_threshold = conf_threshold,
            .nms_threshold = nms_threshold,
            .top_k = top_k,
            .input_h = 384,
            .input_w = 640,
            .use_norm = true });
        openvino_net_ = std::make_unique<ml_net::OpenvinoNet>();
        auto ppp_init_fun = [this](ov::preprocess::PrePostProcessor& ppp) {
            ppp.input()
                .tensor()
                .set_element_type(ov::element::u8)
                .set_layout("NHWC")
                .set_color_format(ov::preprocess::ColorFormat::BGR);

            ppp.input()
                .preprocess()
                .convert_element_type(ov::element::f32)
                .convert_color(ov::preprocess::ColorFormat::RGB)
                .scale({ 255.f });

            ppp.input().model().set_layout("NCHW");

            ppp.output(0).tensor().set_element_type(ov::element::f32);

            ppp.output(1).tensor().set_element_type(ov::element::f32);

            ppp.output(2).tensor().set_element_type(ov::element::f32);
        };
        ml_net::OpenvinoNet::Params params;
        params.model_path = model_path;
        params.device_name = device_name;
        params.mode = config["use_throughputmode"].as<bool>()
            ? ov::hint::PerformanceMode::THROUGHPUT
            : ov::hint::PerformanceMode::LATENCY;
        openvino_net_->init(params, ppp_init_fun);
    }
    ~Impl() {
        openvino_net_.reset();
        green_light_infer_.reset();
    }

    void setCallback(DetectorCallback callback) {
        infer_callback_ = callback;
    }

    void processCallback(const CommonFrame& frame) const {
        Eigen::Matrix3f transform_matrix;
        const cv::Mat resized_img = utils::letterbox(
            frame.src_img,
            transform_matrix,
            green_light_infer_->getInputW(),
            green_light_infer_->getInputH()
        );

        const auto input_info = openvino_net_->getInputInfo();
        const auto input_tensor = ov::Tensor(input_info.first, input_info.second, resized_img.data);

        auto infer_request = openvino_net_->createInferRequest();
        infer_request.set_input_tensor(input_tensor);
        infer_request.infer();
        const auto output = infer_request.get_output_tensor(0);

        // Process output data
        const auto output_shape = output.get_shape();
        const cv::Mat output_buffer(output_shape[1], output_shape[2], CV_32F, output.data());
        const auto objs_result = green_light_infer_->postProcess(output_buffer, transform_matrix);
        if (infer_callback_) {
            infer_callback_(objs_result, frame);
        }
    }
    void pushInput(CommonFrame& frame) {
        frame.id = current_id_++;
        processCallback(frame);
    }
    std::unique_ptr<ml_net::OpenvinoNet> openvino_net_;
    std::unique_ptr<GreenLightInfer> green_light_infer_;
    DetectorCallback infer_callback_;
    int current_id_ = 0;
};
GuidanceDetectorOpenVino::GuidanceDetectorOpenVino(const YAML::Node& config_gobal) {
    _impl = std::make_unique<Impl>(config_gobal);
}
GuidanceDetectorOpenVino::~GuidanceDetectorOpenVino() {
    _impl.reset();
}
void GuidanceDetectorOpenVino::pushInput(CommonFrame& frame) {
    _impl->pushInput(frame);
}
void GuidanceDetectorOpenVino::setCallback(DetectorCallback callback) {
    _impl->setCallback(callback);
}
} // namespace auto_guidance