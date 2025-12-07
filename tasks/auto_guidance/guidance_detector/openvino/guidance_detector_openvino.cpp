#include "guidance_detector_openvino.hpp"
#include "tasks/utils.hpp"
namespace auto_guidance {
GuidanceDetectorOpenVino::GuidanceDetectorOpenVino(const YAML::Node& config_gobal) {
    auto config = config_gobal["openvino"];
    model_path_ = utils::expandEnv(config["model_path"].as<std::string>());
    device_name_ = config["device_type"].as<std::string>();
    top_k_ = config["top_k"].as<int>();
    nms_threshold_ = config["nms_threshold"].as<float>();
    conf_threshold_ = config["conf_threshold"].as<float>();
    green_light_infer_ = GreenLightInfer::makeGreenLightInfer(GreenLightInfer::Params {
        .conf_threshold = conf_threshold_,
        .nms_threshold = nms_threshold_,
        .top_k = top_k_,
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
    params.model_path = model_path_;
    params.device_name = device_name_;
    params.mode = config["use_throughputmode"].as<bool>() ? ov::hint::PerformanceMode::THROUGHPUT
                                                          : ov::hint::PerformanceMode::LATENCY;
    openvino_net_->init(params, ppp_init_fun);
}
GuidanceDetectorOpenVino::~GuidanceDetectorOpenVino() {
    openvino_net_.reset();
    green_light_infer_.reset();
}
void GuidanceDetectorOpenVino::setCallback(DetectorCallback callback) {
    infer_callback_ = callback;
}

void GuidanceDetectorOpenVino::processCallback(const CommonFrame& frame) {
    Eigen::Matrix3f transform_matrix;
    cv::Mat resized_img = green_light_infer_->letterbox(
        frame.src_img,
        transform_matrix,
        green_light_infer_->getInputW(),
        green_light_infer_->getInputH()
    );

    auto input_info = openvino_net_->getInputInfo();
    auto input_tensor = ov::Tensor(input_info.first, input_info.second, resized_img.data);

    auto infer_request = openvino_net_->createInferRequest();
    infer_request.set_input_tensor(input_tensor);
    infer_request.infer();
    auto output = infer_request.get_output_tensor(0);

    // Process output data
    auto output_shape = output.get_shape();
    cv::Mat output_buffer(output_shape[1], output_shape[2], CV_32F, output.data());
    auto objs_result = green_light_infer_->postProcess(output_buffer, transform_matrix);
    if (infer_callback_) {
        infer_callback_(objs_result, frame);
    }
}
void GuidanceDetectorOpenVino::pushInput(CommonFrame& frame) {
    frame.id = current_id_++;
    processCallback(frame);
}
} // namespace auto_guidance