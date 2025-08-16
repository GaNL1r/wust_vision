// Copyright 2025 Zikang Xie
// Copyright 2025 Xiaojian Wu
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

#include "detect/armor_detect/openvino/armor_detector_openvino.hpp"
#include "common/gobal.hpp"
#include "common/logger.hpp"
#include "common/timer.hpp"
#include "detect/armor_detect/armor_infer.hpp"
#include <functional>
#include <opencv2/highgui.hpp>

ArmorDetectOpenVino::ArmorDetectOpenVino(
    std::string model_type,
    const std::filesystem::path& model_path,
    const std::string& device_name,
    const ArmorDetectCommonParams& armor_detect_common_params,
    float conf_threshold,
    int top_k,
    float nms_threshold,
    bool use_throughputmode,
    bool use_armor_detect_common
):

    model_path_(model_path),
    device_name_(device_name),
    conf_threshold_(conf_threshold),
    top_k_(top_k),
    nms_threshold_(nms_threshold),
    use_throughputmode_(use_throughputmode),
    use_armor_detect_common_(use_armor_detect_common) {
    if (use_armor_detect_common_) {
        armor_detect_common_ = std::make_unique<ArmorDetectCommon>(armor_detect_common_params);
    }
    auto model = armor_infer::modeFromString(model_type);
    armor_infer_ =
        std::make_unique<armor_infer::ArmorInfer>(model, conf_threshold, nms_threshold, top_k);
    init();
}

void ArmorDetectOpenVino::init() {
    // 1) Core／Model 读取
    if (!ov_core_) {
        ov_core_ = std::make_unique<ov::Core>();
    }
    // load IR
    model_ = ov_core_->read_model(model_path_);

    // 2) PrePostProcessor 配置
    ov::preprocess::PrePostProcessor ppp(model_);

    // 告诉引擎：输入 tensor 是 u8、NHWC、BGR
    ppp.input()
        .tensor()
        .set_element_type(ov::element::u8)
        .set_layout("NHWC")
        .set_color_format(ov::preprocess::ColorFormat::BGR);

    // 预处理管线：u8→f32、BGR→RGB
    float scale = armor_infer_->getUseNorm() ? 255.0f : 1.0f;
    ppp.input()
        .preprocess()
        .convert_element_type(ov::element::f32)
        .convert_color(ov::preprocess::ColorFormat::RGB)
        .scale({ scale, scale, scale });

    // 告诉引擎：模型内部期望的布局是 NCHW
    ppp.input().model().set_layout("NCHW");

    // 输出也要 f32
    ppp.output().tensor().set_element_type(ov::element::f32);

    // 把预处理节点「贴」到模型里
    model_ = ppp.build();

    // 3) 编译模型（可以带 performance_mode hint）
    ov::hint::PerformanceMode mode = use_throughputmode_ ? ov::hint::PerformanceMode::THROUGHPUT
                                                         : ov::hint::PerformanceMode::LATENCY;
    compiled_model_ = std::make_unique<ov::CompiledModel>(
        ov_core_->compile_model(model_, device_name_, ov::hint::performance_mode(mode))
    );

    // 4) 生成 grid_strides、ThreadPool 等
    strides_ = { 8, 16, 32 };
    armor_infer_->generate_grids_and_stride(
        armor_infer_->getInputW(),
        armor_infer_->getInputH(),
        strides_,
        grid_strides_
    );
}

ArmorDetectOpenVino::~ArmorDetectOpenVino() {
    compiled_model_.reset();
    ov_core_.reset();
    armor_detect_common_.reset();
}

void ArmorDetectOpenVino::setCallback(DetectorCallback callback) {
    infer_callback_ = callback;
}
bool ArmorDetectOpenVino::processCallback(const CommonFrame& frame) {
    auto start = std::chrono::steady_clock::now();
    Eigen::Matrix3f transform_matrix;

    // BGR->RGB, u8(0-255)->f32(0.0-1.0), HWC->NCHW
    // note: TUP's model no need to normalize
    // cv::Mat blob = cv::dnn::blobFromImage(
    //     resized_img,
    //     1.,
    //     cv::Size(INPUT_W, INPUT_H),
    //     cv::Scalar(0, 0, 0),
    //     true
    // );

    // Feed blob into input
    // auto input_port = compiled_model_->input();
    // ov::Tensor input_tensor(
    //     input_port.get_element_type(),
    //     ov::Shape(std::vector<size_t> { 1, 3, INPUT_W, INPUT_H }),
    //     blob.ptr(0)
    // );

    // // Start inference
    // auto infer_request = compiled_model_->create_infer_request();
    // infer_request.set_input_tensor(input_tensor);
    // infer_request.infer();

    // ov::Tensor input_tensor = ov::Tensor(
    //     compiled_model_->input().get_element_type(),
    //     compiled_model_->input().get_shape(),
    //     resized_img.data
    // );
    cv::Mat resized_img = armor_infer_->letterbox(
        frame.src_img,
        transform_matrix,
        armor_infer_->getInputW(),
        armor_infer_->getInputH()
    );
    auto input_tensor = ov::Tensor(
        compiled_model_->input().get_element_type(),
        compiled_model_->input().get_shape(),
        resized_img.data
    );
    auto* data_ptr = input_tensor.data<uint8_t>();

    // 3) InferRequest
    auto t1 = time_utils::now();
    auto infer_request = compiled_model_->create_infer_request();
    auto t2 = time_utils::now();
    infer_request.set_input_tensor(input_tensor);
    auto t3 = time_utils::now();
    infer_request.infer();
    auto t4 = time_utils::now();
    auto output = infer_request.get_output_tensor();
    auto t5 = time_utils::now();
    // std::cout << "time: " << time_utils::durationMs(t1, t2) << " " << time_utils::durationMs(t2, t3)
    //           << " " << time_utils::durationMs(t3, t4) << " " << time_utils::durationMs(t4, t5)
    //           << std::endl;

    // Process output data
    auto output_shape = output.get_shape();
    // 3549 x 21 Matrix
    cv::Mat output_buffer(output_shape[1], output_shape[2], CV_32F, output.data());

    // Parsed variable
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
void ArmorDetectOpenVino::pushInput(CommonFrame& frame) {
    frame.id = current_id_++;
    processCallback(frame);
}