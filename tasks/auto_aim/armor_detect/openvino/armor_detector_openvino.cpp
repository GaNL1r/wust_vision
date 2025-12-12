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

#include "tasks/auto_aim/armor_detect/openvino/armor_detector_openvino.hpp"
#include "tasks/auto_aim/armor_detect/armor_infer.hpp"

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
    openvino_net_ = std::make_unique<ml_net::OpenvinoNet>();
    auto ppp_init_fun = [this](ov::preprocess::PrePostProcessor& ppp) {
        ppp.input()
            .tensor()
            .set_element_type(ov::element::u8)
            .set_layout("NHWC")
            .set_color_format(ov::preprocess::ColorFormat::BGR);

        float scale = armor_infer_->getUseNorm() ? 255.0f : 1.0f;
        ppp.input()
            .preprocess()
            .convert_element_type(ov::element::f32)
            .convert_color(ov::preprocess::ColorFormat::RGB)
            .scale({ scale });

        ppp.input().model().set_layout("NCHW");

        ppp.output().tensor().set_element_type(ov::element::f32);
    };
    ml_net::OpenvinoNet::Params params;
    params.model_path = model_path_;
    params.device_name = device_name_;
    params.mode = use_throughputmode_ ? ov::hint::PerformanceMode::THROUGHPUT
                                      : ov::hint::PerformanceMode::LATENCY;
    openvino_net_->init(params, ppp_init_fun);

    strides_ = { 8, 16, 32 };
    armor_infer_->generate_grids_and_stride(
        armor_infer_->getInputW(),
        armor_infer_->getInputH(),
        strides_,
        grid_strides_
    );
}

ArmorDetectOpenVino::~ArmorDetectOpenVino() {
    openvino_net_.reset();
    armor_detect_common_.reset();
}

void ArmorDetectOpenVino::setCallback(DetectorCallback callback) {
    infer_callback_ = callback;
}
bool ArmorDetectOpenVino::processCallback(const CommonFrame& frame) {
    auto start = std::chrono::steady_clock::now();
    Eigen::Matrix3f transform_matrix;
    auto roi = frame.src_img(frame.expanded);
    cv::Mat resized_img = armor_infer_->letterbox(
        roi,
        transform_matrix,
        armor_infer_->getInputW(),
        armor_infer_->getInputH()
    );
    auto input_info = openvino_net_->getInputInfo();
    auto input_tensor = ov::Tensor(input_info.first, input_info.second, resized_img.data);
    auto output = openvino_net_->infer_thread_local(input_tensor);

    // Process output data
    auto output_shape = output.get_shape();

    cv::Mat output_buffer(output_shape[1], output_shape[2], CV_32F, output.data());

    // Parsed variable
    auto objs_result = armor_infer_->postProcess(output_buffer, transform_matrix, grid_strides_);

    std::vector<armor::ArmorObject> armors;
    if (use_armor_detect_common_ && roi.data != nullptr) {
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
void ArmorDetectOpenVino::pushInput(CommonFrame& frame) {
    frame.id = current_id_++;
    processCallback(frame);
}