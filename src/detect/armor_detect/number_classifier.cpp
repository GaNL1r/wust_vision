// Copyright Chen Jun 2023. Licensed under the MIT License.
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

#include "detect/armor_detect/number_classifier.hpp"

NumberClassifier::NumberClassifier(
    const std::string& classify_model_path,
    const std::string& classify_label_path
):
    classify_model_path_(classify_model_path),
    classify_label_path_(classify_label_path) {
    initNumberClassifier();
}
void NumberClassifier::initNumberClassifier() {
    const std::string model_path = classify_model_path_;
    std::unique_ptr<cv::dnn::Net> number_net_ =
        std::make_unique<cv::dnn::Net>(cv::dnn::readNetFromONNX(model_path));

    if (number_net_->empty()) {
        WUST_ERROR("number_classifier")
            << "Failed to load number classifier model from " << model_path;
        std::exit(EXIT_FAILURE);
    } else {
        WUST_INFO("number_classifier")
            << "Successfully loaded number classifier model from " << model_path;
    }

    const std::string label_path = classify_label_path_;
    std::ifstream label_file(label_path);
    std::string line;

    class_names_.clear();

    while (std::getline(label_file, line)) {
        class_names_.push_back(line);
    }

    if (class_names_.empty()) {
        WUST_ERROR("number_classifier") << "Failed to load labels from " << label_path;
        std::exit(EXIT_FAILURE);
    } else {
        WUST_INFO("number_classifier")
            << "Successfully loaded " << class_names_.size() << " labels from " << label_path;
    }
    number_net_.reset();
}
bool NumberClassifier::classifyNumber(armor::ArmorObject& armor) {
    static thread_local std::unique_ptr<cv::dnn::Net> thread_net;
    if (armor.number_img.empty()) {
        return false;
    }

    if (!thread_net) {
        thread_net = std::make_unique<cv::dnn::Net>(cv::dnn::readNetFromONNX(classify_model_path_));
        WUST_DEBUG("number_classifier") << "Loaded number classifier model for this thread";
        if (thread_net->empty()) {
            WUST_ERROR("number_classifier")
                << "Failed to load thread-local number classifier model.";
            return false;
        }
    }

    cv::Mat image = armor.number_img.clone();
    image = image / 255.0;

    cv::Mat blob;
    cv::dnn::blobFromImage(image, blob);

    thread_net->setInput(blob);
    cv::Mat outputs = thread_net->forward();

    float max_prob = *std::max_element(outputs.begin<float>(), outputs.end<float>());
    cv::Mat softmax_prob;
    cv::exp(outputs - max_prob, softmax_prob);
    float sum = static_cast<float>(cv::sum(softmax_prob)[0]);
    softmax_prob /= sum;

    double confidence;
    cv::Point class_id_point;
    cv::minMaxLoc(softmax_prob.reshape(1, 1), nullptr, &confidence, nullptr, &class_id_point);
    int label_id = class_id_point.x;

    armor.confidence = confidence;

    static const std::map<int, armor::ArmorNumber> label_to_armor_number = {
        { 0, armor::ArmorNumber::NO1 },    { 1, armor::ArmorNumber::NO2 },
        { 2, armor::ArmorNumber::NO3 },    { 3, armor::ArmorNumber::NO4 },
        { 4, armor::ArmorNumber::NO5 },    { 5, armor::ArmorNumber::OUTPOST },
        { 6, armor::ArmorNumber::SENTRY }, { 7, armor::ArmorNumber::BASE }
    };

    if (label_id < 8 && label_to_armor_number.find(label_id) != label_to_armor_number.end()) {
        armor.number = label_to_armor_number.at(label_id);

        return true;
    } else {
        //armor.confidence = 0;
        return false;
    }
}
