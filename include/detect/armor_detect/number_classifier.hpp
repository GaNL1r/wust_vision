#pragma once
#include "type/type.hpp"

class NumberClassifier {
public:
    NumberClassifier(
        const std::string& classify_model_path,
        const std::string& classify_label_path
    );
    void initNumberClassifier();
    bool classifyNumber(ArmorObject& armor);

private:
    cv::dnn::Net number_net_;
    std::vector<std::string> class_names_;
    std::string classify_model_path_;
    std::string classify_label_path_;
};