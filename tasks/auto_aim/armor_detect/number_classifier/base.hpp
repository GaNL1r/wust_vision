#pragma once
#include "tasks/auto_aim/type.hpp"

class NumberClassifierBase {
public:
    virtual ~NumberClassifierBase() = default;
    virtual bool classifyNumber(armor::ArmorObject& armor) = 0;
    virtual void initNumberClassifier() = 0;
};