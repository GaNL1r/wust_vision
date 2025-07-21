#pragma once
#include "type/type.hpp"

double orientationToYaw(const tf::Quaternion& orientation);
void commandCallback(armor::Armors& armors);
void ex(double& a, double& min, double& max);
void commandCallbackYpd(const armor::Armors& armors);