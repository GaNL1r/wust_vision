#pragma once
#include "type/type.hpp"

double orientationToYaw(const Eigen::Quaterniond& q);
void command_callback(Armors& armors);
void ex(double& a, double& min, double& max);
void command_callbackypd(const Armors& armors);