#pragma once
#include "common/3rdparty/matplotlibcpp.h"
#include "common/debug/toolsgobal.hpp"
#include "common/gobal.hpp"
#include "type/type.hpp"
#include <thread>

void plotRobotCmdThread();
void writeCmdLogToJson();
void robotCmdLoggerThread();