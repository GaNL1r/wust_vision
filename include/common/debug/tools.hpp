#pragma once
#include "common/debug/toolsgobal.hpp"
#include "common/tf.hpp"
#include "detect/mono_measure_tool.hpp"
#include "driver/packet_typedef.hpp"
#include "opencv2/opencv.hpp"
#include "tracker/tracker.hpp"
#include "type/type.hpp"
#include <opencv2/core/mat.hpp>
#include <optional>
void drawResult(const imgframe& src_img, const Armors& armors);
void dumpTargetToFile(const Target& target, const std::string& path = "/tmp/target_status.txt");
void dumpImuToFile(const ReceiveImuData& imu, const std::string& path = "/tmp/imu_status.txt");
void dumpAimToFile(const ReceiveAimINFO& aim, const std::string& path = "/tmp/aim_status.txt");
std::string formatAimInfo(const ReceiveAimINFO& aim);
std::string formatImuInfo(const ReceiveImuData& imu);
void writeTargetLogToJson(const Target& target);
void writeAimLogToJson(const ReceiveAimINFO& aim);
std::string GetUniqueVideoFilename(const std::string& folder, const std::string& prefix = "output");
cv::Point2f normalize(const cv::Point2f& v);
void writeCmdLogToJson();
void robotCmdLoggerThread();
void drawDebugOverlayWrite(const DebugArmor& dbg);
void drawDebugOverlayShow(const DebugArmor& dbg);
cv::Mat drawDebugOverlayMat(const DebugArmor& dbg);
void drawDebugOverlayWrite(const DebugRune& dbg);
void drawDebugOverlayShow(const DebugRune& dbg);
cv::Mat drawDebugOverlayMat(const DebugRune& dbg);
void drawDebugArmorContent(cv::Mat& debug_img, const DebugArmor& dbg);
void drawDebugRuneContent(cv::Mat& debug_img, const DebugRune& dbg);