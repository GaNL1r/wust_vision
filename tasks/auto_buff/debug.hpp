#pragma once
#include "tasks/auto_buff/auto_buff.hpp"
#include "tasks/auto_buff/rune_tracker/rune_target.hpp"
#include "tasks/utils/debug_utils.hpp"
#include <wust_vl/video/icamera.hpp>
namespace wust_vision::auto_buff {
struct AutoBuffDebug {
    wust_vl::video::ImageFrame img_frame;
    auto_buff::RuneTarget target;
    AimTarget aim_target;
    auto_buff::PowerRune power_rune;
    double predict_angle;
    GimbalCmd gimbal_cmd;
    Eigen::Matrix4d T_camera_to_odom;
    double latency_ms;
    double obs_angle;
    double pre_angle;
    double fitter_v;
    double obs_v;
    cv::Rect expanded;
    double pnp_distance;
    std::pair<double, double> gimbal_py;
};
void drawDebugRuneContent(
    cv::Mat& debug_img,
    const AutoBuffDebug& dbg,
    std::pair<cv::Mat, cv::Mat> camera_info
);
void writeTargetLogToJson(const auto_buff::RuneTarget& rune_target);
inline void drawDebugOverlayShm(
    const AutoBuffDebug& dbg,
    std::pair<cv::Mat, cv::Mat> camera_info,
    bool auto_fps
) {
    static ShmWriter shm { "/debug_frame" };
    drawDebugOverlayImpl(dbg, camera_info, auto_fps, drawDebugRuneContent, shm);
}
void debuglog(const AutoBuffDebug& dbg);
} // namespace wust_vision::auto_buff