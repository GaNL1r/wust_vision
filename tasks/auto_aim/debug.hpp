#pragma once
#include "tasks/auto_aim/armor_tracker/target.hpp"
#include "tasks/auto_aim/auto_aim_fsm.hpp"
#include "tasks/auto_aim/type.hpp"
#include "tasks/utils/debug_utils.hpp"
namespace wust_vision::auto_aim {

struct AutoAimDebug {
    wust_vl::video::ImageFrame img_frame;
    auto_aim::Armors armors;
    auto_aim::Target target;
    GimbalCmd gimbal_cmd;
    auto_aim::AutoAimFsm fsm;
    AimTarget aim_target;
    double latency_ms;
    Eigen::Matrix4d T_camera_to_odom;
    std::vector<auto_aim::ArmorObject> armor_objs;
    int detect_color = 0;
    cv::Rect expanded;
    std::pair<double, double> gimbal_py;
};
void drawDebugArmorContent(
    cv::Mat& debug_img,
    const AutoAimDebug& dbg,
    std::pair<cv::Mat, cv::Mat> camera_info
);
void writeTargetLogToJson(const auto_aim::Target& armor_target);

inline void drawDebugOverlayShm(
    const AutoAimDebug& dbg,
    std::pair<cv::Mat, cv::Mat> camera_info,
    bool auto_fps
) {
    static ShmWriter shm { "/debug_frame" };
    drawDebugOverlayImpl(dbg, camera_info, auto_fps, drawDebugArmorContent, shm);
}
void debuglog(const AutoAimDebug& dbg_armor);
} // namespace wust_vision::auto_aim