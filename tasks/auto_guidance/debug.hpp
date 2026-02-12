#pragma once
#include "tasks/auto_guidance/guidance_tracker/guidance_target.hpp"
#include "wust_vl/video/icamera.hpp"
namespace wust_vision {
namespace auto_guidance {

    struct AutoGuidanceDebug {
        wust_vl::video::ImageFrame img_frame;
        double latency_ms;
        GuidanceTarget target;
        GreenLights lights;
        cv::Mat mask;
    };
    struct DebugLogs {
        std::vector<double> time_log;
        std::vector<double> cx_log;
    };
    void debuglog(const GuidanceTarget& target);
    void drawDebugOverlayShm(const AutoGuidanceDebug& dbg, bool auto_fps);
    void drawDebugOverlayWrite(const AutoGuidanceDebug& dbg, bool auto_fps);
    void drawDebugOverlayShow(const AutoGuidanceDebug& dbg, bool auto_fps);
    void drawAutoGuidanceDebugContent(cv::Mat& debug_img, const AutoGuidanceDebug& dbg);
} // namespace auto_guidance
} // namespace wust_vision