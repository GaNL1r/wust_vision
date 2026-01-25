#pragma once
#include "tasks/auto_guidance/guidance_tracker/guidance_target.hpp"
namespace wust_vision {
namespace auto_guidance {
    class GuidanceTracker {
    public:
        using Ptr = std::unique_ptr<GuidanceTracker>;
        GuidanceTracker(const YAML::Node& config);
        ~GuidanceTracker();
        static inline Ptr create(const YAML::Node& config) {
            return std::make_unique<GuidanceTracker>(config);
        }
        GuidanceTarget track(const GreenLights& lights);
        std::chrono::steady_clock::time_point getLastTime() const;

    private:
        struct Impl;
        std::unique_ptr<Impl> _impl;
    };
} // namespace auto_guidance
} // namespace wust_vision