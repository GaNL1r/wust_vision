#pragma once

#include "tasks/auto_buff/type.hpp"
namespace wust_vision {
namespace auto_buff {
    class RuneDetectorCV {
    public:
        using DetectorCallback =
            std::function<void(const auto_buff::RuneFan&, const CommonFrame&, cv::Mat&)>;
        using Ptr = std::unique_ptr<RuneDetectorCV>;
        RuneDetectorCV(const YAML::Node& node);
        static inline std::unique_ptr<RuneDetectorCV> make_detector(const YAML::Node& node) {
            return std::make_unique<RuneDetectorCV>(node);
        }
        ~RuneDetectorCV();
        void pushInput(CommonFrame& frame, bool debug = false);
        void setCallback(DetectorCallback callback);

    private:
        struct Impl;
        std::unique_ptr<Impl> _impl;
    };
} // namespace auto_buff
} // namespace wust_vision