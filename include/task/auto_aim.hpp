#pragma once
#include "common/concurrency/queues.hpp"
#include "common/utils/timer.hpp"
#include "control/armor_solver.hpp"
#include "detect/armor_detect/armor_pose_estimator.hpp"
#include "detect/detector_factory.hpp"
#include "tracker/tracker_manager.hpp"
#include "type/type.hpp"
#include "yaml-cpp/yaml.h"
class AutoAim {
    bool init(YAML::Node config);
    void pushInput(const CommonFrame& frame);

    struct Impl;
    std::unique_ptr<Impl> _impl;
};