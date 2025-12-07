#pragma once
#include "tasks/auto_guidance/type.hpp"
#include "tasks/type_common.hpp"
#include <yaml-cpp/node/node.h>
namespace auto_guidance {

class detector_base {
public:
    virtual ~detector_base() = default;
    virtual void pushInput(CommonFrame& frame) = 0;

    using DetectorCallback =
        std::function<void(const std::vector<GreenLight>&, const CommonFrame&)>;

    virtual void setCallback(DetectorCallback cb) = 0;
};
} // namespace auto_guidance