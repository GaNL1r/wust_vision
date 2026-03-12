#pragma once
#include "tasks/type_common.hpp"
#include "wust_vl/common/concurrency/monitored_thread.hpp"
#include <memory>
namespace wust_vision {
class IModule {
public:
    using Ptr = std::shared_ptr<IModule>;
    virtual void start() = 0;
    virtual void pushInput(CommonFrame&) = 0;
    virtual GimbalCmd solve(double bullet_speed) = 0;
    virtual wust_vl::common::concurrency::MonitoredThread::Ptr getThread() = 0;
    virtual void doDebug() = 0;
    virtual ~IModule() = default;
};
} // namespace wust_vision