#pragma once
#include "rune_target.hpp"
namespace auto_buff {
class RuneTracker {
public:
    using Ptr = std::unique_ptr<RuneTracker>;
    RuneTracker(const YAML::Node& config);
    static Ptr create(const YAML::Node& config) {
        return std::make_unique<RuneTracker>(config);
    }
    ~RuneTracker();
    auto_buff::RuneTarget track(const auto_buff::RuneFan& fan);

private:
    struct Impl;
    std::unique_ptr<Impl> _impl;
};
} // namespace auto_buff