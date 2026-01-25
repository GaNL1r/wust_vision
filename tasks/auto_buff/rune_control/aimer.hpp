#pragma once
#include "tasks/auto_buff/rune_tracker/rune_target.hpp"
#include "tasks/type_common.hpp"
#include <wust_vl/common/utils/trajectory_compensator.hpp>
namespace wust_vision {
namespace auto_buff {
    class Aimer {
    public:
        using Ptr = std::unique_ptr<Aimer>;
        Aimer(
            const YAML::Node& config,
            std::shared_ptr<wust_vl::common::utils::TrajectoryCompensator> trajectory_compensator
        );
        static Ptr create(
            const YAML::Node& config,
            std::shared_ptr<wust_vl::common::utils::TrajectoryCompensator> trajectory_compensator
        ) {
            return std::make_unique<Aimer>(config, trajectory_compensator);
        }
        ~Aimer();
        GimbalCmd aim(const auto_buff::RuneTarget& target, double bullet_speed);

    private:
        struct Impl;
        std::unique_ptr<Impl> _impl;
    };

} // namespace auto_buff
} // namespace wust_vision