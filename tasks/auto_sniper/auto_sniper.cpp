#include "auto_sniper.hpp"
#include "trajectory_solver.hpp"
namespace auto_sniper {
struct AutoSniper::Impl {
    bool init(const YAML::Node& config) {
        trajectory_solver_ = std::make_unique<TrajectorySolver>(config);
        return true;
    }
    void updatePosBulletSpeed(const Eigen::Vector3d& pos, double bullet_speed) {
        self_position_ = pos;
        bullet_speed_ = bullet_speed;
        has_pos_ = true;
    }
    GimbalCmd solve() {
        GimbalCmd cmd;
        if (!has_pos_) {
            cmd.appera = false;
            return cmd;
        }
        cmd = trajectory_solver_->solve(self_position_, bullet_speed_);
        cmd.appera = true;
        return cmd;
    }
    std::unique_ptr<TrajectorySolver> trajectory_solver_;
    bool has_pos_ = false;
    Eigen::Vector3d self_position_;
    double bullet_speed_ = 14.0;
};
AutoSniper::AutoSniper(): _impl(std::make_unique<Impl>()) {}
AutoSniper::~AutoSniper() {
    _impl.reset();
}
bool AutoSniper::init(const YAML::Node& config) {
    return _impl->init(config);
}
void AutoSniper::updatePosBulletSpeed(const Eigen::Vector3d& pos, double bullet_speed) {
    _impl->updatePosBulletSpeed(pos, bullet_speed);
}
GimbalCmd AutoSniper::solve(double dt_ms) {
    return _impl->solve();
}
} // namespace auto_sniper