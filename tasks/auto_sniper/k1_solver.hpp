#include <Eigen/Dense>
#include <cmath>
#include <memory>
#include <optional>

class K1BallisticSolver {
public:
    using Ptr = std::unique_ptr<K1BallisticSolver>;
    K1BallisticSolver(double k1 = 0.05, double g = 9.81): k1_(k1), g_(g) {}
    static Ptr create(double k1 = 0.05, double g = 9.81) {
        return std::make_unique<K1BallisticSolver>(k1, g);
    }
    std::optional<double> solvePitch(const Eigen::Vector3d& target_pos, double v0) const {
        double x = std::hypot(target_pos.x(), target_pos.y());
        double z = target_pos.z();

        if (x < 1e-6)
            return std::nullopt;

        auto heightError = [&](double pitch) {
            double cos_theta = std::cos(pitch);
            double sin_theta = std::sin(pitch);

            double denom = v0 * cos_theta;
            if (denom <= 1e-6)
                return 1e6;

            double t = -std::log(1.0 - k1_ * x / denom) / k1_;

            if (!std::isfinite(t))
                return 1e6;

            double z_pred =
                ((v0 * sin_theta + g_ / k1_) / k1_) * (1.0 - std::exp(-k1_ * t)) - (g_ / k1_) * t;

            return z_pred - z;
        };

        double left = -0.3;
        double right = 0.6;

        for (int i = 0; i < 60; ++i) {
            double mid = 0.5 * (left + right);
            double err = heightError(mid);

            if (err > 0)
                right = mid;
            else
                left = mid;
        }

        return 0.5 * (left + right);
    }
    std::vector<Eigen::Vector3d> computeTrajectory(
        const Eigen::Vector3d& start,
        const Eigen::Vector3d& target,
        double v0,
        double dt = 0.01 // 每个离散点的时间间隔
    ) {
        std::vector<Eigen::Vector3d> traj;

        Eigen::Vector3d diff = target - start;
        auto pitch_opt = solvePitch(diff, v0);
        if (!pitch_opt.has_value())
            return traj;

        double pitch = pitch_opt.value();
        double yaw = std::atan2(diff.y(), diff.x());

        double k1 = k1_;
        double g = g_;

        double vx = v0 * std::cos(pitch) * std::cos(yaw);
        double vy = v0 * std::cos(pitch) * std::sin(yaw);
        double vz = v0 * std::sin(pitch);

        double t = 0.0;
        Eigen::Vector3d pos = start;

        while (true) {
            double exp_kt = std::exp(-k1 * t);
            pos.x() = start.x() + (vx / k1) * (1 - exp_kt);
            pos.y() = start.y() + (vy / k1) * (1 - exp_kt);
            pos.z() = start.z() + ((vz + g / k1) / k1) * (1 - exp_kt) - (g / k1) * t;

            traj.push_back(pos);

            t += dt;

            double dx = std::abs(pos.x() - start.x());
            double dy = std::abs(pos.y() - start.y());
            double horizontal_dist = std::sqrt(dx * dx + dy * dy);
            double target_dist = std::sqrt(diff.x() * diff.x() + diff.y() * diff.y());

            if (horizontal_dist >= target_dist) {
                break;
            }
        }

        return traj;
    }

private:
    double k1_;
    double g_;
};