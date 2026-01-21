#pragma once
#include "3rdparty/angles.h"
#include <Eigen/Dense>
#include <vector>
namespace auto_aim {

template<typename T>
concept HasStaticLerp = requires(const T& a, const T& b, double t) {
    {
        T::lerp(a, b, t)
        } -> std::same_as<T>;
};
template<HasStaticLerp PointT>
class Trajectory {
public:
    void push_back(const PointT& p, double dt) {
        if (cp_vec.empty()) {
            cp_vec.push_back(p);
            dt_vec.push_back(dt);

            prefix_time.clear();
            prefix_time.push_back(0.0); // t = 0
            total_duration_ = 0.0;
            return;
        }

        cp_vec.push_back(p);
        dt_vec.push_back(dt);

        double last_t = prefix_time.back();
        double new_t = last_t + dt;

        prefix_time.push_back(new_t);
        total_duration_ = new_t;
    }

    void reserve(int n) {
        cp_vec.reserve(n);
        dt_vec.reserve(n);
        prefix_time.reserve(n);
    }
    void set(const std::vector<PointT>& c, const std::vector<double>& t) {
        cp_vec = c;
        dt_vec = t;

        prefix_time.resize(dt_vec.size() + 1);
        prefix_time[0] = 0.0;
        for (size_t i = 0; i < dt_vec.size(); ++i)
            prefix_time[i + 1] = prefix_time[i] + dt_vec[i];

        total_duration_ = prefix_time.back();
    }

    PointT getStateAtTime(double t) const {
        if (cp_vec.empty())
            return PointT {};

        if (t <= 0.0)
            return cp_vec.front();

        if (t >= total_duration_)
            return cp_vec.back();

        auto it = std::lower_bound(prefix_time.begin(), prefix_time.end(), t);
        int i1 = int(it - prefix_time.begin());
        int i0 = i1 - 1;

        double dt = dt_vec[i0];
        if (dt <= 1e-9)
            return cp_vec[i0];

        double a = (t - prefix_time[i0]) / dt;
        a = std::clamp(a, 0.0, 1.0);

        return PointT::lerp(cp_vec[i0], cp_vec[i1], a);
    }

    double getTotalDuration() const {
        return total_duration_;
    }
    int getSize() const {
        return int(cp_vec.size());
    }

    std::vector<PointT> cp_vec;
    std::vector<double> dt_vec;
    std::vector<double> prefix_time;
    double total_duration_ { 0.0 };
};

} // namespace auto_aim
