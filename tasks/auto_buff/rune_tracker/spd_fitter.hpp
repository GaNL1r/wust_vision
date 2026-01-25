#pragma once
#include <Eigen/Dense>
#include <algorithm>
#include <atomic>
#include <ceres/ceres.h>
#include <cmath>
#include <iostream>
#include <limits>
#include <mutex>
#include <thread>
#include <vector>
namespace wust_vision {
namespace auto_buff {
    class SinSpeedFitter {
    public:
        struct P {
            double a, w, t0;
        };

        SinSpeedFitter() = default;

        SinSpeedFitter(const SinSpeedFitter& other) {
            std::scoped_lock lock(other.mtx_);
            params_ = other.params_;
            times_ = other.times_;
            speeds_ = other.speeds_;
            has_angle_ref_ = other.has_angle_ref_;
            angle_ref_time_ = other.angle_ref_time_;
            angle_ref_value_ = other.angle_ref_value_;
            sign_ = other.sign_;
            fitting_ = false;
        }

        SinSpeedFitter& operator=(const SinSpeedFitter& other) {
            if (this != &other) {
                std::scoped_lock lock(mtx_, other.mtx_);
                params_ = other.params_;
                times_ = other.times_;
                speeds_ = other.speeds_;
                has_angle_ref_ = other.has_angle_ref_;
                angle_ref_time_ = other.angle_ref_time_;
                angle_ref_value_ = other.angle_ref_value_;
                sign_ = other.sign_;
                fitting_ = false;
            }
            return *this;
        }

        void update(double time_s, double speed_rads) {
            std::scoped_lock lock(mtx_);

            auto it = std::lower_bound(times_.begin(), times_.end(), time_s);
            size_t idx = std::distance(times_.begin(), it);
            times_.insert(it, time_s);
            speeds_.insert(speeds_.begin() + idx, speed_rads);

            const double window_sec = 5.0;
            if (!times_.empty()) {
                double latest = times_.back();
                while (!times_.empty() && latest - times_.front() > window_sec) {
                    times_.erase(times_.begin());
                    speeds_.erase(speeds_.begin());
                }
            }
        }

        void fit(bool verbose = false) {
            std::scoped_lock lock(mtx_);
            fitImpl(verbose);
        }
        void fitAsync(bool verbose = false) {
            if (fitting_.exchange(true)) {
                if (verbose)
                    std::cout << "[SinSpeedFitter] Previous async fit still running, skip.\n";
                return;
            }

            std::vector<double> t_copy, s_copy;
            P params_snapshot;
            {
                std::scoped_lock lock(mtx_);
                t_copy = times_;
                s_copy = speeds_;
                params_snapshot = params_;
            }

            std::thread([this, t_copy, s_copy, params_snapshot, verbose]() {
                fitImpl(verbose, &t_copy, &s_copy, &params_snapshot);
                fitting_ = false;
            }).detach();
        }

        double predictSpeed(double t) const {
            std::scoped_lock lock(mtx_);
            double a = params_.a;
            double w = params_.w;
            double b = 2.090 - a;
            return sign_ * (a * std::sin(w * (t - params_.t0)) + b);
        }

        double predictAngle(double t) const {
            std::scoped_lock lock(mtx_);
            if (!has_angle_ref_)
                return 0.0;
            double a = params_.a;
            double w = params_.w;
            double b = 2.090 - a;
            double theta = sign_ * (-a / w * std::cos(w * (t - params_.t0)) + b * (t - params_.t0));
            double theta_ref = sign_
                * (-a / w * std::cos(w * (angle_ref_time_ - params_.t0))
                   + b * (angle_ref_time_ - params_.t0));
            return angle_ref_value_ + (theta - theta_ref);
        }

        void setAngleRef(double time_s, double angle_rad) {
            std::scoped_lock lock(mtx_);
            angle_ref_time_ = time_s;
            angle_ref_value_ = angle_rad;
            has_angle_ref_ = true;
        }

        const P& params() const {
            return params_;
        }
        int sign() const {
            return sign_;
        }
        bool isFitting() const {
            return fitting_.load();
        }

    private:
        struct SinResidual {
            SinResidual(double t, double s, int sign): t_(t), s_(s), sign_(sign) {}
            template<typename T>
            bool operator()(const T* const p, T* residual) const {
                const T& a = p[0];
                const T& w = p[1];
                const T& t0 = p[2];
                T b = T(2.090) - a;
                T pred = T(sign_) * (a * sin(w * (T(t_) - t0)) + b);
                residual[0] = T(s_) - pred;
                return true;
            }
            double t_, s_;
            int sign_;
        };

        bool fitImpl(
            bool verbose,
            const std::vector<double>* t_ptr = nullptr,
            const std::vector<double>* s_ptr = nullptr,
            const P* params_snapshot = nullptr
        ) {
            const auto& t_in = t_ptr ? *t_ptr : times_;
            const auto& s_in = s_ptr ? *s_ptr : speeds_;

            if (t_in.size() < 3) {
                if (verbose)
                    std::cerr << "[SinSpeedFitter] need >=3 samples\n";
                return false;
            }

            std::vector<std::pair<double, double>> tmp;
            tmp.reserve(t_in.size());
            for (size_t i = 0; i < t_in.size(); ++i)
                tmp.emplace_back(t_in[i], s_in[i]);
            std::sort(tmp.begin(), tmp.end());

            std::vector<double> t_unique, s_unique;
            t_unique.reserve(tmp.size());
            s_unique.reserve(tmp.size());
            double last_t = std::numeric_limits<double>::quiet_NaN();
            for (auto& [t, s]: tmp) {
                if (t_unique.empty() || std::abs(t - last_t) > 1e-9) {
                    t_unique.push_back(t);
                    s_unique.push_back(s);
                    last_t = t;
                }
            }

            P params_initial = params_snapshot ? *params_snapshot : params_;

            double err_pos = fit_with_sign(+1, t_unique, s_unique, params_initial, verbose);
            double err_neg = fit_with_sign(-1, t_unique, s_unique, params_initial, verbose);

            std::scoped_lock lock(mtx_);
            sign_ = (err_pos <= err_neg) ? +1 : -1;
            return true;
        }

        double fit_with_sign(
            int sgn,
            const std::vector<double>& t_unique,
            const std::vector<double>& s_unique,
            P params_initial,
            bool verbose
        ) {
            ceres::Problem problem;
            double params[3] = { params_initial.a, params_initial.w, params_initial.t0 };
            for (size_t i = 0; i < t_unique.size(); ++i) {
                problem.AddResidualBlock(
                    new ceres::AutoDiffCostFunction<SinResidual, 1, 3>(
                        new SinResidual(t_unique[i], s_unique[i], sgn)
                    ),
                    nullptr,
                    params
                );
            }

            problem.SetParameterLowerBound(params, 0, a_min_);
            problem.SetParameterUpperBound(params, 0, a_max_);
            problem.SetParameterLowerBound(params, 1, w_min_);
            problem.SetParameterUpperBound(params, 1, w_max_);

            ceres::Solver::Options options;
            options.linear_solver_type = ceres::DENSE_QR;
            options.max_num_iterations = 100;
            options.minimizer_progress_to_stdout = verbose;

            ceres::Solver::Summary summary;
            ceres::Solve(options, &problem, &summary);

            double err_sum = 0.0;
            for (size_t i = 0; i < t_unique.size(); ++i) {
                double pred = sgn
                    * (params[0] * std::sin(params[1] * (t_unique[i] - params[2]))
                       + (2.090 - params[0]));
                double e = s_unique[i] - pred;
                err_sum += e * e;
            }

            if (verbose)
                std::cout << (sgn > 0 ? "[+] " : "[-] ") << summary.BriefReport()
                          << "  error=" << err_sum << std::endl;

            std::scoped_lock lock(mtx_);
            params_.a = params[0];
            params_.w = params[1];
            params_.t0 = params[2];
            return err_sum;
        }

    private:
        mutable std::mutex mtx_;
        P params_ { 1.0, 1.9, 0.0 };
        std::vector<double> times_;
        std::vector<double> speeds_;
        bool has_angle_ref_ = false;
        double angle_ref_time_ = 0.0;
        double angle_ref_value_ = 0.0;
        int sign_ = 1;
        std::atomic<bool> fitting_ { false };

        const double a_min_ = 0.780;
        const double a_max_ = 1.045;
        const double w_min_ = 1.884;
        const double w_max_ = 2.000;
    };
} // namespace auto_buff
} // namespace wust_vision