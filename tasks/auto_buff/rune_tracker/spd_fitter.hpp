#pragma once
#include <vector>
#include <cmath>
#include <iostream>
#include <limits>
#include <algorithm>
#include <Eigen/Dense>
#include <ceres/ceres.h>

class SinSpeedFitter {
public:
    struct P { double a, w, t0; };

    SinSpeedFitter() = default;
    SinSpeedFitter(const SinSpeedFitter&) = default;
    SinSpeedFitter& operator=(const SinSpeedFitter& other) {
        if (this != &other) {
            params_ = other.params_;
            times_ = other.times_;
            speeds_ = other.speeds_;
            has_angle_ref_ = other.has_angle_ref_;
            angle_ref_time_ = other.angle_ref_time_;
            angle_ref_value_ = other.angle_ref_value_;
            sign_ = other.sign_;
        }
        return *this;
    }

    // --- 数据输入 ---
    void update(double time_s, double speed_rads) {
        // 自动插入保持时间有序
        auto it = std::lower_bound(times_.begin(), times_.end(), time_s);
        size_t idx = std::distance(times_.begin(), it);
        times_.insert(it, time_s);
        speeds_.insert(speeds_.begin() + idx, speed_rads);
    }

    // --- 拟合入口 ---
    bool fit(bool verbose = false) {
        if (times_.size() < 3) {
            std::cerr << "[SinSpeedFitter] need >=3 samples\n";
            return false;
        }

        // 1. 按时间排序并去重
        std::vector<std::pair<double, double>> tmp;
        tmp.reserve(times_.size());
        for (size_t i = 0; i < times_.size(); ++i)
            tmp.emplace_back(times_[i], speeds_[i]);
        std::sort(tmp.begin(), tmp.end());

        std::vector<double> t_unique, s_unique;
        t_unique.reserve(tmp.size());
        s_unique.reserve(tmp.size());
        double last_t = std::numeric_limits<double>::quiet_NaN();
        for (auto& [t, s] : tmp) {
            if (t_unique.empty() || std::abs(t - last_t) > 1e-9) {
                t_unique.push_back(t);
                s_unique.push_back(s);
                last_t = t;
            }
        }

        // 2. 尝试正负号拟合
        double err_pos = fit_with_sign(+1, t_unique, s_unique, verbose);
        double err_neg = fit_with_sign(-1, t_unique, s_unique, verbose);
        sign_ = (err_pos <= err_neg) ? +1 : -1;

        return true;
    }

    // --- 预测速度 ---
    double predictSpeed(double t) const {
        double a = params_.a;
        double w = params_.w;
        double b = 2.090 - a;
        return sign_ * (a * std::sin(w * (t - params_.t0)) + b);
    }

    // --- 预测角度 ---
    double predictAngle(double t) const {
        if (!has_angle_ref_) return 0.0;
        double a = params_.a;
        double w = params_.w;
        double b = 2.090 - a;
        double theta = sign_ * (-a / w * std::cos(w * (t - params_.t0)) + b * (t - params_.t0));
        double theta_ref = sign_ * (-a / w * std::cos(w * (angle_ref_time_ - params_.t0)) + b * (angle_ref_time_ - params_.t0));
        return angle_ref_value_ + (theta - theta_ref);
    }

    // --- 设置角度参考 ---
    void setAngleRef(double time_s, double angle_rad) {
        angle_ref_time_ = time_s;
        angle_ref_value_ = angle_rad;
        has_angle_ref_ = true;
    }

    const P& params() const { return params_; }
    int sign() const { return sign_; }

private:
    // --- 残差模型 ---
    struct SinResidual {
        SinResidual(double t, double s, int sign)
            : t_(t), s_(s), sign_(sign) {}

        template <typename T>
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

    double fit_with_sign(int sgn,
                         const std::vector<double>& t_unique,
                         const std::vector<double>& s_unique,
                         bool verbose)
    {
        ceres::Problem problem;
        double params[3] = {params_.a, params_.w, params_.t0};

        for (size_t i = 0; i < t_unique.size(); ++i) {
            problem.AddResidualBlock(
                new ceres::AutoDiffCostFunction<SinResidual, 1, 3>(
                    new SinResidual(t_unique[i], s_unique[i], sgn)),
                nullptr, params);
        }

        // 参数约束
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

        // 计算拟合误差
        double err_sum = 0.0;
        for (size_t i = 0; i < t_unique.size(); ++i) {
            double pred = sgn * (params[0] * std::sin(params[1] * (t_unique[i] - params[2])) +
                                 (2.090 - params[0]));
            double e = s_unique[i] - pred;
            err_sum += e * e;
        }

        if (verbose)
            std::cout << (sgn > 0 ? "[+] " : "[-] ") << summary.BriefReport()
                      << "  error=" << err_sum << std::endl;

        params_.a = params[0];
        params_.w = params[1];
        params_.t0 = params[2];
        return err_sum;
    }

private:
    P params_{1.0, 1.9, 0.0};
    std::vector<double> times_;
    std::vector<double> speeds_;
    bool has_angle_ref_ = false;
    double angle_ref_time_ = 0.0;
    double angle_ref_value_ = 0.0;
    int sign_ = 1;

    const double a_min_ = 0.780;
    const double a_max_ = 1.045;
    const double w_min_ = 1.884;
    const double w_max_ = 2.000;
};
