#pragma once
#include "common/gobal.hpp"
#include <Eigen/Dense>
#include <array>
#include <cassert>
#include <chrono>
#include <cmath>
#include <utility>

class AdaptiveFeedforwardFilter1D {
public:
    AdaptiveFeedforwardFilter1D(
        int poly_order = 2,
        int future_window = 5,
        double base_alpha = 0.2,
        double base_k_ff = 0.5,
        double jump_threshold = 2.0
    ):
        poly_order_(poly_order),
        future_window_(future_window),
        base_alpha_(base_alpha),
        base_k_ff_(base_k_ff),
        jump_threshold_(jump_threshold),
        initialized_(false),
        time_initialized_(false),
        prev_unwrapped_(0.0),
        filtered_prev_(0.0),
        prev_filtered_(0.0),
        velocity_estimate_(0.0),
        history_size_(0),
        history_index_(0) {
        future_unwrapped_.resize(future_window_);
    }

    double update(
        double raw,
        const std::vector<double>& future_vals,
        std::chrono::steady_clock::time_point time_point
    ) {
        assert(future_vals.size() >= static_cast<size_t>(poly_order_ + 1));
        assert(future_vals.size() >= static_cast<size_t>(future_window_));

        // --- Step 0: 计算时间差 ---
        double dt = 1.0 / gobal::control_rate;
        if (time_initialized_) {
            dt = std::chrono::duration<double>(time_point - prev_time_).count();
            dt = std::max(1e-6, dt);
        } else {
            time_initialized_ = true;
        }
        prev_time_ = time_point;

        // --- Step 1: unwrap 当前值和未来值 ---
        double unwrapped = unwrapAngle(raw, prev_unwrapped_);
        prev_unwrapped_ = unwrapped;

        double last = unwrapped;
        for (int i = 0; i < future_window_; ++i) {
            double uv = unwrapAngle(future_vals[i], last);
            future_unwrapped_[i] = uv;
            last = uv;
        }

        // --- Step 2: 判断是否目标跳变 ---
        double diff = std::abs(unwrapped - filtered_prev_);
        if (diff > jump_threshold_) {
            reset(unwrapped);
        }

        // --- Step 3: 更新滑动噪声历史 ---
        history_buffer_[history_index_] = unwrapped;
        history_index_ = (history_index_ + 1) % history_max_;
        if (history_size_ < history_max_)
            history_size_++;

        double noise = computeNoiseLevel();

        // --- Step 4: 滤波参数自适应调整 ---
        double alpha = clamp(0.05, 0.5, base_alpha_ + noise);
        double k_ff = clamp(0.1, 1.0, base_k_ff_ - 0.5 * noise);

        // --- Step 5: 一阶低通滤波 ---
        double smoothed = alpha * unwrapped + (1.0 - alpha) * filtered_prev_;
        filtered_prev_ = smoothed;

        // --- Step 6: 多项式拟合未来趋势 ---
        Eigen::MatrixXd X(future_window_, poly_order_ + 1);
        Eigen::VectorXd y(future_window_);

        for (int i = 0; i < future_window_; ++i) {
            double t = dt * (i + 1);
            y(i) = future_unwrapped_[i];
            for (int j = 0; j <= poly_order_; ++j)
                X(i, j) = std::pow(t, j);
        }

        Eigen::VectorXd coeffs = X.colPivHouseholderQr().solve(y);

        double fitted_now = coeffs(0); // t=0 处估计值

        // --- Step 7: 前馈叠加 ---
        double result = smoothed + k_ff * (fitted_now - unwrapped);

        // --- Step 8: 角速度估计 ---
        velocity_estimate_ = (result - prev_filtered_) / dt;
        prev_filtered_ = result;

        return result;
    }

    double getVelocity() const {
        return velocity_estimate_;
    }

    void reset(double init_val = 0.0) {
        filtered_prev_ = init_val;
        prev_filtered_ = init_val;
        prev_unwrapped_ = init_val;
        velocity_estimate_ = 0.0;
        history_size_ = 0;
        history_index_ = 0;
        time_initialized_ = false;
        initialized_ = true;
    }

private:
    double unwrapAngle(double angle, double ref) const {
        double diff = angle - ref;
        diff = std::fmod(diff + 540.0, 360.0) - 180.0;
        return ref + diff;
    }

    double computeNoiseLevel() const {
        if (history_size_ < 3)
            return 0.0;
        double sum = 0.0;
        for (size_t i = 0; i < history_size_; ++i)
            sum += history_buffer_[i];
        double mean = sum / static_cast<double>(history_size_);
        double var = 0.0;
        for (size_t i = 0; i < history_size_; ++i)
            var += (history_buffer_[i] - mean) * (history_buffer_[i] - mean);
        var /= static_cast<double>(history_size_);
        return std::sqrt(var);
    }

    double clamp(double min_val, double max_val, double v) const {
        return std::max(min_val, std::min(max_val, v));
    }

private:
    // 参数
    int poly_order_;
    int future_window_;
    double base_alpha_, base_k_ff_;
    double jump_threshold_;

    // 状态
    bool initialized_;
    bool time_initialized_;
    double prev_unwrapped_;
    double filtered_prev_;
    double prev_filtered_;
    double velocity_estimate_;

    std::chrono::steady_clock::time_point prev_time_;

    // 历史缓存
    static constexpr size_t history_max_ = 20;
    std::array<double, history_max_> history_buffer_;
    size_t history_size_;
    size_t history_index_;

    // 避免重复分配的缓存
    std::vector<double> future_unwrapped_;
};

// ------------------ 包装类处理 Yaw/Pitch ---------------------
class ControlFilter {
public:
    ControlFilter(int order, int window, double alpha, double kff, double jump):
        yaw_filter_(order, window, alpha, kff, jump),
        pitch_filter_(order, window, alpha, kff, jump) {}

    std::pair<double, double> update(
        double raw_yaw,
        double raw_pitch,
        const std::vector<double>& future_yaw,
        const std::vector<double>& future_pitch,
        std::chrono::steady_clock::time_point time_point
    ) {
        double yaw_out = yaw_filter_.update(raw_yaw, future_yaw, time_point);
        double pitch_out = pitch_filter_.update(raw_pitch, future_pitch, time_point);
        return { yaw_out, pitch_out };
    }

    std::pair<double, double> getVelocity() const {
        return { yaw_filter_.getVelocity(), pitch_filter_.getVelocity() };
    }

    void reset(double yaw = 0.0, double pitch = 0.0) {
        yaw_filter_.reset(yaw);
        pitch_filter_.reset(pitch);
    }

private:
    AdaptiveFeedforwardFilter1D yaw_filter_;
    AdaptiveFeedforwardFilter1D pitch_filter_;
};
