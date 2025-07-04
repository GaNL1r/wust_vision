#pragma once
#include <Eigen/Dense>
#include <algorithm>
#include <array>
#include <cassert>
#include <chrono>
#include <cmath>
#include <numeric>
#include <vector>

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
        prev_unwrapped_(0.0),
        filtered_prev_(0.0),
        prev_filtered_(0.0),
        velocity_estimate_(0.0),
        history_size_(0),
        history_index_(0),
        time_initialized_(false) {
        future_unwrapped_.resize(future_window_);
    }

    double update(
        double raw,
        const std::vector<double>& future_vals,
        std::chrono::steady_clock::time_point time_point
    ) {
        assert(future_vals.size() >= static_cast<size_t>(poly_order_ + 1));
        assert(future_vals.size() >= static_cast<size_t>(future_window_));

        double dt = 1.0 / 1000.0; // fallback default
        if (time_initialized_) {
            dt = std::chrono::duration<double>(time_point - prev_time_).count();
            dt = std::max(1e-6, dt);
        } else {
            time_initialized_ = true;
        }
        prev_time_ = time_point;

        // 1. unwrap 当前值和未来值
        double unwrapped = unwrapAngle(raw, prev_unwrapped_);
        prev_unwrapped_ = unwrapped;

        double last = unwrapped;
        for (int i = 0; i < future_window_; ++i) {
            double uv = unwrapAngle(future_vals[i], last);
            future_unwrapped_[i] = uv;
            last = uv;
        }

        // 2. 判断是否目标切换（当前或未来突变）
        bool jump = std::abs(unwrapped - filtered_prev_) > jump_threshold_;
        int fit_len = future_window_;
        for (int i = 1; i < future_window_; ++i) {
            double d = std::abs(future_unwrapped_[i] - future_unwrapped_[i - 1]);
            if (d > jump_threshold_) {
                jump = true;
                fit_len = i; // 拟合截断
                break;
            }
        }

        if (fit_len < poly_order_ + 1) {
            jump = true; // 拟合点不足
        }

        if (jump) {
            reset(unwrapped);
            prev_filtered_ = unwrapped;
            return unwrapped;
        }

        // 3. 自适应噪声估计
        history_buffer_[history_index_] = unwrapped;
        history_index_ = (history_index_ + 1) % history_max_;
        if (history_size_ < history_max_)
            history_size_++;
        double noise = computeNoiseLevel();

        double alpha = clamp(0.05, 0.5, base_alpha_ + noise);
        double k_ff = clamp(0.1, 1.0, base_k_ff_ - 0.5 * noise);

        // 4. 一阶低通滤波
        double smoothed = alpha * unwrapped + (1.0 - alpha) * filtered_prev_;
        filtered_prev_ = smoothed;

        // 5. 多项式拟合（只用 fit_len 内数据）
        Eigen::MatrixXd X(fit_len, poly_order_ + 1);
        Eigen::VectorXd y(fit_len);

        for (int i = 0; i < fit_len; ++i) {
            double t = dt * (i + 1);
            y(i) = future_unwrapped_[i];
            for (int j = 0; j <= poly_order_; ++j)
                X(i, j) = std::pow(t, j);
        }

        Eigen::VectorXd coeffs = X.colPivHouseholderQr().solve(y);

        // 6. 前馈预测未来一点
        double predict_time = dt; // 未来一帧
        double fitted_future = 0.0;
        for (int j = 0; j <= poly_order_; ++j)
            fitted_future += coeffs(j) * std::pow(predict_time, j);

        // 限幅前馈误差
        double ff_term = clamp(-max_ff_correction_, max_ff_correction_, fitted_future - unwrapped);

        double result = smoothed + k_ff * ff_term;

        // 7. 角速度估计
        velocity_estimate_ = (result - prev_filtered_) / dt;
        prev_filtered_ = result;

        return result;
    }

    double getVelocity() const {
        return velocity_estimate_;
    }

    void reset(double init_val = 0.0) {
        prev_unwrapped_ = init_val;
        filtered_prev_ = init_val;
        prev_filtered_ = init_val;
        velocity_estimate_ = 0.0;
        history_size_ = 0;
        history_index_ = 0;
        time_initialized_ = false;
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
        double mean =
            std::accumulate(history_buffer_.begin(), history_buffer_.begin() + history_size_, 0.0)
            / history_size_;
        double var = 0.0;
        for (size_t i = 0; i < history_size_; ++i)
            var += (history_buffer_[i] - mean) * (history_buffer_[i] - mean);
        return std::sqrt(var / history_size_);
    }

    double clamp(double min_val, double max_val, double v) const {
        return std::max(min_val, std::min(max_val, v));
    }

private:
    int poly_order_;
    int future_window_;
    double base_alpha_, base_k_ff_;
    double jump_threshold_;
    double prev_unwrapped_;
    double filtered_prev_;
    double prev_filtered_;
    double velocity_estimate_;
    std::chrono::steady_clock::time_point prev_time_;
    bool time_initialized_;

    std::vector<double> future_unwrapped_;

    static constexpr size_t history_max_ = 20;
    std::array<double, history_max_> history_buffer_;
    size_t history_size_;
    size_t history_index_;

    static constexpr double max_ff_correction_ = 10.0; // 限制最大前馈值
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
