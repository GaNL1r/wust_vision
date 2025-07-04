// Copyright 2025 Xiaojian Wu
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#pragma once

#include <Eigen/Dense>
#include <ceres/jet.h>
#include <common/angles.h>
#include <functional>

/**
 * @brief Error-State Extended Kalman Filter (ESEKF) implementation using Ceres Jet for automatic differentiation.
 * 
 * This filter separates the nominal state and error state to improve robustness and numerical stability in 
 * nonlinear systems. It uses automatic differentiation to compute Jacobians of the process and measurement models.
 * 
 * @tparam N_X         Dimension of the full (nominal) state vector.
 * @tparam N_Z         Dimension of the measurement vector.
 * @tparam PredicFunc  Functor type for the process model f: x_{k+1} = f(x_k).
 *                     Must be callable as: `void operator()(const ceres::Jet<double, N_X>*, ceres::Jet<double, N_X>*)`.
 * @tparam MeasureFunc Functor type for the measurement model h: z = h(x).
 *                     Must be callable as: `void operator()(const ceres::Jet<double, N_X>*, ceres::Jet<double, N_Z>*)`.
 */
template<int N_X, int N_Z, class PredicFunc, class MeasureFunc>
class ErrorStateEKF {
public:
    ErrorStateEKF() = default;

    // Matrix aliases
    using MatrixXX = Eigen::Matrix<double, N_X, N_X>; ///< N_X x N_X matrix (square)
    using MatrixZX = Eigen::Matrix<double, N_Z, N_X>; ///< N_Z x N_X matrix
    using MatrixXZ = Eigen::Matrix<double, N_X, N_Z>; ///< N_X x N_Z matrix
    using MatrixZZ = Eigen::Matrix<double, N_Z, N_Z>; ///< N_Z x N_Z matrix
    using MatrixX1 = Eigen::Matrix<double, N_X, 1>; ///< N_X-dimensional state vector
    using MatrixZ1 = Eigen::Matrix<double, N_Z, 1>; ///< N_Z-dimensional measurement vector

    // Functor types
    using UpdateQFunc =
        std::function<MatrixXX()>; ///< Function to produce process noise covariance Q
    using UpdateRFunc =
        std::function<MatrixZZ(const MatrixZ1& z)>; ///< Function to produce measurement noise R

    /**
     * @brief Constructor to initialize the ESEKF with system models and noise definitions.
     *
     * @param f     Process model function f(x).
     * @param h     Measurement model function h(x).
     * @param u_q   Function to return process noise covariance Q.
     * @param u_r   Function to return measurement noise covariance R given measurement z.
     * @param P0    Initial error-state covariance matrix.
     */
    explicit ErrorStateEKF(
        const PredicFunc& f,
        const MeasureFunc& h,
        const UpdateQFunc& u_q,
        const UpdateRFunc& u_r,
        const MatrixXX& P0
    ) noexcept:
        f(f),
        h(h),
        update_Q(u_q),
        update_R(u_r),
        P_delta(P0) // Initial error-state covariance
    {
        F.setZero();
        H.setZero();
    }

    /**
     * @brief Set the injection function that applies the error state to the nominal state.
     * 
     * @param inject_func A function taking (delta_x, x_nominal) and modifying x_nominal in-place.
     */
    void setInjectFunc(std::function<void(const MatrixX1&, MatrixX1&)> inject_func) {
        inject_state = inject_func;
    }

    /**
     * @brief Set the filter's current nominal state estimate.
     * 
     * @param x0 Initial nominal state vector.
     */
    void setState(const MatrixX1& x0) noexcept {
        x_nominal = x0;
        delta_x.setZero();
    }

    /**
     * @brief Override the process model functor.
     * 
     * @param f New process model function f(x).
     */
    void setPredictFunc(const PredicFunc& f) noexcept {
        this->f = f;
    }

    /**
     * @brief Override the measurement model functor.
     * 
     * @param h New measurement model function h(x).
     */
    void setMeasureFunc(const MeasureFunc& h) noexcept {
        this->h = h;
    }

    /**
     * @brief Set the number of Gauss-Newton iterations to run in each update step.
     * 
     * @param num Number of iterations.
     */
    void setIterationNum(int num) {
        iteration_num_ = num;
    }

    /**
     * @brief Specify which measurement dimensions represent angles.
     * 
     * These components will be normalized using `angles::shortest_angular_distance()`.
     * 
     * @param dims Indices of angle components in measurement vector.
     */
    void setAngleDims(const std::vector<int>& dims) {
        angle_dims_ = dims;
    }

    /**
     * @brief Perform the prediction step of the ESEKF.
     * 
     * Propagates the nominal state forward and updates the error-state covariance.
     * 
     * @return Predicted nominal state vector.
     */
    MatrixX1 predict() noexcept {
        // Convert nominal state to Jet for automatic differentiation
        ceres::Jet<double, N_X> x_jet[N_X];
        for (int i = 0; i < N_X; ++i) {
            x_jet[i].a = x_nominal[i];
            x_jet[i].v.setZero();
            x_jet[i].v[i] = 1.0;
        }

        // Propagate through process model
        ceres::Jet<double, N_X> x_pred_jet[N_X];
        f(x_jet, x_pred_jet);

        // Extract predicted state and Jacobian
        MatrixX1 x_pri;
        for (int i = 0; i < N_X; ++i) {
            x_pri[i] = x_pred_jet[i].a;
            F.block(i, 0, 1, N_X) = x_pred_jet[i].v.transpose();
        }

        // Update error covariance
        Q = update_Q();
        P_delta = F * P_delta * F.transpose() + Q;
        P_delta = 0.5 * (P_delta + P_delta.transpose()); // Ensure symmetry

        x_nominal = x_pri;
        delta_x.setZero(); // Reset error

        return x_pri;
    }

    /**
     * @brief Perform the measurement update step of the ESEKF.
     * 
     * Refines the nominal state using the measurement and error-state correction.
     * 
     * @param z Measurement vector.
     * @return Updated nominal state estimate.
     */
    MatrixX1 update(const MatrixZ1& z) noexcept {
        MatrixX1 delta_iter = delta_x;
        MatrixXX P_iter = P_delta;

        for (int iter = 0; iter < iteration_num_; ++iter) {
            // Inject delta into nominal to get full state
            MatrixX1 x_full = x_nominal;
            if (inject_state) {
                inject_state(delta_iter, x_full);
            }

            // Evaluate measurement model
            ceres::Jet<double, N_X> x_jet[N_X];
            for (int i = 0; i < N_X; ++i) {
                x_jet[i].a = x_full[i];
                x_jet[i].v.setZero();
                x_jet[i].v[i] = 1.0;
            }

            ceres::Jet<double, N_X> z_jet[N_Z];
            h(x_jet, z_jet);

            MatrixZ1 z_pred;
            for (int i = 0; i < N_Z; ++i) {
                z_pred[i] = z_jet[i].a;
                H.block(i, 0, 1, N_X) = z_jet[i].v.transpose();
            }

            // Kalman gain and residual
            R = update_R(z);
            MatrixZZ S = H * P_iter * H.transpose() + R;
            S += 1e-6 * MatrixZZ::Identity(); // Numerical stability

            MatrixXZ K = P_iter * H.transpose() * S.inverse();
            MatrixZ1 residual = z - z_pred;

            // Normalize angle residuals
            for (int idx: angle_dims_) {
                residual[idx] = angles::shortest_angular_distance(z_pred[idx], z[idx]);
            }

            // Clamp large jumps
            for (int i = 0; i < residual.size(); ++i) {
                residual[i] = std::clamp(residual[i], -1e2, 1e2);
            }

            // Update error state
            delta_iter += K * residual;
            P_iter = (MatrixXX::Identity() - K * H) * P_iter;
            P_iter = 0.5 * (P_iter + P_iter.transpose());
        }

        // Inject final delta into nominal state
        if (inject_state) {
            inject_state(delta_iter, x_nominal);
        }

        delta_x.setZero();
        P_delta = P_iter;
        return x_nominal;
    }

    /**
     * @brief Get the current nominal state.
     * @return Reference to current nominal state.
     */
    const MatrixX1& getState() const noexcept {
        return x_nominal;
    }

    /**
     * @brief Get the current error-state covariance.
     * @return Reference to error covariance matrix.
     */
    const MatrixXX& getCovariance() const noexcept {
        return P_delta;
    }

private:
    // System models and noise functions
    PredicFunc f; ///< Nonlinear process model
    MeasureFunc h; ///< Nonlinear measurement model
    UpdateQFunc update_Q; ///< Process noise generator
    UpdateRFunc update_R; ///< Measurement noise generator

    // Jacobians
    MatrixXX F = MatrixXX::Zero(); ///< State transition Jacobian
    MatrixZX H = MatrixZX::Zero(); ///< Measurement Jacobian

    // Covariance matrices
    MatrixXX Q = MatrixXX::Zero(); ///< Process noise covariance
    MatrixZZ R = MatrixZZ::Zero(); ///< Measurement noise covariance

    // Filter state
    MatrixX1 x_nominal = MatrixX1::Zero(); ///< Nominal state
    MatrixX1 delta_x = MatrixX1::Zero(); ///< Error state
    MatrixXX P_delta = MatrixXX::Identity(); ///< Error covariance

    // Config
    std::vector<int> angle_dims_; ///< Indices of angular measurement components
    int iteration_num_ = 1; ///< Number of Gauss-Newton iterations

    // Error-state to nominal injection
    std::function<void(const MatrixX1&, MatrixX1&)> inject_state; ///< Injection function
};
