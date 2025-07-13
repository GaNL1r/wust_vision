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
#include "common/3rdparty/angles.h"
#include <Eigen/Dense>
#include <algorithm>
#include <array>
#include <cmath>
#include <functional>
#include <vector>

template<int N_X, int N_Z, class PredictFunc, class MeasureFunc>
class UnscentedKalmanFilter {
public:
    using MatrixXX = Eigen::Matrix<double, N_X, N_X>;
    using MatrixZX = Eigen::Matrix<double, N_Z, N_X>;
    using MatrixXZ = Eigen::Matrix<double, N_X, N_Z>;
    using MatrixZZ = Eigen::Matrix<double, N_Z, N_Z>;
    using MatrixX1 = Eigen::Matrix<double, N_X, 1>;
    using MatrixZ1 = Eigen::Matrix<double, N_Z, 1>;

    using UpdateQFunc = std::function<MatrixXX()>;
    using UpdateRFunc = std::function<MatrixZZ(const MatrixZ1&)>;

    explicit UnscentedKalmanFilter(
        const PredictFunc& f,
        const MeasureFunc& h,
        const UpdateQFunc& u_q,
        const UpdateRFunc& u_r,
        const MatrixXX& P0,
        double alpha = 1e-3,
        double beta = 2.0,
        double kappa = 0.0
    ) noexcept:
        f(f),
        h(h),
        update_Q(u_q),
        update_R(u_r),
        P_post(P0) {
        lambda = alpha * alpha * (N_X + kappa) - N_X;
        gamma = std::sqrt(N_X + lambda);

        weights_mean[0] = lambda / (N_X + lambda);
        weights_cov[0] = weights_mean[0] + (1 - alpha * alpha + beta);
        for (int i = 1; i < 2 * N_X + 1; ++i) {
            weights_mean[i] = weights_cov[i] = 1.0 / (2 * (N_X + lambda));
        }

        Xsig_pred.setZero();
    }

    void setState(const MatrixX1& x0) noexcept {
        x_post = x0;
    }

    void setAngleDims(const std::vector<int>& dims) {
        angle_dims_ = dims;
    }

    const MatrixXX& getPriorCovariance() const noexcept {
        return P_pri;
    }
    const MatrixXX& getPosteriorCovariance() const noexcept {
        return P_post;
    }

    MatrixX1 predict() noexcept {
        Q = update_Q();
        generateSigmaPoints(x_post, P_post, Xsig);

        for (int i = 0; i < 2 * N_X + 1; ++i)
            Xsig_pred.col(i) = f(Xsig.col(i));

        x_pri.setZero();
        for (int i = 0; i < 2 * N_X + 1; ++i)
            x_pri += weights_mean[i] * Xsig_pred.col(i);

        P_pri.setZero();
        for (int i = 0; i < 2 * N_X + 1; ++i) {
            auto dx = Xsig_pred.col(i) - x_pri;
            P_pri += weights_cov[i] * dx * dx.transpose();
        }
        P_pri += Q;

        x_post = x_pri;
        return x_pri;
    }

    MatrixX1 update(const MatrixZ1& z) noexcept {
        R = update_R(z);

        Eigen::Matrix<double, N_Z, 2 * N_X + 1> Zsig;
        for (int i = 0; i < 2 * N_X + 1; ++i)
            Zsig.col(i) = h(Xsig_pred.col(i));

        MatrixZ1 z_pred = MatrixZ1::Zero();
        for (int i = 0; i < 2 * N_X + 1; ++i)
            z_pred += weights_mean[i] * Zsig.col(i);

        MatrixZZ S = MatrixZZ::Zero();
        for (int i = 0; i < 2 * N_X + 1; ++i) {
            MatrixZ1 dz = Zsig.col(i) - z_pred;
            for (int idx: angle_dims_)
                dz[idx] = angles::shortest_angular_distance(z_pred[idx], Zsig.col(i)[idx]);
            S += weights_cov[i] * dz * dz.transpose();
        }
        S += R;

        MatrixXZ Tc = MatrixXZ::Zero();
        for (int i = 0; i < 2 * N_X + 1; ++i) {
            MatrixX1 dx = Xsig_pred.col(i) - x_pri;
            MatrixZ1 dz = Zsig.col(i) - z_pred;
            for (int idx: angle_dims_)
                dz[idx] = angles::shortest_angular_distance(z_pred[idx], Zsig.col(i)[idx]);
            Tc += weights_cov[i] * dx * dz.transpose();
        }

        K = Tc * S.inverse();

        MatrixZ1 residual = z - z_pred;
        for (int idx: angle_dims_)
            residual[idx] = angles::shortest_angular_distance(z_pred[idx], z[idx]);
        for (int i = 0; i < N_Z; ++i) {
            if (!std::isfinite(residual[i]))
                residual[i] = 0.0;
            residual[i] = std::clamp(residual[i], -1e2, 1e2);
        }

        x_post = x_pri + K * residual;
        P_post = P_pri - K * S * K.transpose();
        return x_post;
    }

private:
    PredictFunc f;
    MeasureFunc h;
    UpdateQFunc update_Q;
    UpdateRFunc update_R;

    double lambda = 0, gamma = 0;
    std::array<double, 2 * N_X + 1> weights_mean {}, weights_cov {};
    Eigen::Matrix<double, N_X, 2 * N_X + 1> Xsig, Xsig_pred;

    MatrixXX Q = MatrixXX::Zero(), P_pri = MatrixXX::Identity(), P_post = MatrixXX::Identity();
    MatrixZZ R = MatrixZZ::Zero();
    MatrixXZ K = MatrixXZ::Zero();
    MatrixX1 x_pri = MatrixX1::Zero(), x_post = MatrixX1::Zero();

    std::vector<int> angle_dims_;

    void generateSigmaPoints(
        const MatrixX1& x,
        const MatrixXX& P,
        Eigen::Matrix<double, N_X, 2 * N_X + 1>& Xsig_out
    ) {
        Eigen::Matrix<double, N_X, N_X> A = P.llt().matrixL();
        Xsig_out.col(0) = x;
        for (int i = 0; i < N_X; ++i) {
            Xsig_out.col(i + 1) = x + gamma * A.col(i);
            Xsig_out.col(i + 1 + N_X) = x - gamma * A.col(i);
        }
    }
};
