// Copyright Chen Jun 2023. Licensed under the MIT License.
// Copyright xinyang 2021.
//
// Additional modifications and features by Chengfu Zou, Labor. Licensed under
// Apache License 2.0.
//
// Copyright (C) FYT Vision Group. All rights reserved.
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

// std
#include <functional>
// Eigen
#include <Eigen/Dense>
// ceres
#include <ceres/jet.h>
#include <common/angles.h>

// Extended Kalman Filter with auto differentiation
// N_X: state vector dimension
// N_Z: measurement vector dimension
// PredicFunc: process nonlinear vector function
// MeasureFunc: observation nonlinear vector function
#pragma once

// std
#include <functional>
#include <vector>
#include <cassert>
// Eigen
#include <Eigen/Dense>
// ceres
#include <ceres/jet.h>
#include <common/angles.h>

// Extended Kalman Filter with auto differentiation
template<int N_X, int N_Z, class PredicFunc, class MeasureFunc>
class ExtendedKalmanFilter {
public:
    ExtendedKalmanFilter() = default;

    using MatrixXX = Eigen::Matrix<double, N_X, N_X>;
    using MatrixZX = Eigen::Matrix<double, N_Z, N_X>;
    using MatrixXZ = Eigen::Matrix<double, N_X, N_Z>;
    using MatrixZZ = Eigen::Matrix<double, N_Z, N_Z>;
    using MatrixX1 = Eigen::Matrix<double, N_X, 1>;
    using MatrixZ1 = Eigen::Matrix<double, N_Z, 1>;

    using UpdateQFunc = std::function<MatrixXX()>;
    using UpdateRFunc = std::function<MatrixZZ(const MatrixZ1& z)>;

    explicit ExtendedKalmanFilter(
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
        P_post(P0) {
        F.setZero();
        H.setZero();
    }

    void setState(const MatrixX1& x0) noexcept {
        x_post = x0;
    }

    void setPredictFunc(const PredicFunc& f) noexcept {
        this->f = f;
    }

    void setMeasureFunc(const MeasureFunc& h) noexcept {
        this->h = h;
    }

    void setIterationNum(int num) {
        iteration_num_ = num;
    }

    void setAngleDims(const std::vector<int>& dims) {
        angle_dims_ = dims;
    }

    MatrixX1 predict() noexcept {
        ceres::Jet<double, N_X> x_e_jet[N_X];
        for (int i = 0; i < N_X; ++i) {
            x_e_jet[i].a = x_post[i];
            x_e_jet[i].v.setZero(); 
            x_e_jet[i].v[i] = 1.0;
        }

        ceres::Jet<double, N_X> x_p_jet[N_X];
        f(x_e_jet, x_p_jet);

        for (int i = 0; i < N_X; ++i) {
            x_pri[i] = std::isfinite(x_p_jet[i].a) ? x_p_jet[i].a : 0.0; 
            F.block(i, 0, 1, N_X) = x_p_jet[i].v.transpose();
        }

        Q = update_Q();
        P_pri = F * P_post * F.transpose() + Q;
        P_pri = 0.5 * (P_pri + P_pri.transpose());

        x_post = x_pri;
        return x_pri;
    }

    MatrixX1 update(const MatrixZ1& z) noexcept {
        MatrixX1 x_iter = x_post;

        for (int iter = 0; iter < iteration_num_; ++iter) {
            // Jet 初始化
            ceres::Jet<double, N_X> x_p_jet[N_X];
            for (int i = 0; i < N_X; ++i) {
                x_p_jet[i].a = x_iter[i];
                x_p_jet[i].v.setZero(); 
                x_p_jet[i].v[i] = 1.0;
            }

            ceres::Jet<double, N_X> z_p_jet[N_Z];
            h(x_p_jet, z_p_jet);

            MatrixZ1 z_pri;
            for (int i = 0; i < N_Z; ++i) {
                z_pri[i] = std::isfinite(z_p_jet[i].a) ? z_p_jet[i].a : 0.0; 
                H.block(i, 0, 1, N_X) = z_p_jet[i].v.transpose();
            }

            R = update_R(z);
            MatrixZZ S = H * P_pri * H.transpose() + R;
            S += 1e-6 * MatrixZZ::Identity(); 
            K = P_pri * H.transpose() * S.inverse();

            MatrixZ1 residual = z - z_pri;
            for (int idx : angle_dims_) {
                residual[idx] = angles::shortest_angular_distance(z_pri[idx], z[idx]);
            }
            for (int i = 0; i < N_Z; ++i) {
                if (!std::isfinite(residual[i])) residual[i] = 0.0;
                residual[i] = std::clamp(residual[i], -1e2, 1e2); 
            }

            MatrixX1 x_new = x_iter + K * residual;

         
            for (int i = 0; i < N_X; ++i) {
                if (!std::isfinite(x_new[i])) x_new[i] = x_iter[i];
            }

            x_iter = x_new;
        }

        x_post = x_iter;
        for (int i = 0; i < N_X; ++i) {
            if (!std::isfinite(x_post[i])) x_post[i] = 0.0;
        }

        P_post = (MatrixXX::Identity() - K * H) * P_pri;
        P_post = 0.5 * (P_post + P_post.transpose()); 
        return x_post;
    }

private:
    PredicFunc f;
    MeasureFunc h;
    UpdateQFunc update_Q;
    UpdateRFunc update_R;

    MatrixXX F = MatrixXX::Zero();
    MatrixZX H = MatrixZX::Zero();
    MatrixXX Q = MatrixXX::Zero();
    MatrixZZ R = MatrixZZ::Zero();

    MatrixXX P_pri = MatrixXX::Identity();
    MatrixXX P_post = MatrixXX::Identity();
    MatrixXZ K = MatrixXZ::Zero();

    MatrixX1 x_pri = MatrixX1::Zero();
    MatrixX1 x_post = MatrixX1::Zero();

    std::vector<int> angle_dims_;
    int iteration_num_ = 1;
};
