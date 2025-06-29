// Created by Labor 2023.8.25
// Maintained by Labor, Chengfu Zou
// Copyright (C) FYT Vision Group. All rights reserved.
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

#ifndef ARMOR_DETECTOR_BA_SOLVER_HPP_
#define ARMOR_DETECTOR_BA_SOLVER_HPP_

// std
#include <array>
#include <cstddef>
#include <tuple>
#include <vector>
// 3rd party
#include <Eigen/Core>
#include <Eigen/Dense>
#include <opencv2/core.hpp>
#include <sophus/so3.hpp>
// #include <std_msgs/msg/float32.hpp>
// g2o
#include <g2o/core/base_multi_edge.h>
#include <g2o/core/base_vertex.h>
#include <g2o/core/optimization_algorithm.h>
#include <g2o/core/optimization_algorithm_factory.h>
#include <g2o/core/optimization_algorithm_levenberg.h>
#include <g2o/core/robust_kernel.h>
#include <g2o/core/sparse_optimizer.h>
// project
#include "detect/graph_optimizer.hpp"
#include "type/type.hpp"
inline double normalizeAngleaa(double angle) {
    while (angle > 180.0)
        angle -= 360.0;
    while (angle < -180.0)
        angle += 360.0;
    return angle;
}

// BA algorithm based Optimizer for the armor pose estimation (Particularly for
// the Yaw angle)
class BaSolver {
public:
    BaSolver(std::array<double, 9>& camera_matrix, std::vector<double>& dist_coeffs);
    ~BaSolver() = default;

    // Solve the armor pose using the BA algorithm, return the optimized rotation
    Eigen::Matrix3d solveBa(
        const ArmorObject& armor,
        const Eigen::Vector3d& t_camera_armor,
        const Eigen::Matrix3d& R_camera_armor,
        const Eigen::Matrix3d& R_imu_camera,
        int type_number
    ) noexcept;
    Eigen::Vector3d solveBa_t(
        const ArmorObject& armor,
        const Eigen::Vector3d& t_camera_armor_init,
        const Eigen::Matrix3d& R_camera_armor,
        const Eigen::Matrix3d& R_imu_camera,
        int type_number
    ) noexcept;

private:
    Eigen::Matrix3d K_;
    g2o::SparseOptimizer optimizer_;
    g2o::OptimizationAlgorithmProperty solver_property_;
    g2o::OptimizationAlgorithmLevenberg* lm_algorithm_;
};
// 在 BaSolver.h 中，声明 EdgeSymmetry
#include <g2o/core/base_unary_edge.h>

class EdgeSymmetry: public g2o::BaseUnaryEdge<2, Eigen::Vector2d, VertexYaw> {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    EdgeSymmetry(
        const Sophus::SO3d& R_cam_imu,
        const Sophus::SO3d& R_pitch,
        const Eigen::Vector3d& t_cam_arm,
        const Eigen::Matrix3d& K,
        const Eigen::Vector3d& P1,
        const Eigen::Vector3d& P2,
        const Eigen::Vector2d& meas,
        double weight = 1.0
    ):
        R_cam_imu_(R_cam_imu),
        R_pitch_(R_pitch),
        t_cam_arm_(t_cam_arm),
        K_(K),
        P1_(P1),
        P2_(P2),
        weight_(weight) {
        setMeasurement(meas);
        information() = Eigen::Matrix2d::Identity() * weight_;
    }

    // 计算残差：两顶点投影中心与测量中心的差
    void computeError() override {
        const VertexYaw* v = static_cast<const VertexYaw*>(_vertices[0]);
        double yaw = v->estimate();
        Sophus::SO3d R_yaw = Sophus::SO3d::exp(Eigen::Vector3d(0, 0, yaw));
        Eigen::Matrix3d R = (R_cam_imu_ * R_yaw * R_pitch_).matrix();

        // 投影 P1 与 P2
        Eigen::Vector3d Xc1 = R * P1_ + t_cam_arm_;
        Eigen::Vector3d Xc2 = R * P2_ + t_cam_arm_;

        Eigen::Vector3d p1 = K_ * Xc1; // 相机坐标到像素
        Eigen::Vector3d p2 = K_ * Xc2;

        Eigen::Vector2d pi(p1.x() / p1.z(), p1.y() / p1.z());
        Eigen::Vector2d pj(p2.x() / p2.z(), p2.y() / p2.z());

        // 残差 = (pi + pj)/2 - meas
        _error = ((pi + pj) * 0.5) - _measurement;
    }

    bool read(std::istream& in) override {
        return false;
    }
    bool write(std::ostream& out) const override {
        return false;
    }

private:
    Sophus::SO3d R_cam_imu_, R_pitch_;
    Eigen::Vector3d t_cam_arm_;
    Eigen::Matrix3d K_;
    Eigen::Vector3d P1_, P2_;
    double weight_;
};
// VertexTranslation.h
#include <Eigen/Core>
#include <g2o/core/base_vertex.h>

class VertexTranslation: public g2o::BaseVertex<3, Eigen::Vector3d> {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    // 重置为零向量
    void setToOriginImpl() override {
        _estimate.setZero();
    }
    // 更新: 添加增量
    void oplusImpl(const double* update) override {
        Eigen::Map<const Eigen::Vector3d> v(update);
        _estimate += v;
    }
    bool read(std::istream& /*is*/) override {
        return false;
    }
    bool write(std::ostream& /*os*/) const override {
        return false;
    }
};

// 2) 仅优化 tvec 的投影边
class EdgeProjectionTvecOnly: public g2o::BaseUnaryEdge<2, Eigen::Vector2d, VertexTranslation> {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    EdgeProjectionTvecOnly(
        const Sophus::SO3d& R_fixed,
        const Eigen::Vector3d& Xw,
        const Eigen::Matrix3d& K
    ):
        R_fixed_(R_fixed),
        Xw_(Xw),
        K_(K) {}

    void computeError() override {
        const auto* v_t = static_cast<const VertexTranslation*>(_vertices[0]);
        Eigen::Vector3d Pc = R_fixed_ * Xw_ + v_t->estimate();
        Eigen::Vector3d p = K_ * Pc;
        Eigen::Vector2d uv(p.x() / p.z(), p.y() / p.z());
        _error = uv - _measurement;
    }
    bool read(std::istream&) override {
        return false;
    }
    bool write(std::ostream&) const override {
        return false;
    }

private:
    Sophus::SO3d R_fixed_;
    Eigen::Vector3d Xw_;
    Eigen::Matrix3d K_;
};

// 3) 仅优化 tvec 的对称误差边
class EdgeSymmetryTvecOnly: public g2o::BaseUnaryEdge<2, Eigen::Vector2d, VertexTranslation> {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    EdgeSymmetryTvecOnly(
        const Sophus::SO3d& R_fixed,
        const Eigen::Vector3d& P1,
        const Eigen::Vector3d& P2,
        const Eigen::Vector2d& measCenter,
        const Eigen::Matrix3d& K,
        double weight = 1.0
    ):
        R_fixed_(R_fixed),
        P1_(P1),
        P2_(P2),
        K_(K) {
        setMeasurement(measCenter);
        information() = Eigen::Matrix2d::Identity() * weight;
    }

    void computeError() override {
        const auto* v_t = static_cast<const VertexTranslation*>(_vertices[0]);
        Eigen::Vector3d Pc1 = R_fixed_ * P1_ + v_t->estimate();
        Eigen::Vector3d Pc2 = R_fixed_ * P2_ + v_t->estimate();
        Eigen::Vector2d u1((K_ * Pc1).hnormalized());
        Eigen::Vector2d u2((K_ * Pc2).hnormalized());
        Eigen::Vector2d center = 0.5 * (u1 + u2);
        _error = center - _measurement;
    }
    bool read(std::istream&) override {
        return false;
    }
    bool write(std::ostream&) const override {
        return false;
    }

private:
    Sophus::SO3d R_fixed_;
    Eigen::Vector3d P1_, P2_;
    Eigen::Matrix3d K_;
};

#endif // ARMOR_DETECTOR_BAS_SOLVER_HPP_
