// Created by Labor 2023.8.25
// Maintained by Labor, Chengfu Zou
// Copyright (C) FYT Vision Group. All rights reserved.
// Copyright 2025 XiaoJian Wu
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
#include <array>
// g2o
#include <g2o/core/auto_differentiation.h>
#include <g2o/core/base_binary_edge.h>
#include <g2o/core/base_unary_edge.h>
#include <g2o/core/base_vertex.h>
#include <g2o/core/optimization_algorithm.h>
#include <g2o/core/optimization_algorithm_factory.h>
#include <g2o/core/sparse_optimizer.h>
// 3rd party
#include <Eigen/Dense>
#include <g2o/types/slam3d/vertex_pointxyz.h>
#include <opencv2/core.hpp>
#include <sophus/se3.hpp>
#include <sophus/so3.hpp>

// project
#include "type/type.hpp"

// Vertex of graph optimization algorithm for the yaw angle
class VertexYaw: public g2o::BaseVertex<1, double> {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

    VertexYaw() = default;
    virtual void setToOriginImpl() override {
        _estimate = 0;
    }
    virtual void oplusImpl(const double* update) override;

    virtual bool read(std::istream& in) override {
        return true;
    }
    virtual bool write(std::ostream& out) const override {
        return true;
    }
};

// Edge of graph optimization algorithm for reporjection error calculation using
// yaw angle and observation
class EdgeProjection:
    public g2o::BaseBinaryEdge<2, Eigen::Vector2d, VertexYaw, g2o::VertexPointXYZ> {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
    using InfoMatrixType = Eigen::Matrix<double, 2, 2>;

    EdgeProjection(
        const Sophus::SO3d& R_camera_imu,
        const Sophus::SO3d& R_pitch,
        const Eigen::Vector3d& t,
        const Eigen::Matrix3d& K
    );
    virtual void computeError() override;

    virtual bool read(std::istream& in) override {
        return true;
    }
    virtual bool write(std::ostream& out) const override {
        return true;
    }

private:
    Sophus::SO3d R_camera_imu_;
    Sophus::SO3d R_pitch_;
    Eigen::Vector3d t_;
    Eigen::Matrix3d K_;
};

class VertexTranslation: public g2o::BaseVertex<3, Eigen::Vector3d> {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    void setToOriginImpl() override;
    void oplusImpl(const double* update) override;
    bool read(std::istream&) override;
    bool write(std::ostream&) const override;
};

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
    );
    void computeError() override;
    bool read(std::istream&) override;
    bool write(std::ostream&) const override;

private:
    Sophus::SO3d R_cam_imu_, R_pitch_;
    Eigen::Vector3d t_cam_arm_;
    Eigen::Matrix3d K_;
    Eigen::Vector3d P1_, P2_;
    double weight_;
};

class EdgeProjectionTvecOnly: public g2o::BaseUnaryEdge<2, Eigen::Vector2d, VertexTranslation> {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    EdgeProjectionTvecOnly(
        const Sophus::SO3d& R_fixed,
        const Eigen::Vector3d& Xw,
        const Eigen::Matrix3d& K
    );
    void computeError() override;
    bool read(std::istream&) override;
    bool write(std::ostream&) const override;

private:
    Sophus::SO3d R_fixed_;
    Eigen::Vector3d Xw_;
    Eigen::Matrix3d K_;
};

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
    );
    void computeError() override;
    bool read(std::istream&) override;
    bool write(std::ostream&) const override;

private:
    Sophus::SO3d R_fixed_;
    Eigen::Vector3d P1_, P2_;
    Eigen::Matrix3d K_;
};

template<typename Func>
inline static double fibonacci(double a, double b, double epsilone, Func concav_upward) {
    if (b < a)
        std::swap(a, b);

    double fn1 = 2, fn2 = 3;
    while ((b - a) / epsilone > fn2) {
        fn2 = fn1 + fn2;
        fn1 = fn2 - fn1;
    }

    double x1 = a + (fn2 - fn1) / fn2 * (b - a), x2 = a + fn1 / fn2 * (b - a);
    double cv1 = concav_upward(x1);
    double cv2 = concav_upward(x2);
    while ((b - a) > epsilone) {
        fn1 = fn2 - fn1;
        fn2 = fn2 - fn1;
        if (cv1 < cv2) {
            b = x2;
            x2 = x1;
            x1 = a + (fn2 - fn1) / fn2 * (b - a);
            cv2 = cv1;
            cv1 = concav_upward(x1);
        } else {
            a = x1;
            x1 = x2;
            x2 = a + fn1 / fn2 * (b - a);
            cv1 = cv2;
            cv2 = concav_upward(x2);
        }
    }
    return (a + b) / 2;
}