// Created by Labor 2023.8.25
// Maintained by Chengfu Zou, Labor
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

#include "detect/graph_optimizer.hpp"
// std
#include <algorithm>
// third party
#include <Eigen/Core>
#include <g2o/types/slam3d/vertex_pointxyz.h>
#include <sophus/so3.hpp>
// project
#include "common/utils.hpp"
#include "type/type.hpp"

void VertexYaw::oplusImpl(const double* update) {
    Sophus::SO3d R_yaw = Sophus::SO3d::exp(Eigen::Vector3d(0, 0, update[0]))
        * Sophus::SO3d::exp(Eigen::Vector3d(0, 0, _estimate));
    _estimate = R_yaw.log()(2);
}

EdgeProjection::EdgeProjection(
    const Sophus::SO3d& R_camera_imu,
    const Sophus::SO3d& R_pitch,
    const Eigen::Vector3d& t,
    const Eigen::Matrix3d& K
):
    R_camera_imu_(R_camera_imu),
    R_pitch_(R_pitch),
    t_(t),
    K_(K) {}

void EdgeProjection::computeError() {
    // Get the rotation
    double yaw = static_cast<VertexYaw*>(_vertices[0])->estimate();
    Sophus::SO3d R_yaw = Sophus::SO3d::exp(Eigen::Vector3d(0, 0, yaw));
    Sophus::SO3d R = R_camera_imu_ * R_yaw * R_pitch_;

    // Get the 3D point
    Eigen::Vector3d p_3d = static_cast<g2o::VertexPointXYZ*>(_vertices[1])->estimate();

    // Get the observed 2D point
    Eigen::Vector2d obs(_measurement);

    // Project the 3D point to the 2D point
    Eigen::Vector3d p_2d = R * p_3d + t_;
    p_2d = K_ * (p_2d / p_2d.z());

    // Calculate the error
    _error = obs - p_2d.head<2>();
}

void VertexTranslation::setToOriginImpl() {
    _estimate.setZero();
}

void VertexTranslation::oplusImpl(const double* update) {
    Eigen::Map<const Eigen::Vector3d> v(update);
    _estimate += v;
}

bool VertexTranslation::read(std::istream&) {
    return false;
}
bool VertexTranslation::write(std::ostream&) const {
    return false;
}

EdgeSymmetry::EdgeSymmetry(
    const Sophus::SO3d& R_cam_imu,
    const Sophus::SO3d& R_pitch,
    const Eigen::Vector3d& t_cam_arm,
    const Eigen::Matrix3d& K,
    const Eigen::Vector3d& P1,
    const Eigen::Vector3d& P2,
    const Eigen::Vector2d& meas,
    double weight
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

void EdgeSymmetry::computeError() {
    const VertexYaw* v = static_cast<const VertexYaw*>(_vertices[0]);
    double yaw = v->estimate();
    Sophus::SO3d R_yaw = Sophus::SO3d::exp(Eigen::Vector3d(0, 0, yaw));
    Eigen::Matrix3d R = (R_cam_imu_ * R_yaw * R_pitch_).matrix();

    Eigen::Vector3d Xc1 = R * P1_ + t_cam_arm_;
    Eigen::Vector3d Xc2 = R * P2_ + t_cam_arm_;

    Eigen::Vector3d p1 = K_ * Xc1;
    Eigen::Vector3d p2 = K_ * Xc2;

    Eigen::Vector2d pi(p1.x() / p1.z(), p1.y() / p1.z());
    Eigen::Vector2d pj(p2.x() / p2.z(), p2.y() / p2.z());

    _error = ((pi + pj) * 0.5) - _measurement;
}

bool EdgeSymmetry::read(std::istream&) {
    return false;
}
bool EdgeSymmetry::write(std::ostream&) const {
    return false;
}

EdgeProjectionTvecOnly::EdgeProjectionTvecOnly(
    const Sophus::SO3d& R_fixed,
    const Eigen::Vector3d& Xw,
    const Eigen::Matrix3d& K
):
    R_fixed_(R_fixed),
    Xw_(Xw),
    K_(K) {}

void EdgeProjectionTvecOnly::computeError() {
    const auto* v_t = static_cast<const VertexTranslation*>(_vertices[0]);
    Eigen::Vector3d Pc = R_fixed_ * Xw_ + v_t->estimate();
    Eigen::Vector3d p = K_ * Pc;
    Eigen::Vector2d uv(p.x() / p.z(), p.y() / p.z());
    _error = uv - _measurement;
}

bool EdgeProjectionTvecOnly::read(std::istream&) {
    return false;
}
bool EdgeProjectionTvecOnly::write(std::ostream&) const {
    return false;
}

EdgeSymmetryTvecOnly::EdgeSymmetryTvecOnly(
    const Sophus::SO3d& R_fixed,
    const Eigen::Vector3d& P1,
    const Eigen::Vector3d& P2,
    const Eigen::Vector2d& measCenter,
    const Eigen::Matrix3d& K,
    double weight
):
    R_fixed_(R_fixed),
    P1_(P1),
    P2_(P2),
    K_(K) {
    setMeasurement(measCenter);
    information() = Eigen::Matrix2d::Identity() * weight;
}

void EdgeSymmetryTvecOnly::computeError() {
    const auto* v_t = static_cast<const VertexTranslation*>(_vertices[0]);
    Eigen::Vector3d Pc1 = R_fixed_ * P1_ + v_t->estimate();
    Eigen::Vector3d Pc2 = R_fixed_ * P2_ + v_t->estimate();
    Eigen::Vector2d u1((K_ * Pc1).hnormalized());
    Eigen::Vector2d u2((K_ * Pc2).hnormalized());
    Eigen::Vector2d center = 0.5 * (u1 + u2);
    _error = center - _measurement;
}

bool EdgeSymmetryTvecOnly::read(std::istream&) {
    return false;
}
bool EdgeSymmetryTvecOnly::write(std::ostream&) const {
    return false;
}
void VertexDistance::setToOriginImpl() {
    _estimate = 1.0;
}
void VertexDistance::oplusImpl(const double* update) {
    _estimate += update[0];
}

bool VertexDistance::read(std::istream&) {
    return false;
}
bool VertexDistance::write(std::ostream&) const {
    return false;
}

Eigen::Vector3d VertexDistance::translation() const {
    return _estimate * dir_;
}

Eigen::Vector3d VertexDistance::direction() const {
    return dir_;
}
EdgeProjectionDistanceOnly::EdgeProjectionDistanceOnly(
    const Sophus::SO3d& R,
    const Eigen::Vector3d& pt_obj,
    const Eigen::Matrix3d& K
):
    R_wc_(R),
    pt_obj_(pt_obj),
    K_(K) {}
void EdgeProjectionDistanceOnly::computeError() {
    const VertexDistance* v = static_cast<const VertexDistance*>(_vertices[0]);
    Eigen::Vector3d t_wc = v->translation();
    Eigen::Vector3d pt_cam = R_wc_ * pt_obj_ + t_wc;

    Eigen::Vector2d proj = project(pt_cam);
    _error = proj - _measurement;
}

Eigen::Vector2d EdgeProjectionDistanceOnly::project(const Eigen::Vector3d& pt_cam) const {
    Eigen::Vector3d proj = K_ * pt_cam;
    return proj.hnormalized();
}

bool EdgeProjectionDistanceOnly::read(std::istream&) {
    return false;
}
bool EdgeProjectionDistanceOnly::write(std::ostream&) const {
    return false;
}
EdgeSymmetryDistanceOnly::EdgeSymmetryDistanceOnly(
    const Sophus::SO3d& R_wc,
    const Eigen::Vector3d& p1,
    const Eigen::Vector3d& p2,
    const Eigen::Vector2d& meas_center,
    const Eigen::Matrix3d& K,
    double weight
):
    R_wc_(R_wc),
    p1_(p1),
    p2_(p2),
    meas_center_(meas_center),
    K_(K),
    weight_(weight) {}

void EdgeSymmetryDistanceOnly::computeError() {
    const VertexDistance* v = static_cast<const VertexDistance*>(_vertices[0]);
    Eigen::Vector3d t = v->translation();

    Eigen::Vector3d pc1 = R_wc_ * p1_ + t;
    Eigen::Vector3d pc2 = R_wc_ * p2_ + t;

    Eigen::Vector2d proj_center = 0.5 * ((K_ * pc1).hnormalized() + (K_ * pc2).hnormalized());

    _error = weight_ * (proj_center - _measurement); // _measurement 是 meas_center_
}

bool EdgeSymmetryDistanceOnly::read(std::istream&) {
    return false;
}
bool EdgeSymmetryDistanceOnly::write(std::ostream&) const {
    return false;
}