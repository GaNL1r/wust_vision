// Maintained by Chengfu Zou, Labor
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

#include "control/rune_solver.hpp"

// std
#include <memory>
// third party
#include <common/3rdparty/angles.h>

#include <Eigen/Geometry>
// project
#include "common/gobal.hpp"
#include "common/logger.hpp"
#include "common/tf.hpp"
#include "common/utils.hpp"
#include "type/type.hpp"

RuneSolver::RuneSolver(const YAML::Node& config) {
    // Init
    rune_solver_params_ = RuneSolver::RuneSolverParams {
        .compensator_type = gobal::config["rune_solver"]["compensator_type"].as<std::string>(),
        .gravity = gobal::config["rune_solver"]["gravity"].as<double>(9.8),
        .angle_offset_thres = gobal::config["rune_solver"]["angle_offset_thres"].as<double>(0.78),
        .lost_time_thres = gobal::config["rune_solver"]["lost_time_thres"].as<double>(0.5),
        .auto_type_determined = gobal::config["rune_solver"]["auto_type_determined"].as<bool>(true),
    };
    tracker_state = LOST;
    curve_fitter_ = std::make_unique<CurveFitter>(MotionType::UNKNOWN);
    curve_fitter_->setAutoTypeDetermined(rune_solver_params_.auto_type_determined);
    trajectory_compensator_ =
        CompensatorFactory::createCompensator(rune_solver_params_.compensator_type);
    trajectory_compensator_->gravity_ = rune_solver_params_.gravity;
    trajectory_compensator_->resistance_ = 0.01;
    ekf_state_ = Eigen::Vector4d::Zero();
    manual_compensator_ = std::make_unique<ManualCompensator>();
    predict_offset_ = gobal::config["rune_solver"]["predict_offset"].as<double>(0.0);
    pnp_solver_ = std::make_unique<PnPSolver>();
    pnp_solver_->setObjectPoints("rune", RUNE_OBJECT_POINTS);
    std::vector<OffsetEntry> entries;
    auto shoot_config = config["shoot"];
    if (gobal::config["rune_solver"]["trajectory_offset"]) {
        for (const auto& node: gobal::config["rune_solver"]["trajectory_offset"]) {
            OffsetEntry e;
            e.d_min = node["d_min"].as<double>();
            e.d_max = node["d_max"].as<double>();
            e.h_min = node["h_min"].as<double>();
            e.h_max = node["h_max"].as<double>();
            e.pitch_off = node["pitch_off"].as<double>();
            e.yaw_off = node["yaw_off"].as<double>();
            entries.push_back(e);
        }
    }
    manual_compensator_->updateMapFlow(entries);
    // EKF for filtering the position of R tag
    // state: yaw, pitch, distance, orientation_yaw
    // measurement: x, y, z, yaw
    // f - Process function
    auto yf = ypdrune_motion_model::Predict();
    // h - Observation function
    auto yh = ypdrune_motion_model::Measure();
    // update_Q - process noise covariance matrix

    yq_vec_ = gobal::config["rune_solver"]["ekf"]["q_ypdyaw"].as<std::vector<double>>();
    auto yu_q = [this]() {
        Eigen::Matrix<double, ypdrune_motion_model::X_N, ypdrune_motion_model::X_N> q =
            Eigen::MatrixXd::Zero(4, 4);
        q.diagonal() << yq_vec_[0], yq_vec_[1], yq_vec_[2], yq_vec_[3];
        return q;
    };
    // update_R - measurement noise covariance matrix

    yr_vec_ = gobal::config["rune_solver"]["ekf"]["r_ypdyaw"].as<std::vector<double>>();
    auto yu_r = [this](const Eigen::Matrix<double, ypdrune_motion_model::Z_N, 1>& z) {
        Eigen::Matrix<double, ypdrune_motion_model::Z_N, ypdrune_motion_model::Z_N> r =
            Eigen::MatrixXd::Zero(4, 4);
        // clang-format off
            r <<pow(yr_vec_[0] * M_PI / 180.0, 2), 0, 0, 0,
                0, pow(yr_vec_[1] * M_PI / 180.0, 2) , 0, 0,
                0, 0, yr_vec_[2] * std::abs(z[2]) *std::abs(z[2]), 0,
                0, 0, 0, yr_vec_[3];
        // clang-format on
        return r;
    };
    // P - error estimate covariance matrix
    Eigen::MatrixXd p0 = Eigen::MatrixXd::Identity(4, 4);
    Eigen::MatrixXd yp0 = Eigen::MatrixXd::Identity(4, 4);
    iteration_num_ = gobal::config["rune_solver"]["ekf"]["iteration_num"].as<int>(1);

    ekf_ypd_ = std::make_unique<ypdrune_motion_model::RuneCenterEKF>(yf, yh, yu_q, yu_r, yp0);
    ekf_ypd_->setAngleDims({ 0, 3 });
    ekf_ypd_->setIterationNum(iteration_num_);
}

double RuneSolver::init(const rune::Rune received_target, Eigen::Matrix4d T_camera_to_odom) {
    if (received_target.is_lost) {
        return 0;
    }

    WUST_INFO(rune_solver_logger_) << "Init rune solver";

    // Init EKF
    try {
        Eigen::Matrix4d T_odom_2_rune = solvePose(received_target, T_camera_to_odom);

        // Filter out outliers
        Eigen::Vector3d t = T_odom_2_rune.block(0, 3, 3, 1);
        if (t.norm() < MIN_RUNE_DISTANCE || t.norm() > MAX_RUNE_DISTANCE) {
            WUST_ERROR(rune_solver_logger_) << "Rune position is out of range";
            return 0;
        }

        ekf_state_ = getStateFromTransform(T_odom_2_rune);
        if (!utils::isStateValid(ekf_state_)) {
            WUST_ERROR(rune_solver_logger_) << "Is not valid";
            return 0;
        }

        ekf_ypd_->setState(ekf_state_);

    } catch (...) {
        WUST_ERROR(rune_solver_logger_) << "Init failed";
        return 0;
    }

    // Init observation variables
    tracker_state = DETECTING;
    double observed_angle = getNormalAngle(received_target);
    double observed_time = 0;
    curve_fitter_->update(observed_time, observed_angle);

    last_observed_angle_ = observed_angle;
    last_angle_ = last_observed_angle_;
    std::chrono::steady_clock::time_point timestamp = received_target.timestamp;
    start_time_ = std::chrono::duration<double>(timestamp.time_since_epoch()).count();

    last_time_ = start_time_;

    return observed_angle;
}

double RuneSolver::update(const rune::Rune received_target, Eigen::Matrix4d T_camera_to_odom) {
    std::chrono::steady_clock::time_point timestamp = received_target.timestamp;
    double now_time = std::chrono::duration<double>(timestamp.time_since_epoch()).count();
    double delta_time = now_time - last_time_;

    if (received_target.is_big_rune) {
        curve_fitter_->setType(MotionType::BIG);
    } else {
        curve_fitter_->setType(MotionType::SMALL);
    }

    if (!received_target.is_lost) {
        // Update EKF
        try {
            Eigen::Matrix4d T_odom_2_rune = solvePose(received_target, T_camera_to_odom);

            // Filter out outliers
            Eigen::Vector3d t = T_odom_2_rune.block(0, 3, 3, 1);
            if (t.norm() < MIN_RUNE_DISTANCE || t.norm() > MAX_RUNE_DISTANCE) {
                WUST_ERROR(rune_solver_logger_) << "Rune position is out of range";
                return 0;
            }

            Eigen::Vector4d measurement = getStateFromTransform(T_odom_2_rune);

            ekf_ypd_->predict();
            tf::Position p(measurement[0], measurement[1], measurement[2]);
            double ypd_y = std::atan2(p.y, p.x);
            ypd_y = this->last_ypd_y_ + angles::shortest_angular_distance(this->last_ypd_y_, ypd_y);
            this->last_ypd_y_ = ypd_y;
            double ypd_p = std::atan2(p.z, std::sqrt(p.x * p.x + p.y * p.y));
            double ypd_d = std::sqrt(p.x * p.x + p.y * p.y + p.z * p.z);
            Eigen::Vector4d state;
            state << ypd_y, ypd_p, ypd_d, measurement[3];
            ekf_state_ = ekf_ypd_->update(state);

        } catch (...) {
            WUST_ERROR(rune_solver_logger_) << "EKF update failed";
            return 0;
        }

        // Get the data to be fitted
        double observed_time = now_time - start_time_;
        double normal_angle = getNormalAngle(received_target);
        double observed_angle = getObservedAngle(normal_angle);

        // Update fitter
        curve_fitter_->update(observed_time, observed_angle);

        last_time_ = now_time;
        last_angle_ = normal_angle;
        last_observed_angle_ = observed_angle;
    }

    // Update tracker state
    switch (tracker_state) {
        case DETECTING: {
            if (received_target.is_lost && delta_time > rune_solver_params_.lost_time_thres) {
                tracker_state = LOST;
                curve_fitter_->reset();
            } else if (curve_fitter_->statusVerified()) {
                tracker_state = TRACKING;
            }
            break;
        }
        case TRACKING: {
            if (received_target.is_lost && delta_time > rune_solver_params_.lost_time_thres) {
                tracker_state = LOST;
                curve_fitter_->reset();
            }
            break;
        }
        case LOST: {
            if (!received_target.is_lost) {
                tracker_state = DETECTING;
            }
            break;
        }
    }
    return last_observed_angle_;
}

double RuneSolver::predictTarget(Eigen::Vector3d& predicted_position, double timestamp) {
    double t1 = timestamp - start_time_;
    double t0 = last_time_ - start_time_;
    double predict_angle_diff = curve_fitter_->predict(t1) - curve_fitter_->predict(t0);

    // Get the predicted position
    predicted_position = getTargetPosition(predict_angle_diff);

    return predict_angle_diff + last_observed_angle_;
}

Eigen::Matrix4d
RuneSolver::solvePose(const rune::Rune& predicted_target, Eigen::Matrix4d T_camera_to_odom) {
    Eigen::Matrix4d pose = Eigen::Matrix4d::Identity();
    std::vector<cv::Point2f> image_points(predicted_target.pts.size());
    std::transform(
        predicted_target.pts.begin(),
        predicted_target.pts.end(),
        image_points.begin(),
        [](const auto& pt) { return cv::Point2f(pt.x, pt.y); }
    );

    cv::Mat rvec(3, 1, CV_64F), tvec(3, 1, CV_64F);
    if (pnp_solver_
        && pnp_solver_->solvePnP(
            image_points,
            rvec,
            tvec,
            "rune",
            gobal::camera_intrinsic,
            gobal::camera_distortion
        ))
    {
        // Get the transformation matrix from rune to odom
        try {
            // Get rotation matrix from rvec
            cv::Mat rmat;
            cv::Rodrigues(rvec, rmat);
            Eigen::Matrix3d rot;
            // clang-format off
            rot << rmat.at<double>(0, 0), rmat.at<double>(0, 1), rmat.at<double>(0, 2),
                    rmat.at<double>(1, 0), rmat.at<double>(1, 1), rmat.at<double>(1, 2), 
                    rmat.at<double>(2, 0), rmat.at<double>(2, 1), rmat.at<double>(2, 2);
            // clang-format on
            Eigen::Quaterniond quat(rot);

            // Init pose msg
            tf::Transform tf;

            // Fill pose msg
            tf.orientation.x = quat.x();
            tf.orientation.y = quat.y();
            tf.orientation.z = quat.z();
            tf.orientation.w = quat.w();
            tf.position.x = tvec.at<double>(0);
            tf.position.y = tvec.at<double>(1);
            tf.position.z = tvec.at<double>(2);

            // Transform to odom

            // 构造 4x4 位姿矩阵 pose_camera
            Eigen::Matrix4d pose_camera = Eigen::Matrix4d::Identity();

            // 位置
            pose_camera(0, 3) = tf.position.x;
            pose_camera(1, 3) = tf.position.y;
            pose_camera(2, 3) = tf.position.z;

            // 旋转：四元数 -> 旋转矩阵
            Eigen::Quaterniond
                q_cam(tf.orientation.w, tf.orientation.x, tf.orientation.y, tf.orientation.z);
            pose_camera.block<3, 3>(0, 0) = q_cam.normalized().toRotationMatrix();

            // 从 camera 坐标系变换到 gimbal_odom 坐标系
            Eigen::Matrix4d pose_odom = T_camera_to_odom * pose_camera;

            // 提取变换结果
            tf::Transform pose_in_target_frame;
            pose_in_target_frame.position.x = pose_odom(0, 3);
            pose_in_target_frame.position.y = pose_odom(1, 3);
            pose_in_target_frame.position.z = pose_odom(2, 3);

            // 提取旋转部分转为四元数
            Eigen::Matrix3d R_odom = pose_odom.block<3, 3>(0, 0);
            Eigen::Quaterniond q_odom(R_odom);

            pose_in_target_frame.orientation.w = q_odom.w();
            pose_in_target_frame.orientation.x = q_odom.x();
            pose_in_target_frame.orientation.y = q_odom.y();
            pose_in_target_frame.orientation.z = q_odom.z();

            // Fill pose
            pose(0, 3) = pose_in_target_frame.position.x;
            pose(1, 3) = pose_in_target_frame.position.y;
            pose(2, 3) = pose_in_target_frame.position.z;

            Eigen::Quaterniond quat_odom;
            quat_odom.x() = pose_in_target_frame.orientation.x;
            quat_odom.y() = pose_in_target_frame.orientation.y;
            quat_odom.z() = pose_in_target_frame.orientation.z;
            quat_odom.w() = pose_in_target_frame.orientation.w;

            Eigen::Matrix3d rot_odom = quat_odom.toRotationMatrix();
            pose.block(0, 0, 3, 3) = rot_odom;

        } catch (const std::exception& e) {
            WUST_ERROR(rune_solver_logger_) << e.what();
        }
    } else {
        WUST_ERROR(rune_solver_logger_) << "PnP failed";
        throw std::runtime_error("PnP failed");
    }
    Eigen::VectorXd pose_flatten = Eigen::Map<const Eigen::VectorXd>(pose.data(), pose.size());
    if (!utils::isStateValid(pose_flatten)) {
        WUST_ERROR(rune_solver_logger_) << "Pose is not valid";
        return Eigen::Matrix4d();
    }
    return pose;
}

GimbalCmd RuneSolver::solveGimbalCmd(const Eigen::Vector3d& target) {
    // Get current yaw and pitch of gimbal
    double current_yaw = 0.0, current_pitch = 0.0;

    if (gobal::communication_delay_μs != 0) {
        std::chrono::microseconds delay =
            std::chrono::microseconds(static_cast<int64_t>(std::round(gobal::communication_delay_μs)
            ));
        auto t_query = std::chrono::steady_clock::now() - delay;
        auto past_att = gobal::motion_buffer.get_interpolated(t_query);
        if (past_att) {
            double delay_yaw = past_att->yaw;
            double delay_pitch = past_att->pitch;
            double delay_roll = past_att->roll;
            current_pitch = delay_pitch + gobal::gimbal2camera_pitch;
            current_yaw = delay_yaw + gobal::gimbal2camera_yaw;
        } else {
            current_pitch = gobal::last_pitch + gobal::gimbal2camera_pitch;
            current_yaw = gobal::last_yaw + gobal::gimbal2camera_yaw;
        }
    } else {
        current_pitch = gobal::last_pitch + gobal::gimbal2camera_pitch;
        current_yaw = gobal::last_yaw + gobal::gimbal2camera_yaw;
    }

    // Calculate yaw and pitch
    double yaw = atan2(target.y(), target.x());
    double pitch = atan2(target.z(), target.head(2).norm());

    // Set parameters of compensator

    trajectory_compensator_->gravity_ = rune_solver_params_.gravity;
    trajectory_compensator_->iteration_times_ = 30;

    if (double temp_pitch = pitch; trajectory_compensator_->compensate(target, temp_pitch)) {
        pitch = temp_pitch;
    }
    double distance = target.norm();

    // Compensate angle by angle_offset_map
    auto angle_offset = manual_compensator_->angleHardCorrect(target.head(2).norm(), target.z());
    double pitch_offset = angle_offset[0] * M_PI / 180;
    double yaw_offset = angle_offset[1] * M_PI / 180;
    double cmd_pitch = pitch + pitch_offset;
    double cmd_yaw = angles::normalize_angle(yaw + yaw_offset);

    GimbalCmd gimbal_cmd;
    gimbal_cmd.yaw = cmd_yaw * 180 / M_PI;
    gimbal_cmd.pitch = cmd_pitch * 180 / M_PI;
    gimbal_cmd.yaw_diff = (cmd_yaw - current_yaw) * 180 / M_PI;
    if (gimbal_cmd.yaw_diff > 180) {
        gimbal_cmd.yaw_diff -= 360;
    }
    if (gimbal_cmd.yaw_diff < -180) {
        gimbal_cmd.yaw_diff += 360;
    }
    gimbal_cmd.pitch_diff = (cmd_pitch - current_pitch) * 180 / M_PI;
    gimbal_cmd.distance = distance;

    // Judge whether to shoot
    constexpr double TARGET_RADIUS = 0.308;
    double shooting_range_yaw = std::abs(atan2(TARGET_RADIUS / 2, distance)) * 180 / M_PI;
    double shooting_range_pitch = std::abs(atan2(TARGET_RADIUS / 2, distance)) * 180 / M_PI;
    // Limit the shooting area to 1 degree to avoid not shooting when distance is
    // too large
    shooting_range_yaw = std::max(shooting_range_yaw, 1.0);
    shooting_range_pitch = std::max(shooting_range_pitch, 1.0);
    if (std::abs(gimbal_cmd.yaw_diff) < shooting_range_yaw
        && std::abs(gimbal_cmd.pitch_diff) < shooting_range_pitch)
    {
        gimbal_cmd.fire_advice = true;
        WUST_DEBUG(rune_solver_logger_) << "You Can Fire!";
    } else {
        gimbal_cmd.fire_advice = false;
    }

    return gimbal_cmd;
}

double RuneSolver::getNormalAngle(const rune::Rune received_target) {
    auto center_point = cv::Point2f(received_target.pts[0].x, received_target.pts[0].y);
    std::array<cv::Point2f, ARMOR_KEYPOINTS_NUM> armor_points;
    std::transform(
        received_target.pts.begin() + 1,
        received_target.pts.end(),
        armor_points.begin(),
        [](const auto& pt) { return cv::Point2f(pt.x, pt.y); }
    );

    cv::Point2f armor_center = getCenterPoint(armor_points);
    double x_diff = armor_center.x - center_point.x;
    double y_diff = -(armor_center.y - center_point.y);
    double normal_angle = std::atan2(y_diff, x_diff);
    // Normalize angle
    normal_angle = angles::normalize_angle_positive(normal_angle);

    return normal_angle;
}

double RuneSolver::getObservedAngle(double normal_angle) {
    double angle_diff = angles::shortest_angular_distance(last_angle_, normal_angle);
    // Handle rune target switch
    if (std::abs(angle_diff) > rune_solver_params_.angle_offset_thres) {
        angle_diff = normal_angle - last_angle_;
        int offset = std::round(double(angle_diff / DEG_72));
        angle_diff -= offset * DEG_72;
    }

    double observed_angle = last_observed_angle_ + angle_diff;

    return observed_angle;
}

Eigen::Vector3d RuneSolver::getCenterPosition() const {
    return ekf_state_.head(3);
}

Eigen::Vector3d RuneSolver::getTargetPosition(double angle_diff) const {
    Eigen::Vector3d t_odom_2_rune = ekf_state_.head(3);

    // Considering the large error and jitter(抖动) in the orientation obtained
    // from PnP, and the fact that the position of the Rune are static in the odom
    // frame, it is advisable to reconstruct the rotation matrix using geometric
    // information
    double yaw = ekf_state_(3);
    double pitch = 0;
    double roll = -last_angle_;
    Eigen::Matrix3d R_odom_2_rune =
        utils::eulerToMatrix(Eigen::Vector3d { roll, pitch, yaw }, utils::EulerOrder::XYZ);

    // Calculate the position of the armor in rune frame
    Eigen::Vector3d p_rune = Eigen::AngleAxisd(-angle_diff, Eigen::Vector3d::UnitX()).matrix()
        * Eigen::Vector3d(0, -ARM_LENGTH, 0);

    // Transform to odom frame
    Eigen::Vector3d p_odom = R_odom_2_rune * p_rune + t_odom_2_rune;

    return p_odom;
}

Eigen::Vector4d RuneSolver::getStateFromTransform(const Eigen::Matrix4d& transform) const {
    // Get yaw
    Eigen::Matrix3d R_odom_2_rune = transform.block(0, 0, 3, 3);
    Eigen::Quaterniond q_eigen = Eigen::Quaterniond(R_odom_2_rune);
    tf::Quaternion q_tf = tf::Quaternion(q_eigen.x(), q_eigen.y(), q_eigen.z(), q_eigen.w());
    double roll, pitch, yaw;
    tf::Matrix3x3(q_tf).getRPY(roll, pitch, yaw);
    yaw = angles::normalize_angle(yaw);

    // Make yaw continuos
    yaw = ekf_state_(3) + angles::shortest_angular_distance(ekf_state_(3), yaw);

    Eigen::Vector4d state;
    state << transform(0, 3), transform(1, 3), transform(2, 3), yaw;
    return state;
}

double RuneSolver::getCurAngle() const {
    return last_angle_;
}

cv::Point2f
RuneSolver::getCenterPoint(const std::array<cv::Point2f, ARMOR_KEYPOINTS_NUM>& armor_points) {
    return std::accumulate(armor_points.begin(), armor_points.end(), cv::Point2f(0, 0))
        / ARMOR_KEYPOINTS_NUM;
}

GimbalCmd RuneSolver::solve() {
    GimbalCmd gimbal_control_cmd;
    // Calculate predict time
    Eigen::Vector3d cur_pos = getTargetPosition(0);
    double flying_time = trajectory_compensator_->getFlyingTime(cur_pos);
    auto now = std::chrono::steady_clock::now();

    auto predict_time_point = now + std::chrono::duration<double>(flying_time + predict_offset_);
    double predict_time =
        std::chrono::duration<double>(predict_time_point.time_since_epoch()).count();

    double predict_angle = 0;

    Eigen::Vector3d pred_pos = Eigen::Vector3d::Zero();

    if (tracker_state == RuneSolver::TRACKING) {
        // Predict target
        predict_angle = predictTarget(pred_pos, predict_time);

        last_pre_angle = predict_angle;
        try {
            gimbal_control_cmd = solveGimbalCmd(pred_pos);
        } catch (...) {
            WUST_ERROR(rune_solver_logger_) << "solveGimbalCmd error";
            gimbal_control_cmd.yaw_diff = 0;
            gimbal_control_cmd.pitch_diff = 0;
            gimbal_control_cmd.distance = -1;
            gimbal_control_cmd.pitch = 0;
            gimbal_control_cmd.yaw = 0;
            gimbal_control_cmd.fire_advice = false;
        }
    } else {
        gimbal_control_cmd.yaw_diff = 0;
        gimbal_control_cmd.pitch_diff = 0;
        gimbal_control_cmd.distance = -1;
        gimbal_control_cmd.pitch = 0;
        gimbal_control_cmd.yaw = 0;
        gimbal_control_cmd.fire_advice = false;
    }
    return gimbal_control_cmd;
}
