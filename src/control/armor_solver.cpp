// Created by Chengfu Zou
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

#include "control/armor_solver.hpp"
#include "common/3rdparty/angles.h"
#include "common/gobal.hpp"
#include "common/logger.hpp"
#include "common/utils.hpp"
#include "yaml-cpp/yaml.h"
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <string>
ArmorSolver::ArmorSolver(const YAML::Node& config) {
    init(config);
}
void ArmorSolver::init(const YAML::Node& config) {
    if (!config["armor_solver"]) {
        throw std::runtime_error("Missing 'armor_solver' node in config");
    }
    auto s = config["armor_solver"];

    small_shooting_range_w_ = s["small_shooting_range_w"].as<double>(0.12);
    small_shooting_range_h_ = s["small_shooting_range_h"].as<double>(0.12);
    big_shooting_range_w_ = s["big_shooting_range_w"].as<double>(0.12);
    big_shooting_range_h_ = s["big_shooting_range_h"].as<double>(0.12);
    max_tracking_v_yaw_ = s["max_tracking_v_yaw"].as<double>(60.0);
    prediction_delay_ = s["prediction_delay"].as<double>(0.0);

    side_angle_ = s["side_angle"].as<double>(20.0);
    min_switching_v_yaw_ = s["min_switching_v_yaw"].as<double>(1.0);
    double gravity_ = s["gravity"].as<double>(10.0);
    double resistance_ = s["resistance"].as<double>(0.092);
    int iteration_times_ = s["iteration_times"].as<int>(20);
    oneswitch_position_thres_ = s["oneswitch_position_thres"].as<double>(0.2);
    oneswitch_angle_thres_ = s["oneswitch_angle_thres"].as<double>(0.2);

    oneswitch_hold_time_ = s["oneswitch_hold_time"].as<double>(0.5);

    std::string comp_type = s["compenstator_type"].as<std::string>("ideal");

    // 3. 初始化弹道补偿器
    trajectory_compensator_ = CompensatorFactory::createCompensator(comp_type);
    trajectory_compensator_->iteration_times_ = iteration_times_;
    trajectory_compensator_->gravity_ = gravity_;
    trajectory_compensator_->resistance_ = resistance_;

    // 4. 手动补偿表（pitch_offset）
    manual_compensator_ = std::make_unique<ManualCompensator>();
    std::vector<OffsetEntry> entries;

    if (s["trajectory_offset"]) {
        for (const auto& node: s["trajectory_offset"]) {
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

    auto gobal_state = gobal::stringanyting.get_value<GobalState>("gobal_state");
    gobal_state.armor_slove_state = GobalState::ArmorSloveState::TRACKING_ARMOR;
    gobal::stringanyting.set_value("gobal_state", gobal_state);
    overflow_count_ = 0;
    transfer_thresh_ = 5;
}

GimbalCmd ArmorSolver::solve(
    const ArmorSolverTarget& armor_solver_target,
    std::chrono::steady_clock::time_point current_time
) {
    // 1. 获取最新的云台 RPY
    std::array<double, 3> rpy;
    auto motion_buffer = gobal::stringanyting.get_ptr<MotionBuffer>("motion_buffer");
    auto gimbal2camera_rpy =
        gobal::stringanyting.get_value<std::array<double, 3>>("gimbal2camera_rpy");
    if (motion_buffer) {
        auto last_att = motion_buffer->get_last();
        if (last_att) {
            rpy[0] = last_att->roll + gimbal2camera_rpy[0];
            rpy[1] = last_att->pitch + gimbal2camera_rpy[1];
            rpy[2] = last_att->yaw + gimbal2camera_rpy[2];
        }
    }
    double controller_delay = gobal::stringanyting.get_value<double>("controller_delay");
    // 2. 选择最优单目标索引
    const auto& one_targets = armor_solver_target.one_targets;
    const auto& target = armor_solver_target.target;
    int one_idx = selectBestTarget(one_targets, target.tracking);
    armor::OneTarget best_target;
    if (one_idx >= 0) {
        best_target = one_targets[one_idx];
    }

    // 拿到 best_target
    bool use_multi = (!best_target.tracking);

    if (use_multi) {
        // 3.1 预测位置、速度、yaw
        Eigen::Vector3d pos = target.position_.toEigen();
        Eigen::Vector3d vel = target.velocity_.toEigen();
        Eigen::Vector3d acc = target.acceleration_.toEigen();
        double yaw = target.yaw;

        using namespace std::chrono;
        double fly_t = trajectory_compensator_->getFlyingTime(pos);
        double dt_sec =
            fly_t + prediction_delay_ + duration<double>(current_time - target.timestamp).count();

        vel += dt_sec * acc;
        pos += dt_sec * vel;
        yaw += dt_sec * target.v_yaw;

        // 3. 获取装甲板位置和速度
        auto armors = getArmorPositions(
            pos,
            yaw,
            target.radius_1,
            target.radius_2,
            target.d_zc,
            target.d_za,
            target.armors_num
        );
        auto v_armors = getArmorVelocities(
            pos,
            yaw,
            vel,
            target.v_yaw,
            target.radius_1,
            target.radius_2,
            target.d_zc,
            target.d_za,
            target.armors_num
        );

        int idx = selectBestArmor(armors, pos, yaw, target.v_yaw, target.armors_num);
        if (idx < 0 || idx >= (int)armors.size() || armors[idx].norm() < 0.1) {
            return returnDefaultCmd();
        }

        // 4. 状态机逻辑
        bool fire_advice = false;
        double raw_yaw = 0.0, raw_pitch = 0.0;
        double fake_yaw = 0.0, fake_pitch = 0.0;
        double distance = 0.0;
        double fire_yaw = 0.0, fire_pitch = 0.0;
        double control_yaw = 0.0, control_pitch = 0.0;
        Eigen::Vector3d chosen;
        bool is_large =
            (target.id == armor::ArmorNumber::NO1 || target.id == armor::ArmorNumber::BASE);
        auto gobal_state = gobal::stringanyting.get_value<GobalState>("gobal_state");

        switch (gobal_state.armor_slove_state) {
            case GobalState::ArmorSloveState::TRACKING_ARMOR:
                if (std::abs(target.v_yaw) > max_tracking_v_yaw_) {
                    ++overflow_count_;
                } else {
                    overflow_count_ = 0;
                }
                if (overflow_count_ > transfer_thresh_) {
                    gobal_state.armor_slove_state = GobalState::ArmorSloveState::TRACKING_CENTER;
                    gobal::stringanyting.set_value("gobal_state", gobal_state);
                    overflow_count_ = 0;
                }
                chosen = armors[idx];

                if (controller_delay > 1e-6) {
                    pos += controller_delay * vel;
                    yaw += controller_delay * target.v_yaw;
                    auto delayed_armors = getArmorPositions(
                        pos,
                        yaw,
                        target.radius_1,
                        target.radius_2,
                        target.d_zc,
                        target.d_za,
                        target.armors_num
                    );
                    if (idx >= 0 && idx < (int)delayed_armors.size()) {
                        chosen = delayed_armors[idx];
                        if (chosen.norm() < 0.1)
                            return returnDefaultCmd();
                    }
                }
                calcYawAndPitch(chosen, rpy, raw_yaw, raw_pitch);
                distance = chosen.norm();

                std::tie(fire_yaw, fire_pitch) =
                    manual_compensator_
                        ->applyManualCompensator(distance, chosen.z(), raw_yaw, raw_pitch);
                fire_advice = isOnTarget(rpy[2], rpy[1], fire_yaw, fire_pitch, distance, is_large);
                control_yaw = fire_yaw;
                control_pitch = fire_pitch;
                break;

            case GobalState::ArmorSloveState::TRACKING_CENTER:
                if (std::abs(target.v_yaw) < max_tracking_v_yaw_) {
                    ++overflow_count_;
                } else {
                    overflow_count_ = 0;
                }
                if (overflow_count_ > transfer_thresh_) {
                    gobal_state.armor_slove_state = GobalState::ArmorSloveState::TRACKING_ARMOR;
                    gobal::stringanyting.set_value("gobal_state", gobal_state);
                    overflow_count_ = 0;
                }
                chosen = armors[idx];
                if (controller_delay > 1e-6) {
                    pos += controller_delay * vel;
                    yaw += controller_delay * target.v_yaw;
                    auto delayed_armors = getArmorPositions(
                        pos,
                        yaw,
                        target.radius_1,
                        target.radius_2,
                        target.d_zc,
                        target.d_za,
                        target.armors_num
                    );
                    if (idx >= 0 && idx < (int)delayed_armors.size()) {
                        chosen = delayed_armors[idx];
                        if (chosen.norm() < 0.1)
                            return returnDefaultCmd();
                    }
                }
                calcYawAndPitch(chosen, rpy, fake_yaw, raw_pitch);
                calcYawAndPitch(pos, rpy, raw_yaw, fake_pitch);
                distance = chosen.norm();

                std::tie(fire_yaw, fire_pitch) =
                    manual_compensator_
                        ->applyManualCompensator(distance, chosen.z(), fake_yaw, raw_pitch);
                fire_advice = isOnTarget(rpy[2], rpy[1], fire_yaw, fire_pitch, distance, is_large);
                std::tie(control_yaw, fake_pitch) =
                    manual_compensator_
                        ->applyManualCompensator(distance, chosen.z(), raw_yaw, raw_pitch);
                control_pitch = fire_pitch;
                break;

            default:
                gobal_state.armor_slove_state = GobalState::ArmorSloveState::TRACKING_ARMOR;
                gobal::stringanyting.set_value("gobal_state", gobal_state);
                break;
        }

        Eigen::Vector3d chosen_vel = v_armors[idx];
        double v_yaw, v_pitch;
        double v_yaw_fake, v_pitch_fake;
        calcVYawAndVPitch(pos, vel, rpy, v_yaw, v_pitch_fake);
        calcVYawAndVPitch(chosen, chosen_vel, rpy, v_yaw_fake, v_pitch);

        GimbalCmd cmd;
        cmd.timestamp = current_time;
        cmd.distance = distance;
        cmd.fire_advice = fire_advice;
        cmd.yaw = control_yaw * 180.0 / M_PI;
        cmd.pitch = control_pitch * 180.0 / M_PI;

        cmd.yaw_diff = normalize_angle(control_yaw - rpy[2]) * 180.0 / M_PI;
        if (cmd.yaw_diff > 180)
            cmd.yaw_diff -= 360;
        if (cmd.yaw_diff < -180)
            cmd.yaw_diff += 360;

        cmd.pitch_diff = (control_pitch - rpy[1]) * 180.0 / M_PI;
        cmd.v_yaw = v_yaw * 180.0 / M_PI;
        cmd.v_pitch = v_pitch * 180.0 / M_PI;
        cmd.select_id = idx;
        return cmd;
    } else {
        // 4.1 预测 best_target
        Eigen::Vector3d pos = best_target.position_.toEigen();
        Eigen::Vector3d vel = best_target.velocity_.toEigen();
        Eigen::Vector3d acc = best_target.acceleration_.toEigen();
        double yaw = best_target.yaw;

        using namespace std::chrono;
        double fly_t = trajectory_compensator_->getFlyingTime(pos);
        double dt_sec = fly_t + prediction_delay_
            + duration<double>(current_time - best_target.timestamp).count() + controller_delay;

        vel += dt_sec * acc;
        pos += dt_sec * vel;
        yaw += dt_sec * best_target.v_yaw;
        // 4.2 计算角度与补偿
        double raw_yaw, raw_pitch;
        calcYawAndPitch(pos, rpy, raw_yaw, raw_pitch);

        double distance = pos.norm();
        auto [fire_yaw, fire_pitch] =
            manual_compensator_->applyManualCompensator(distance, pos.z(), raw_yaw, raw_pitch);

        // 4.3 发射判断
        bool is_large =
            (best_target.id == armor::ArmorNumber::NO1 || best_target.id == armor::ArmorNumber::BASE
            );
        bool fire_advice = isOnTarget(rpy[2], rpy[1], fire_yaw, fire_pitch, distance, is_large);

        // 4.4 速度角计算
        double v_yaw, v_pitch;
        calcVYawAndVPitch(pos, vel, rpy, v_yaw, v_pitch);
        double control_yaw = fire_yaw;
        double control_pitch = fire_pitch;
        // 4.5 填充并返回
        GimbalCmd cmd;
        cmd.timestamp = current_time;
        cmd.distance = distance;
        cmd.fire_advice = fire_advice;
        cmd.yaw = control_yaw * 180.0 / M_PI;
        cmd.pitch = control_pitch * 180.0 / M_PI;

        cmd.yaw_diff = normalize_angle(control_yaw - rpy[2]) * 180.0 / M_PI;
        if (cmd.yaw_diff > 180)
            cmd.yaw_diff -= 360;
        if (cmd.yaw_diff < -180)
            cmd.yaw_diff += 360;

        cmd.pitch_diff = (control_pitch - rpy[1]) * 180.0 / M_PI;
        cmd.v_yaw = v_yaw * 180.0 / M_PI;
        cmd.v_pitch = v_pitch * 180.0 / M_PI;
        cmd.select_id = one_idx + target.armors_num;

        return cmd;
    }
}

std::vector<std::pair<double, double>> ArmorSolver::getTrajectory() const noexcept {
    auto traj = trajectory_compensator_->getTrajectory(15, rpy_[1]);
    for (auto& p: traj) {
        double x = p.first, y = p.second;
        p.first = x * std::cos(rpy_[1]) + y * std::sin(rpy_[1]);
        p.second = -x * std::sin(rpy_[1]) + y * std::cos(rpy_[1]);
    }
    return traj;
}

bool ArmorSolver::isOnTarget(
    const double cur_yaw,
    const double cur_pitch,
    const double target_yaw,
    const double target_pitch,
    const double distance,
    const bool is_large_armor
) const noexcept {
    // 计算你与目标朝向之间的最短夹角
    double yaw_diff = angles::shortest_angular_distance(std::abs(target_yaw), std::abs(cur_yaw));
    double width_scale = std::abs(std::cos(yaw_diff)); // 角度越一致，目标越“宽”
    double shooting_range_yaw, shooting_range_pitch;
    if (is_large_armor) {
        // 缩放后的目标宽度
        double dynamic_w = big_shooting_range_w_ * width_scale;

        // 开火角容差
        shooting_range_yaw = std::abs(std::atan2(big_shooting_range_w_ / 2.0, distance));
        shooting_range_pitch = std::abs(std::atan2(big_shooting_range_h_ / 2.0, distance));
    } else {
        // 缩放后的目标宽度
        double dynamic_w = small_shooting_range_w_ * width_scale;

        // 开火角容差
        shooting_range_yaw = std::abs(std::atan2(small_shooting_range_w_ / 2.0, distance));
        shooting_range_pitch = std::abs(std::atan2(small_shooting_range_h_ / 2.0, distance));
    }

    // 限制最小角度为 1°
    constexpr double min_angle_rad = 1.0 * M_PI / 180.0;
    shooting_range_yaw = std::max(shooting_range_yaw, min_angle_rad);
    shooting_range_pitch = std::max(shooting_range_pitch, min_angle_rad);

    // 判断是否命中窗口
    if (std::abs(yaw_diff) < shooting_range_yaw
        && std::abs(cur_pitch - target_pitch) < shooting_range_pitch)
    {
        return true;
    }

    return false;
}

std::vector<Eigen::Vector3d> ArmorSolver::getArmorPositions(
    const Eigen::Vector3d& target_center,
    const double target_yaw,
    const double r1,
    const double r2,
    const double d_zc,
    const double d_za,
    const size_t armors_num
) const noexcept {
    auto armor_positions = std::vector<Eigen::Vector3d>(armors_num, Eigen::Vector3d::Zero());
    // Calculate the position of each armor
    bool is_current_pair = true;
    double r = 0., target_dz = 0.;
    for (size_t i = 0; i < armors_num; i++) {
        double temp_yaw = target_yaw + i * (2 * M_PI / armors_num);
        if (armors_num == 4) {
            r = is_current_pair ? r1 : r2;
            target_dz = d_zc + (is_current_pair ? 0 : d_za);
            is_current_pair = !is_current_pair;
        } else {
            r = r1;
            target_dz = d_zc;
        }
        armor_positions[i] =
            target_center + Eigen::Vector3d(-r * cos(temp_yaw), -r * sin(temp_yaw), target_dz);
    }
    return armor_positions;
}
void ArmorSolver::calcYawAndPitch(
    const Eigen::Vector3d& p,
    const std::array<double, 3>& rpy,
    double& yaw,
    double& pitch
) const noexcept {
    // Calculate yaw and pitch
    yaw = atan2(p.y(), p.x());
    pitch = atan2(p.z(), p.head(2).norm());

    if (double temp_pitch = pitch; trajectory_compensator_->compensate(p, temp_pitch)) {
        pitch = temp_pitch;
    }
}
void ArmorSolver::calcVYawAndVPitch(
    const Eigen::Vector3d& p,
    const Eigen::Vector3d& v,
    const std::array<double, 3>& rpy,
    double& vyaw,
    double& vpitch
) const noexcept {
    double px = p.x();
    double py = p.y();
    double pz = p.z();

    double vx = v.x();
    double vy = v.y();
    double vz = v.z();

    double px2_py2 = px * px + py * py;
    double p_norm_sq = px * px + py * py + pz * pz;
    double p_horiz = std::sqrt(px2_py2);
    if (px2_py2 < 1e-12 || p_norm_sq < 1e-12) {
        vyaw = 0.0;
        vpitch = 0.0;
        return;
    }
    vyaw = (px * vy - py * vx) / px2_py2;
    vpitch = (pz * (px * vx + py * vy) / p_horiz - p_horiz * vz) / p_norm_sq;
}

int ArmorSolver::selectBestArmor(
    const std::vector<Eigen::Vector3d>& armor_positions,
    const Eigen::Vector3d& target_center,
    const double target_yaw,
    const double target_v_yaw,
    const size_t armors_num
) const noexcept {
    if (armor_positions.empty())
        return -1;
    // Angle between the car's center and the X-axis
    double alpha = std::atan2(target_center.y(), target_center.x());
    // Angle between the front of observed armor and the X-axis
    double beta = target_yaw;

    // clang-format off
    Eigen::Matrix2d R_odom2center;
    Eigen::Matrix2d R_odom2armor;
    R_odom2center << std::cos(alpha), std::sin(alpha), 
    -std::sin(alpha), std::cos(alpha);
    R_odom2armor << std::cos(beta), std::sin(beta), 
    -std::sin(beta), std::cos(beta);
    // clang-format on
    Eigen::Matrix2d R_center2armor = R_odom2center.transpose() * R_odom2armor;

    // Equal to (alpha - beta) in most cases
    double decision_angle = -std::asin(R_center2armor(0, 1));

    // Angle thresh of the armor jump
    double theta = (target_v_yaw > 0 ? side_angle_ : -side_angle_) / 180.0 * M_PI;

    // Avoid the frequent switch between two armor
    if (std::abs(target_v_yaw) < min_switching_v_yaw_) {
        theta = 0;
    }

    double temp_angle = decision_angle + M_PI / armors_num - theta;

    if (temp_angle < 0) {
        temp_angle += 2 * M_PI;
    }

    int selected_id = static_cast<int>(temp_angle / (2 * M_PI / armors_num));
    return selected_id;
}
int ArmorSolver::selectBestTarget(
    const std::vector<armor::OneTarget>& targets,
    bool is_target_tracking
) const noexcept {
    int best_idx = -1;
    auto current_time = std::chrono::steady_clock::now();
    double min_angle_diff = std::numeric_limits<double>::max();
    const armor::OneTarget* best_target_ptr = nullptr;

    // 遍历所有正在跟踪的目标
    for (int i = 0; i < targets.size(); ++i) {
        const auto& tgt = targets[i];
        if (!tgt.tracking)
            continue;

        // 计算目标方向角度差
        double alpha = std::atan2(tgt.position_.y, tgt.position_.x);
        double beta = tgt.yaw;

        Eigen::Matrix2d Ra, Rb;
        Ra << std::cos(alpha), std::sin(alpha), -std::sin(alpha), std::cos(alpha);
        Rb << std::cos(beta), std::sin(beta), -std::sin(beta), std::cos(beta);

        double decision_angle = -std::asin((Ra.transpose() * Rb)(0, 1));

        if (std::abs(decision_angle) < min_angle_diff) {
            min_angle_diff = std::abs(decision_angle);
            best_idx = i;
            best_target_ptr = &tgt;
        }
    }

    if (best_idx == -1 || best_target_ptr == nullptr)
        return -1;

    const armor::OneTarget& best_target = *best_target_ptr;

    if (!has_last_target_) {
        last_target_ = best_target;
        has_last_target_ = true;
        return best_idx;
    }

    double dist = (best_target.position_ - last_target_.position_).norm();
    bool same_id = best_target.id == last_target_.id;
    bool same_target = same_id && dist < oneswitch_position_thres_;

    double alpha_last = std::atan2(last_target_.position_.y, last_target_.position_.x);
    double beta_last = last_target_.yaw;
    Eigen::Matrix2d Ral, Rbl;
    Ral << std::cos(alpha_last), std::sin(alpha_last), -std::sin(alpha_last), std::cos(alpha_last);
    Rbl << std::cos(beta_last), std::sin(beta_last), -std::sin(beta_last), std::cos(beta_last);
    double last_angle_diff = std::abs(std::asin((Ral.transpose() * Rbl)(0, 1)));

    if (!same_target
        && std::abs(last_angle_diff - min_angle_diff) > oneswitch_angle_thres_ * M_PI / 180.0)
    {
        if (!has_pending_target_ || (best_target.id != pending_target_.id)) {
            pending_target_ = best_target;
            pending_target_start_time_ = current_time;
            has_pending_target_ = true;
        } else {
            double hold_time =
                std::chrono::duration<double>(current_time - pending_target_start_time_).count();
            if (hold_time > oneswitch_hold_time_) {
                last_target_ = pending_target_;
                return best_idx;
            }
        }

        return -1;
    }

    last_target_ = best_target;
    has_pending_target_ = false;
    return best_idx;
}

std::vector<Eigen::Vector3d> ArmorSolver::getArmorVelocities(
    const Eigen::Vector3d& target_center,
    const double target_yaw,
    const Eigen::Vector3d& target_velocity,
    const double target_yaw_rate,
    const double r1,
    const double r2,
    const double d_zc,
    const double d_za,
    const size_t armors_num
) const noexcept {
    std::vector<Eigen::Vector3d> velocities;
    velocities.reserve(armors_num);

    bool is_current_pair = true;
    for (size_t i = 0; i < armors_num; ++i) {
        double temp_yaw = target_yaw + i * (2 * M_PI / armors_num);

        // 选择当前装甲的半径和高度
        double r = (armors_num == 4 && !is_current_pair) ? r2 : r1;
        double target_dz = d_zc + ((armors_num == 4 && !is_current_pair) ? d_za : 0.0);
        is_current_pair = !is_current_pair;

        // 装甲板相对于中心的偏移向量
        Eigen::Vector3d offset(-r * cos(temp_yaw), -r * sin(temp_yaw), target_dz);
        Eigen::Vector3d armor_pos = target_center + offset;
        Eigen::Vector3d r_vec = armor_pos - target_center;

        // 角速度矢量（仅绕 z）
        Eigen::Vector3d omega(0, 0, target_yaw_rate);

        // 切向速度
        Eigen::Vector3d v_rot = omega.cross(r_vec);

        // 总速度 = 平动 + 切向
        Eigen::Vector3d v_total = target_velocity + v_rot;

        velocities.push_back(v_total);
    }

    return velocities;
}
