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

    small_shooting_range_w = s["small_shooting_range_w"].as<double>(0.12);
    small_shooting_range_h = s["small_shooting_range_h"].as<double>(0.12);
    big_shooting_range_w = s["big_shooting_range_w"].as<double>(0.12);
    big_shooting_range_h = s["big_shooting_range_h"].as<double>(0.12);
    max_tracking_v_yaw = s["max_tracking_v_yaw"].as<double>(60.0);
    prediction_delay = s["prediction_delay"].as<double>(0.0);
    gobal::controller_delay = s["controller_delay"].as<double>(0.0);
    side_angle = s["side_angle"].as<double>(20.0);
    min_switching_v_yaw = s["min_switching_v_yaw"].as<double>(1.0);

    bullet_speed = s["bullet_speed"].as<double>(25.0);
    gravity = s["gravity"].as<double>(10.0);
    resistance = s["resistance"].as<double>(0.092);
    iteration_times = s["iteration_times"].as<int>(20);
    oneswitch_position_thres_ = s["oneswitch_position_thres"].as<double>(0.2);
    oneswitch_angle_thres_ = s["oneswitch_angle_thres"].as<double>(0.2);

    oneswitch_hold_time_ = s["oneswitch_hold_time"].as<double>(0.5);

    std::string comp_type = s["compenstator_type"].as<std::string>("ideal");

    // 3. 初始化弹道补偿器
    trajectory_compensator_ = CompensatorFactory::createCompensator(comp_type);
    trajectory_compensator_->iteration_times = iteration_times;
    gobal::velocity = bullet_speed;
    trajectory_compensator_->gravity = gravity;
    trajectory_compensator_->resistance = resistance;

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

    // 5. 状态机初值
    state_ = State::TRACKING_ARMOR;
    overflow_count_ = 0;
    transfer_thresh = 5;
}

GimbalCmd
ArmorSolver::solve(const Target& target, std::chrono::steady_clock::time_point current_time) {
    // 1. 获取最新的云台 RPY
    std::array<double, 3> rpy {};

    rpy[0] = gobal::last_roll + gobal::gimbal2camera_roll;
    rpy[1] = gobal::last_pitch + gobal::gimbal2camera_pitch;
    rpy[2] = gobal::last_yaw + gobal::gimbal2camera_yaw;

    //  2. 预测目标位置与朝向
    Eigen::Vector3d pos(target.position_.x, target.position_.y, target.position_.z);
    double yaw = target.yaw;

    using namespace std::chrono;

    double fly_t = trajectory_compensator_->getFlyingTime(pos);
    auto dt_seconds = duration<double>(fly_t + prediction_delay);
    auto dt = duration_cast<steady_clock::duration>(dt_seconds);
    auto total_dt = (current_time - target.timestamp) + dt;
    double dt_seconds_double = duration<double>(total_dt).count();
    auto center_v = Eigen::Vector3d(target.velocity_.x, target.velocity_.y, target.velocity_.z);
    center_v += dt_seconds_double
        * Eigen::Vector3d(target.acceleration_.x, target.acceleration_.y, target.acceleration_.z);
    pos += dt_seconds_double * center_v;
    yaw += dt_seconds_double * target.v_yaw;

    // 3. 选装甲板并计算原始 yaw/pitch
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
        center_v,
        target.v_yaw,
        target.radius_1,
        target.radius_2,
        target.d_zc,
        target.d_za,
        target.armors_num
    );
    int idx = selectBestArmor(armors, pos, yaw, target.v_yaw, target.armors_num);

    Eigen::Vector3d chosen = armors.at(idx);
    Eigen::Vector3d v_chosen = v_armors.at(idx);
    if (chosen.norm() < 0.1) {
        throw std::runtime_error("No valid armor to shoot");
    }
    double raw_yaw, raw_pitch;
    double v_yaw, v_pitch;
    double v_yaw_ = 0;
    double v_pitch_ = 0;
    calcYawAndPitch(chosen, rpy, raw_yaw, raw_pitch);
    calcVYawAndVPitch(v_chosen, rpy, v_yaw, v_pitch);
    double distance = chosen.norm();
    std::vector<double> offs;
    double pitch_off;
    double yaw_off;
    double fire_yaw;
    double fire_pitch;
    double raw_yaw_, raw_pitch_;
    // 4. 状态机逻辑
    bool fire_advice = false;
    bool is_large;
    switch (state_) {
        case TRACKING_ARMOR:
            if (std::abs(target.v_yaw) > max_tracking_v_yaw) {
                ++overflow_count_;
            } else {
                overflow_count_ = 0;
            }
            if (overflow_count_ > transfer_thresh) {
                state_ = TRACKING_CENTER;
            }
            // 如果一直没对上，也加 controller_delay 预测
            if (gobal::controller_delay != 0.0) {
                center_v += gobal::controller_delay
                    * Eigen::Vector3d(
                                target.acceleration_.x,
                                target.acceleration_.y,
                                target.acceleration_.z
                    );
                pos += gobal::controller_delay * center_v;
                yaw += gobal::controller_delay * target.v_yaw;
                auto tmp = getArmorPositions(
                               pos,
                               yaw,
                               target.radius_1,
                               target.radius_2,
                               target.d_zc,
                               target.d_za,
                               target.armors_num
                )
                               .at(idx);
                if (tmp.norm() < 0.1) {
                    throw std::runtime_error("No valid armor after controller delay");
                }
                calcYawAndPitch(tmp, rpy, raw_yaw, raw_pitch);
                distance = tmp.norm();
            }
            calcYawAndPitch(pos, rpy, raw_yaw_, raw_pitch);
            offs = manual_compensator_->angleHardCorrect(distance, chosen.z());
            yaw_off = offs[1] * M_PI / 180.0;
            pitch_off = offs[0] * M_PI / 180.0;
            fire_yaw = raw_yaw + yaw_off;
            fire_pitch = raw_pitch + pitch_off;
            is_large = (target.id == ArmorNumber::NO1 || target.id == ArmorNumber::BASE);
            fire_advice = isOnTarget(rpy[2], rpy[1], fire_yaw, fire_pitch, distance, is_large);
            break;

        case TRACKING_CENTER:
            if (std::abs(target.v_yaw) < max_tracking_v_yaw) {
                ++overflow_count_;
            } else {
                overflow_count_ = 0;
            }
            if (overflow_count_ > transfer_thresh) {
                state_ = TRACKING_ARMOR;
                overflow_count_ = 0;
            }

            calcYawAndPitch(chosen, rpy, raw_yaw_, raw_pitch);
            if (gobal::controller_delay != 0.0) {
                center_v += gobal::controller_delay
                    * Eigen::Vector3d(
                                target.acceleration_.x,
                                target.acceleration_.y,
                                target.acceleration_.z
                    );
                pos += gobal::controller_delay * center_v;
                yaw += gobal::controller_delay * target.v_yaw;
                auto tmp = getArmorPositions(
                               pos,
                               yaw,
                               target.radius_1,
                               target.radius_2,
                               target.d_zc,
                               target.d_za,
                               target.armors_num
                )
                               .at(idx);
                if (tmp.norm() < 0.1) {
                    throw std::runtime_error("No valid armor after controller delay");
                }
                calcYawAndPitch(tmp, rpy, raw_yaw_, raw_pitch);
                distance = tmp.norm();
            }
            // fire_advice = true;
            calcYawAndPitch(pos, rpy, raw_yaw, raw_pitch_);
            calcVYawAndVPitch(center_v, rpy, v_yaw, v_pitch_);
            distance = pos.norm();
            offs = manual_compensator_->angleHardCorrect(distance, chosen.z());
            yaw_off = offs[1] * M_PI / 180.0;
            pitch_off = offs[0] * M_PI / 180.0;

            fire_yaw = raw_yaw_ + yaw_off;
            fire_pitch = raw_pitch + pitch_off;
            is_large = (target.id == ArmorNumber::NO1 || target.id == ArmorNumber::BASE);
            fire_advice = isOnTarget(rpy[2], rpy[1], fire_yaw, fire_pitch, distance, is_large);
            break;
    }

    // 5. 弹道+手动补偿

    double cmd_pitch = raw_pitch + pitch_off;
    double cmd_yaw = normalize_angle(raw_yaw + yaw_off);

    // 6. 填充输出
    GimbalCmd cmd;
    cmd.timestamp = current_time;
    cmd.distance = distance;
    cmd.fire_advice = fire_advice;
    cmd.yaw = cmd_yaw * 180.0 / M_PI;
    cmd.pitch = cmd_pitch * 180.0 / M_PI;
    cmd.yaw_diff = (cmd_yaw - rpy[2]) * 180.0 / M_PI;
    if (cmd.yaw_diff > 180) {
        cmd.yaw_diff -= 360;
    }
    if (cmd.yaw_diff < -180) {
        cmd.yaw_diff += 360;
    }
    cmd.pitch_diff = (cmd_pitch - rpy[1]) * 180.0 / M_PI;
    cmd.v_yaw = v_yaw * 180.0 / M_PI;
    cmd.v_pitch = v_pitch * 180.0 / M_PI;
    cmd.select_id = idx;
    return cmd;
}
GimbalCmd ArmorSolver::solve(
    const Target& target,
    std::vector<OneTarget> one_targets_,
    std::chrono::steady_clock::time_point current_time
) {
    // 1. 获取最新的云台 RPY
    std::array<double, 3> rpy {};

    rpy[0] = gobal::last_roll + gobal::gimbal2camera_roll;
    rpy[1] = gobal::last_pitch + gobal::gimbal2camera_pitch;
    rpy[2] = gobal::last_yaw + gobal::gimbal2camera_yaw;

    int one_idx = selectBestTarget(one_targets_, target.tracking);
    int target_armor_num = target.armors_num;
    // 2. 选择最佳单目标
    OneTarget best_target;
    if (one_idx >= 0) {
        best_target = one_targets_[one_idx];
    }
    bool use_single = best_target.tracking;
    //  2. 预测目标位置与朝向

    if (!use_single || target.tracking) {
        Eigen::Vector3d pos(target.position_.x, target.position_.y, target.position_.z);
        double yaw = target.yaw;

        using namespace std::chrono;

        double fly_t = trajectory_compensator_->getFlyingTime(pos);
        auto dt_seconds = duration<double>(fly_t + prediction_delay);
        auto dt = duration_cast<steady_clock::duration>(dt_seconds);
        auto total_dt = (current_time - target.timestamp) + dt;
        double dt_seconds_double = duration<double>(total_dt).count();
        auto center_v = Eigen::Vector3d(target.velocity_.x, target.velocity_.y, target.velocity_.z);
        center_v += dt_seconds_double
            * Eigen::Vector3d(
                        target.acceleration_.x,
                        target.acceleration_.y,
                        target.acceleration_.z
            );
        pos += dt_seconds_double * center_v;
        yaw += dt_seconds_double * target.v_yaw;
        // 3. 选装甲板并计算原始 yaw/pitch
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
            center_v,
            target.v_yaw,
            target.radius_1,
            target.radius_2,
            target.d_zc,
            target.d_za,
            target.armors_num
        );
        int idx = selectBestArmor(armors, pos, yaw, target.v_yaw, target.armors_num);

        if (idx < 0 || idx >= static_cast<int>(armors.size())) {
            return returndefaultCmd();
        }

        Eigen::Vector3d chosen = armors[idx];

        Eigen::Vector3d v_chosen = v_armors[idx];
        if (chosen.norm() < 0.1) {
            throw std::runtime_error("No valid armor to shoot");
        }
        double raw_yaw, raw_pitch;
        double v_yaw, v_pitch;
        double v_yaw_ = 0;
        double v_pitch_ = 0;
        calcYawAndPitch(chosen, rpy, raw_yaw, raw_pitch);
        calcVYawAndVPitch(v_chosen, rpy, v_yaw, v_pitch);
        double distance = chosen.norm();
        std::vector<double> offs;
        double pitch_off;
        double yaw_off;
        double fire_yaw;
        double fire_pitch;
        double raw_yaw_, raw_pitch_;
        bool fire_advice = false;
        bool is_large;
        switch (state_) {
            case TRACKING_ARMOR:
                if (std::abs(target.v_yaw) > max_tracking_v_yaw) {
                    ++overflow_count_;
                } else {
                    overflow_count_ = 0;
                }
                if (overflow_count_ > transfer_thresh) {
                    state_ = TRACKING_CENTER;
                }
                // 如果一直没对上，也加 controller_delay 预测
                if (gobal::controller_delay != 0.0) {
                    center_v += gobal::controller_delay
                        * Eigen::Vector3d(
                                    target.acceleration_.x,
                                    target.acceleration_.y,
                                    target.acceleration_.z
                        );
                    pos += gobal::controller_delay * center_v;
                    yaw += gobal::controller_delay * target.v_yaw;
                    auto tmp = getArmorPositions(
                                   pos,
                                   yaw,
                                   target.radius_1,
                                   target.radius_2,
                                   target.d_zc,
                                   target.d_za,
                                   target.armors_num
                    )
                                   .at(idx);
                    if (tmp.norm() < 0.1) {
                        throw std::runtime_error("No valid armor after controller delay");
                    }
                    calcYawAndPitch(tmp, rpy, raw_yaw, raw_pitch);
                    distance = tmp.norm();
                }
                //calcYawAndPitch(pos, rpy, raw_yaw_, raw_pitch);
                offs = manual_compensator_->angleHardCorrect(distance, chosen.z());
                yaw_off = offs[1] * M_PI / 180.0;
                pitch_off = offs[0] * M_PI / 180.0;
                fire_yaw = raw_yaw + yaw_off;
                fire_pitch = raw_pitch + pitch_off;
                is_large = (target.id == ArmorNumber::NO1 || target.id == ArmorNumber::BASE);
                fire_advice = isOnTarget(rpy[2], rpy[1], fire_yaw, fire_pitch, distance, is_large);
                break;

            case TRACKING_CENTER:
                if (std::abs(target.v_yaw) < max_tracking_v_yaw) {
                    ++overflow_count_;
                } else {
                    overflow_count_ = 0;
                }
                if (overflow_count_ > transfer_thresh) {
                    state_ = TRACKING_ARMOR;
                    overflow_count_ = 0;
                }

                calcYawAndPitch(chosen, rpy, raw_yaw_, raw_pitch);
                if (gobal::controller_delay != 0.0) {
                    center_v += gobal::controller_delay
                        * Eigen::Vector3d(
                                    target.acceleration_.x,
                                    target.acceleration_.y,
                                    target.acceleration_.z
                        );
                    pos += gobal::controller_delay * center_v;
                    yaw += gobal::controller_delay * target.v_yaw;
                    auto tmp = getArmorPositions(
                                   pos,
                                   yaw,
                                   target.radius_1,
                                   target.radius_2,
                                   target.d_zc,
                                   target.d_za,
                                   target.armors_num
                    )
                                   .at(idx);
                    if (tmp.norm() < 0.1) {
                        throw std::runtime_error("No valid armor after controller delay");
                    }
                    calcYawAndPitch(tmp, rpy, raw_yaw_, raw_pitch);
                    distance = tmp.norm();
                }
                // fire_advice = true;
                calcYawAndPitch(pos, rpy, raw_yaw, raw_pitch_);
                calcVYawAndVPitch(center_v, rpy, v_yaw, v_pitch_);
                distance = pos.norm();
                offs = manual_compensator_->angleHardCorrect(distance, chosen.z());
                yaw_off = offs[1] * M_PI / 180.0;
                pitch_off = offs[0] * M_PI / 180.0;

                fire_yaw = raw_yaw_ + yaw_off;
                fire_pitch = raw_pitch + pitch_off;
                is_large = (target.id == ArmorNumber::NO1 || target.id == ArmorNumber::BASE);
                fire_advice = isOnTarget(rpy[2], rpy[1], fire_yaw, fire_pitch, distance, is_large);
                break;
        }

        // 5. 弹道+手动补偿

        double cmd_pitch = raw_pitch + pitch_off;
        double cmd_yaw = normalize_angle(raw_yaw + yaw_off);

        // 6. 填充输出
        GimbalCmd cmd;
        cmd.timestamp = current_time;
        cmd.distance = distance;
        cmd.fire_advice = fire_advice;
        cmd.yaw = cmd_yaw * 180.0 / M_PI;
        cmd.pitch = cmd_pitch * 180.0 / M_PI;
        cmd.yaw_diff = (cmd_yaw - rpy[2]) * 180.0 / M_PI;
        if (cmd.yaw_diff > 180) {
            cmd.yaw_diff -= 360;
        }
        if (cmd.yaw_diff < -180) {
            cmd.yaw_diff += 360;
        }
        cmd.pitch_diff = (cmd_pitch - rpy[1]) * 180.0 / M_PI;
        cmd.v_yaw = v_yaw * 180.0 / M_PI;
        cmd.v_pitch = v_pitch * 180.0 / M_PI;
        cmd.select_id = idx;
        return cmd;
    } else if (best_target.tracking) {
        Eigen::Vector3d pos(
            best_target.position_.x,
            best_target.position_.y,
            best_target.position_.z
        );
        double yaw = best_target.yaw;

        using namespace std::chrono;

        double fly_t = trajectory_compensator_->getFlyingTime(pos);
        auto dt_seconds = duration<double>(fly_t + prediction_delay);
        auto dt = duration_cast<steady_clock::duration>(dt_seconds);
        auto total_dt = (current_time - best_target.timestamp) + dt;
        double dt_seconds_double = duration<double>(total_dt).count();
        auto armor_v = Eigen::Vector3d(
            best_target.velocity_.x,
            best_target.velocity_.y,
            best_target.velocity_.z
        );
        armor_v += dt_seconds_double
            * Eigen::Vector3d(
                       best_target.acceleration_.x,
                       best_target.acceleration_.y,
                       best_target.acceleration_.z
            );
        pos += dt_seconds_double * armor_v;
        yaw += dt_seconds_double * best_target.v_yaw;
        double v_yaw, v_pitch;
        calcVYawAndVPitch(armor_v, rpy, v_yaw, v_pitch);
        double raw_yaw, raw_pitch;
        calcYawAndPitch(pos, rpy, raw_yaw, raw_pitch);
        double distance = pos.norm();
        std::vector<double> offs;
        double pitch_off;
        double yaw_off;
        double fire_yaw;
        double fire_pitch;
        double raw_yaw_, raw_pitch_;
        bool fire_advice = false;
        if (gobal::controller_delay != 0.0) {
            armor_v += gobal::controller_delay
                * Eigen::Vector3d(
                           best_target.acceleration_.x,
                           best_target.acceleration_.y,
                           best_target.acceleration_.z
                );
            pos += gobal::controller_delay * armor_v;
            yaw += gobal::controller_delay * best_target.v_yaw;

            if (pos.norm() < 0.1) {
                throw std::runtime_error("No valid armor after controller delay");
            }
            calcYawAndPitch(pos, rpy, raw_yaw, raw_pitch);
            distance = pos.norm();
        }
        offs = manual_compensator_->angleHardCorrect(distance, pos.z());
        yaw_off = offs[1] * M_PI / 180.0;
        pitch_off = offs[0] * M_PI / 180.0;

        fire_yaw = raw_yaw + yaw_off;
        fire_pitch = raw_pitch + pitch_off;
        bool is_large = (best_target.id == ArmorNumber::NO1 || best_target.id == ArmorNumber::BASE);
        fire_advice = isOnTarget(rpy[2], rpy[1], fire_yaw, fire_pitch, distance, is_large);
        double cmd_pitch = raw_pitch + pitch_off;
        double cmd_yaw = normalize_angle(raw_yaw + yaw_off);

        GimbalCmd cmd;
        cmd.timestamp = current_time;
        cmd.distance = distance;
        cmd.fire_advice = fire_advice;
        cmd.yaw = cmd_yaw * 180.0 / M_PI;
        cmd.pitch = cmd_pitch * 180.0 / M_PI;
        cmd.yaw_diff = (cmd_yaw - rpy[2]) * 180.0 / M_PI;
        if (cmd.yaw_diff > 180) {
            cmd.yaw_diff -= 360;
        }
        if (cmd.yaw_diff < -180) {
            cmd.yaw_diff += 360;
        }
        cmd.pitch_diff = (cmd_pitch - rpy[1]) * 180.0 / M_PI;
        cmd.v_yaw = v_yaw * 180.0 / M_PI;
        cmd.v_pitch = v_pitch * 180.0 / M_PI;
        cmd.select_id = one_idx + target_armor_num;
        return cmd;
    } else {
        return returndefaultCmd();
    }
}
std::vector<GimbalCmd> ArmorSolver::solve_vector(
    const Target& target,
    std::vector<OneTarget> one_targets_,
    std::chrono::steady_clock::time_point current_time,
    int step,
    double dt
) {
    std::vector<GimbalCmd> cmds;
    for (int i = 1; i <= step; ++i) {
        auto cmd = solve(
            target,
            one_targets_,
            current_time + std::chrono::milliseconds(static_cast<int>(std::round(i * dt)))
        );
        cmds.push_back(cmd);
    }
    return cmds;
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
        double dynamic_w = big_shooting_range_w * width_scale;

        // 开火角容差
        shooting_range_yaw = std::abs(std::atan2(big_shooting_range_w / 2.0, distance));
        shooting_range_pitch = std::abs(std::atan2(big_shooting_range_h / 2.0, distance));
    } else {
        // 缩放后的目标宽度
        double dynamic_w = small_shooting_range_w * width_scale;

        // 开火角容差
        shooting_range_yaw = std::abs(std::atan2(small_shooting_range_w / 2.0, distance));
        shooting_range_pitch = std::abs(std::atan2(small_shooting_range_h / 2.0, distance));
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
    const Eigen::Vector3d& v,
    const std::array<double, 3>& rpy,
    double& vyaw,
    double& vpitch
) const noexcept {
    Eigen::Vector3d p = v.normalized();

    const double x = p.x(), y = p.y(), z = p.z();
    const double vx = v.x(), vy = v.y(), vz = v.z();

    const double xy_sq = x * x + y * y;
    const double d_norm_xy = std::sqrt(xy_sq);
    const double d_norm_xyz = std::sqrt(xy_sq + z * z);
    const double eps = 1e-6;
    const double denom_yaw = xy_sq + eps;
    const double denom_pitch = (xy_sq + z * z) * d_norm_xy + eps;

    vyaw = (x * vy - y * vx) / denom_yaw;
    vpitch = ((xy_sq)*vz - z * (x * vx + y * vy)) / denom_pitch;

    // if (trajectory_compensator_) {
    //     double compensated = vpitch;
    //     if (trajectory_compensator_->compensateVelocity(p, v, compensated)) {
    //         vpitch = compensated;
    //     }
    // }
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
    double theta = (target_v_yaw > 0 ? side_angle : -side_angle) / 180.0 * M_PI;

    // Avoid the frequent switch between two armor
    if (std::abs(target_v_yaw) < min_switching_v_yaw) {
        theta = 0;
    }

    double temp_angle = decision_angle + M_PI / armors_num - theta;

    if (temp_angle < 0) {
        temp_angle += 2 * M_PI;
    }

    int selected_id = static_cast<int>(temp_angle / (2 * M_PI / armors_num));
    return selected_id;
}
int ArmorSolver::selectBestTarget(const std::vector<OneTarget>& targets, bool is_target_tracking)
    const noexcept {
    int best_idx = -1;
    auto current_time = std::chrono::steady_clock::now();
    double min_angle_diff = std::numeric_limits<double>::max();
    const OneTarget* best_target_ptr = nullptr;

    // Step 1: 优先选择 is_omni == false 的目标
    for (int pass = 0; pass < 2; ++pass) {
        bool require_omni = (pass == 1); // pass 0: 只选 false，pass 1: 选 true

        // 如果目标已在跟踪中，不再考虑 is_omni == true 的目标
        if (is_target_tracking && require_omni)
            continue;

        for (int i = 0; i < targets.size(); ++i) {
            const auto& tgt = targets[i];
            if (!tgt.tracking)
                continue;
            if (tgt.is_omni != require_omni)
                continue;

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

        if (best_target_ptr)
            break; // 如果本轮找到合适目标，就退出
    }

    if (best_idx == -1 || best_target_ptr == nullptr)
        return -1;

    const OneTarget& best_target = *best_target_ptr;

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
