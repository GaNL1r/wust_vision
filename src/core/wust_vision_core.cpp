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
#include "common/calculation.hpp"
#include "common/debug/toolsgobal.hpp"
#include "common/utils.hpp"
#include "core/wust_vision.hpp"

void WustVision::processImage(const ImageFrame& frame) {
    CommonFrame common_frame;
    common_frame.timestamp = frame.timestamp;
    common_frame.v = frame.v;

    if (!use_video_) {
        common_frame.src_img = convertToMat(frame);
    } else {
        //common_frame.src_img = convertToMatbgr(frame);
        if (!frame.src_img.empty()) {
            common_frame.src_img = std::move(frame.src_img);
            //common_frame.src_img.convertTo(common_frame.src_img, -1, video_alpha_, video_beta_);
        }
    }
    common_frame.T_camera_to_odom = utils::computeCameraToOdomTransform(
        frame.R_gimbal2odom,
        R_camera2gimbal_,
        t_gimbal_to_camera_
    );
    common_frame.id = current_id_++;

    printStats();

    AttackMode mode = toAttackMode(gobal::attack_mode);
    switch (mode) {
        case AttackMode::ARMOR: {
            if (armor_detector_) {
                armor_detector_->pushInput(common_frame);
            }
        } break;
        case AttackMode::SMALL_RUNE: {
            // if (use_manual_r_ && !manual_r_init_ && !manual_r_runing_) {
            //     cv::Point2f center(common_frame.src_img.cols/2.0,common_frame.src_img.rows/2.0);
            //     calculationManualR(center);
            //     return;
            // }
            if (use_manual_r_ && gobal::if_manual_reset) {
                cv::Point2f center(
                    common_frame.src_img.cols / 2.0,
                    common_frame.src_img.rows / 2.0
                );
                calculationManualR(center);
            }
            if (rune_detector_) {
                rune_detector_->pushInput(common_frame);
            }

        } break;
        case AttackMode::BIG_RUNE: {
            // if (use_manual_r_ && !manual_r_init_ && !manual_r_runing_) {
            //     calculationManualR(common_frame.src_img);
            //     return;
            // }
            if (use_manual_r_ && gobal::if_manual_reset) {
                cv::Point2f center(
                    common_frame.src_img.cols / 2.0,
                    common_frame.src_img.rows / 2.0
                );
                calculationManualR(center);
            }

            if (rune_detector_) {
                rune_detector_->pushInput(common_frame);
            }
        } break;
        case AttackMode::UNKNOWN: {
            if (armor_detector_) {
                armor_detector_->pushInput(common_frame);
            }
        } break;
    }
}

void WustVision::ArmorDetectCallback(
    const std::vector<armor::ArmorObject>& objs,
    const CommonFrame& frame
) {
    std::lock_guard<std::mutex> lock(callback_mutex_);
    std::vector<armor::ArmorObject> sorted_objs = objs;

    if (sorted_objs.size() > max_detect_armors_) {
        WUST_WARN(vision_logger_) << "Detected " << sorted_objs.size()
                                  << " objects, too many, keeping top " << max_detect_armors_;

        std::partial_sort(
            sorted_objs.begin(),
            sorted_objs.begin() + max_detect_armors_,
            sorted_objs.end(),
            [](const armor::ArmorObject& a, const armor::ArmorObject& b) {
                return a.confidence > b.confidence;
            }
        );

        sorted_objs.resize(max_detect_armors_);
    }

    armor::Armors armors;
    armors.timestamp = frame.timestamp;
    armors.frame_id = "camera_optical_frame";

    armors.armors = armor_pose_estimator_->extractArmorPoses(
        sorted_objs,
        frame.T_camera_to_odom,
        gobal::camera_intrinsic,
        gobal::camera_distortion
    );

    gobal::measure_tool->processDetectedArmors(
        sorted_objs,
        armors,
        frame.T_camera_to_odom,
        gobal::camera_intrinsic,
        gobal::camera_distortion
    );

    if (use_auto_labeler_) {
        saveAutoLabelData(objs, frame);
    }
    Eigen::Matrix3d R_gimbal2odom =
        utils::getRGimbalToOdom(T_camera_to_odom_, R_camera2gimbal_, t_gimbal_to_camera_);
    armors.R_gimbal2odom = R_gimbal2odom;
    armors.v = frame.v;
    armors.id = frame.id;
    armor_queue_->enqueue(armors);
    if (gobal::debug_mode) {
        std::lock_guard<std::mutex> lock(dbg_mutex_);
        debug_gobal_frame_.imgframe.img = frame.src_img.clone();
        debug_gobal_frame_.imgframe.timestamp = armors.timestamp;
        debug_gobal_frame_.armors_gobal = armors;
    }
    T_camera_to_odom_ = frame.T_camera_to_odom;
    detect_finish_count_++;
}

void WustVision::RuneDetectCallback(std::vector<rune::RuneObject>& objs, const CommonFrame& frame) {
    std::lock_guard<std::mutex> lock(callback_mutex_);
    static bool last_rune_big = false;

    rune::Rune rune_target { .frame_id = "camera_optical_frame",
                             .timestamp = frame.timestamp,
                             .is_big_rune = false,
                             .is_lost = true };

    cv::Mat debug_img;
    if (gobal::debug_mode)
        debug_img = frame.src_img.clone();

    if (!objs.empty()) {
        // 1. 按概率排序
        std::sort(objs.begin(), objs.end(), [](auto& a, auto& b) { return a.prob > b.prob; });

        // 2. 计算 R tag 中心
        cv::Point2f r_tag;
        cv::Mat binary_roi(1, 1, CV_8UC3, cv::Scalar(0));

        auto detectRTagAt = [&](const cv::Point2f& center, bool manual) {
            return rune_detector_->detectRTag(frame.src_img, rune_binary_thresh_, center, manual);
        };

        if (use_manual_r_ && manual_r_init_) {
            gobal::measure_tool->projectRTargetToImage(
                T_r_,
                frame.T_camera_to_odom,
                manual_r_box_,
                gobal::camera_intrinsic,
                gobal::camera_distortion
            );
            manual_r_center_ = utils::computeCenter(manual_r_box_);
            r_tag = manual_r_center_;
            if (detect_r_tag_ && !frame.src_img.empty())
                std::tie(r_tag, binary_roi) = detectRTagAt(manual_r_center_, true);
        } else if (detect_r_tag_ && !frame.src_img.empty()) {
            std::tie(r_tag, binary_roi) = detectRTagAt(objs.front().pts.r_center, false);
        } else {
            r_tag = std::accumulate(
                objs.begin(),
                objs.end(),
                cv::Point2f(0, 0),
                [n = float(objs.size())](cv::Point2f p, auto& o) { return p + o.pts.r_center / n; }
            );
        }

        // 3. 赋值 R tag 中心
        for (auto& o: objs)
            o.pts.r_center = r_tag;

        // 4. 绘制调试图
        if (gobal::debug_mode && !debug_img.empty()) {
            cv::Rect roi(debug_img.cols - binary_roi.cols, 0, binary_roi.cols, binary_roi.rows);
            binary_roi.copyTo(debug_img(roi));
            cv::rectangle(debug_img, roi, cv::Scalar(150, 150, 150), 2);
        }

        // 5. 选择目标（未激活 & 颜色匹配）
        auto target_it =
            std::find_if(objs.begin(), objs.end(), [c = EnemyColor(gobal::detect_color)](auto& o) {
                return o.type == rune::RuneType::INACTIVATED && o.color == c;
            });

        if (target_it != objs.end()) {
            rune_target.is_lost = false;
            auto& p = target_it->pts;
            rune_target.pts[0].x = p.r_center.x;
            rune_target.pts[0].y = p.r_center.y;
            rune_target.pts[1].x = p.bottom_left.x;
            rune_target.pts[1].y = p.bottom_left.y;
            rune_target.pts[2].x = p.top_left.x;
            rune_target.pts[2].y = p.top_left.y;
            rune_target.pts[3].x = p.top_right.x;
            rune_target.pts[3].y = p.top_right.y;
            rune_target.pts[4].x = p.bottom_right.x;
            rune_target.pts[4].y = p.bottom_right.y;
        }
    }

    // 6. 攻击模式切换
    switch (toAttackMode(gobal::attack_mode)) {
        case AttackMode::BIG_RUNE:
            rune_target.is_big_rune = last_rune_big = true;
            break;
        case AttackMode::SMALL_RUNE:
            rune_target.is_big_rune = last_rune_big = false;
            break;
        case AttackMode::ARMOR:
        case AttackMode::UNKNOWN:
            rune_target.is_big_rune = last_rune_big;
            break;
    }

    rune_target.id = frame.id;
    rune_target.T_camera_to_odom = frame.T_camera_to_odom;
    rune_queue_->enqueue(rune_target);
    //runeTargetCallback(rune_target);
    T_camera_to_odom_ = frame.T_camera_to_odom;

    auto latency_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
                          std::chrono::steady_clock::now() - rune_target.timestamp
    )
                          .count();
    toolsgobal::latency_averager.add(latency_ns);
    toolsgobal::latency_ms = toolsgobal::latency_averager.average_ms();

    // 8. 保存调试信息
    if (gobal::debug_mode) {
        std::lock_guard<std::mutex> lock(dbg_mutex_);
        debug_gobal_frame_.imgframe = { debug_img.clone(), rune_target.timestamp };
        debug_gobal_frame_.rune_objects = objs;
    }

    detect_finish_count_++;
}

void WustVision::processingLoop() {
    while (gobal::is_inited_) {
        auto mode = toAttackMode(gobal::attack_mode);
        switch (mode) {
            case AttackMode::ARMOR: {
                armor::Armors armors_;
                if (!armor_queue_->try_dequeue(armors_)) {
                    std::this_thread::sleep_for(std::chrono::milliseconds(1));
                    continue;
                }

                armorsCallback(armors_);
                finish_count_++;
            }
            case AttackMode::SMALL_RUNE:
            case AttackMode::BIG_RUNE:

            {
                rune::Rune rune;
                if (!rune_queue_->try_dequeue(rune)) {
                    std::this_thread::sleep_for(std::chrono::milliseconds(1));
                    continue;
                }
                runeTargetCallback(rune);
                finish_count_++;
            }
            case AttackMode::UNKNOWN:
            default: {
                armor::Armors armors;
                if (!armor_queue_->try_dequeue(armors)) {
                    std::this_thread::sleep_for(std::chrono::milliseconds(1));
                    continue;
                }
                armorsCallback(armors);
                finish_count_++;
            }
        }
    }
}
void WustVision::runeTargetCallback(const rune::Rune rune_target) {
    if (rune_target.timestamp <= last_rune_target_time_) {
        WUST_WARN(vision_logger_) << "Received out-of-order rune data, discarded.";
        return;
    }
    last_rune_target_time_ = rune_target.timestamp;
    if (rune_solver_->pnp_solver_ == nullptr) {
        return;
    }
    double observed_angle = 0;
    if (rune_solver_->tracker_state == RuneSolver::LOST) {
        observed_angle = rune_solver_->init(rune_target, rune_target.T_camera_to_odom);
    } else {
        observed_angle = rune_solver_->update(rune_target, rune_target.T_camera_to_odom);
    }

    auto now = std::chrono::steady_clock::now();
    auto latency_nano =
        std::chrono::duration_cast<std::chrono::nanoseconds>(now - rune_target.timestamp).count();
    toolsgobal::latency_ms = static_cast<double>(latency_nano) / 1e6;
}

void WustVision::armorsCallback(armor::Armors armors) {
    if (armors.timestamp <= tracker_manager_->last_time_) {
        WUST_WARN(vision_logger_) << "Received out-of-order armor data, discarded.";
        return;
    }
    if (use_calculation_) {
        commandCallbackYpd(armors);
    }
    armor::Target target;
    std::vector<armor::OneTarget> one_targets;
    auto time = armors.timestamp;

    tracker_manager_->update(target, one_targets, armors, time, armors.R_gimbal2odom, armors.v);
    {
        std::lock_guard<std::mutex> lock(armor_solver_target_mutex_);
        armor_solver_target_.target = target;
        armor_solver_target_.one_targets = one_targets;
    }

    auto now = std::chrono::steady_clock::now();

    auto latency_nano =
        std::chrono::duration_cast<std::chrono::nanoseconds>(now - target.timestamp).count();
    toolsgobal::latency_averager.add(latency_nano);
    toolsgobal::latency_ms = toolsgobal::latency_averager.average_ms();
    if (gobal::debug_mode) {
        std::lock_guard<std::mutex> lock(dbg_mutex_);
        debug_gobal_frame_.armor_target = target;
        debug_gobal_frame_.one_armor_targets = one_targets;
    }
}

GimbalCmd WustVision::solveByMode(
    AttackMode mode,
    const ArmorSolverTarget& armor_solver_target,
    const std::chrono::steady_clock::time_point& now
) {
    switch (mode) {
        case AttackMode::ARMOR: {
            auto cmd = armor_solver_->solve(armor_solver_target, now);
            auto next_time =
                now + std::chrono::microseconds(static_cast<int64_t>(1e6 / gobal::control_rate));
            auto next_cmd = armor_solver_->solve(armor_solver_target, next_time);
            if (std::abs(cmd.yaw - next_cmd.yaw) > jump_yaw
                || std::abs(cmd.yaw - gobal::last_cmd.yaw) > jump_yaw)
                cmd.fire_advice = false;
            return cmd;
        }
        case AttackMode::SMALL_RUNE:
        case AttackMode::BIG_RUNE:
            return rune_solver_->solve();
        case AttackMode::UNKNOWN:
        default:
            return armor_solver_->solve(armor_solver_target, now);
    }
}
void WustVision::timerCallback(double dt_ms) {
    if (!gobal::is_inited_)
        return;

    auto now = std::chrono::steady_clock::now();
    timer_count_++;
    armor::Target target;
    std::vector<armor::OneTarget> one_targets;
    {
        std::lock_guard<std::mutex> lock(armor_solver_target_mutex_);
        target = armor_solver_target_.target;
        one_targets = armor_solver_target_.one_targets;
    }

    if (std::chrono::duration<double>(now - last_track_target_).count() > hit_omni_dt_) {
        for (const auto& omni: gobal::omni_targets) {
            if (std::abs(std::chrono::duration<double>(now - omni.timestamp).count())
                <= receive_omni_dt_) {
                one_targets.push_back(omni);
            }
        }
    }

    bool appear = utils::checkTargetAppear(target, one_targets);
    Tracker::State state = appear ? Tracker::TRACKING : Tracker::LOST;
    if (appear)
        last_track_target_ = now;
    AttackMode mode = toAttackMode(gobal::attack_mode);

    GimbalCmd gimbal_cmd;

    if (appear || rune_solver_->tracker_state == Tracker::TRACKING) {
        if (appear || rune_solver_->tracker_state == Tracker::TRACKING) {
            try {
                ArmorSolverTarget armor_solver_target;
                armor_solver_target.one_targets = one_targets;
                armor_solver_target.target = target;
                gimbal_cmd = solveByMode(mode, armor_solver_target, now);
                gobal::last_cmd = gimbal_cmd;
                if (gimbal_cmd.fire_advice)
                    fire_count_++;
            } catch (const std::exception& e) {
                std::cerr << "solver error: " << e.what() << '\n';
                gimbal_cmd = gobal::last_cmd;
            }
        } else {
            gimbal_cmd = gobal::last_cmd;
        }

        serial_->transformGimbalCmd(gimbal_cmd, appear);
    }

    if (gobal::debug_mode) {
        //debuglog();
    }
}
