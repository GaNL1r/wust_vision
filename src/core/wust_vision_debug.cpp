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
#include "common/debug/tools.hpp"
#include "common/debug/toolsgobal.hpp"
#include "common/gobal.hpp"
#include "common/utils.hpp"
#include "core/wust_vision.hpp"

void WustVision::printStats() {
    static int timer_check_count = 0;
    using namespace std::chrono;

    auto now = steady_clock::now();

    if (last_stat_time_steady_.time_since_epoch().count() == 0) {
        last_stat_time_steady_ = now;
        return;
    }

    auto elapsed = duration_cast<duration<double>>(now - last_stat_time_steady_);
    if (elapsed.count() >= 1.0) {
        if (timer_count_ < gobal::control_rate / 10) {
            timer_check_count++;
        }
        if (timer_check_count > 5) {
            if (timer_) {
                auto timercallback =
                    std::bind(&WustVision::timerCallback, this, std::placeholders::_1);
                double rate_hz = static_cast<double>(gobal::control_rate);
                timer_->start(rate_hz, timercallback);
            }
            timer_check_count = 0;
        }
        WUST_INFO(vision_logger_) << "Rec: " << img_recv_count_ << ", Det: " << detect_finish_count_
                                  << ", Fin: " << finish_count_
                                  << ", Lat: " << toolsgobal::latency_ms << "ms"
                                  << ", Fire: " << fire_count_ << ", Tc: " << timer_count_;
        timer_count_ = 0;
        img_recv_count_ = 0;
        detect_finish_count_ = 0;
        fire_count_ = 0;
        finish_count_ = 0;
        last_stat_time_steady_ = now;
    }
}
void WustVision::debugThread() {
    using namespace std::chrono;

    double us_interval = 1e6 / static_cast<double>(toolsgobal::debug_fps);
    auto kInterval = std::chrono::microseconds(static_cast<int64_t>(us_interval));
    while (gobal::is_inited_ && gobal::debug_mode) {
        auto start_time = steady_clock::now();

        debugvisualize(false);
        debuglog();
        writeCmdLogToJson();
        try {
            reloadConfig();
        } catch (std::exception& e) {
            WUST_ERROR(vision_logger_) << "reloadConfig error:" << e.what();
        }

        auto elapsed = steady_clock::now() - start_time;
        if (elapsed < kInterval) {
            std::this_thread::sleep_for(kInterval - elapsed);
        }
    }
}
void WustVision::debuglog() {
    std::lock_guard<std::mutex> lock(dbg_mutex_);
    auto now = std::chrono::steady_clock::now();
    armor::Armors armors;
    armors = debug_gobal_frame_.armors_gobal;
    double t = std::chrono::duration<double>(now - toolsgobal::start_time_).count();
    armor::Target target = debug_gobal_frame_.armor_target;
    writeTargetLogToJson(target);
    {
        std::lock_guard<std::mutex> lock(toolsgobal::robot_cmd_mutex_);

        double armor_yaw = 0.0, ypd_y = 0.0, ypd_p = 0.0, armor_distance = 0.0;
        if (!armors.armors.empty()) {
            std::vector<armor::Armor> ok_armors;
            for (const auto& armor: armors.armors) {
                if (armor.number != armor::ArmorNumber::OUTPOST)
                    ok_armors.push_back(armor);
            }

            if (!ok_armors.empty()) {
                const armor::Armor& min_armor = *std::min_element(
                    ok_armors.begin(),
                    ok_armors.end(),
                    [](const armor::Armor& a, const armor::Armor& b) {
                        return a.distance_to_image_center < b.distance_to_image_center;
                    }
                );

                last_armor_ = min_armor;

                armor_distance = std::hypot(
                    min_armor.target_pos.x,
                    min_armor.target_pos.y,
                    min_armor.target_pos.z
                );

                armor_yaw = last_armor_yaw_
                    + angles::shortest_angular_distance(last_armor_yaw_, min_armor.yaw);
                last_armor_yaw_ = armor_yaw;

                ypd_y = std::atan2(min_armor.target_pos.y, min_armor.target_pos.x);
                ypd_y = last_ypd_y_ + angles::shortest_angular_distance(last_ypd_y_, ypd_y);
                last_ypd_y_ = ypd_y;

                ypd_p = std::atan2(
                    min_armor.target_pos.z,
                    std::hypot(min_armor.target_pos.x, min_armor.target_pos.y)
                );
                last_ypd_p_ = ypd_p;

                last_distance_ = armor_distance;
            }
        }

        DebugLogs& log = toolsgobal::debug_logs_;

        log.time_log.push_back(t);
        log.cmd_yaw_log.push_back(gobal::last_cmd.yaw);
        log.cmd_pitch_log.push_back(gobal::last_cmd.pitch);
        log.rune_obs_log.push_back(rune_solver_->last_observed_angle_);
        log.rune_pre_log.push_back(rune_solver_->last_pre_angle);
        log.rune_v_log.push_back(rune_solver_->curve_fitter_->getFittingParam()[0]);
        log.armor_yaw_log.push_back(armor_yaw * 180.0 / M_PI);
        log.armor_x_log.push_back(last_armor_.target_pos.x);
        log.armor_y_log.push_back(last_armor_.target_pos.y);
        log.armor_z_log.push_back(last_armor_.target_pos.z);
        log.ypd_y_log.push_back(last_ypd_y_ * 180.0 / M_PI);
        log.ypd_p_log.push_back(last_ypd_p_ * 180.0 / M_PI);
        log.armor_dis_log.push_back(last_distance_);

        // 控制长度不超过 1000
        auto trim = [](std::vector<double>& v) {
            if (v.size() > 1000)
                v.erase(v.begin());
        };

        trim(log.time_log);
        trim(log.cmd_yaw_log);
        trim(log.cmd_pitch_log);
        trim(log.rune_obs_log);
        trim(log.rune_pre_log);
        trim(log.rune_v_log);
        trim(log.armor_yaw_log);
        trim(log.armor_x_log);
        trim(log.armor_y_log);
        trim(log.armor_z_log);
        trim(log.ypd_y_log);
        trim(log.ypd_p_log);
        trim(log.armor_dis_log);
    }
}
void WustVision::debugvisualize(bool auto_fps) {
    std::lock_guard<std::mutex> lock(dbg_mutex_);
    auto now = std::chrono::steady_clock::now();
    armor::Armors armors = debug_gobal_frame_.armors_gobal;
    armor::Target target = debug_gobal_frame_.armor_target;
    std::vector<armor::OneTarget> one_targets = debug_gobal_frame_.one_armor_targets;
    AttackMode mode = toAttackMode(gobal::attack_mode);
    bool appear = utils::checkTargetAppear(target, one_targets);
    Tracker::State state = appear ? Tracker::TRACKING : Tracker::LOST;
    GimbalCmd gimbal_cmd = gobal::last_cmd;

    if (mode == AttackMode::ARMOR) {
        armor::Armors armor_data = visualizeTargetProjection(target, one_targets);
        utils::transformArmorData(armor_data, T_camera_to_odom_.inverse());
        Target_info target_info;
        target_info.select_id = gimbal_cmd.select_id;

        if (!gobal::measure_tool->reprojectArmorsCorners(
                armor_data,
                target_info,
                gobal::camera_intrinsic,
                gobal::camera_distortion
            ))
            return;
        try {
            DebugArmor dbg;
            dbg.src_img = debug_gobal_frame_.imgframe;
            dbg.target = target;
            dbg.target_info = target_info;
            dbg.armors = armors;
            dbg.gimbal_cmd = gobal::last_cmd;
            dbg.tracker_state = state;
            drawDebugOverlayShm(dbg, auto_fps);
        } catch (const std::exception& e) {
            std::cerr << "drawDebugArmor failed: " << e.what() << '\n';
        }

    } else {
        double predict_angle = rune_solver_->last_pre_angle - rune_solver_->last_observed_angle_;

        try {
            DebugRune dbg;
            dbg.src_img = debug_gobal_frame_.imgframe;
            dbg.objs = debug_gobal_frame_.rune_objects;
            dbg.predict_angle = predict_angle;
            dbg.gimbal_cmd = gobal::last_cmd;
            dbg.manual_r_box = manual_r_box_;
            dbg.debug_text = rune_solver_->curve_fitter_->getDebugText();
            drawDebugOverlayShm(dbg, auto_fps);
        } catch (const std::exception& e) {
            std::cerr << "drawRuneAndPre failed: " << e.what() << '\n';
        }
    }
}
void WustVision::calculationManualR(const cv::Point2f center) {
    manual_r_runing_ = true;
    const int half_size = 5;
    float x = center.x;
    float y = center.y;
    manual_r_center_ = { x, y };
    manual_r_box_ = { {
        { x - half_size, y - half_size }, // 左上 → 对应点0
        { x - half_size, y + half_size }, // 左下 → 对应点1
        { x + half_size, y + half_size }, // 右下 → 对应点2
        { x + half_size, y - half_size } // 右上 → 对应点3
    } };
    gobal::measure_tool->calcRTarget(
        manual_r_box_,
        T_r_,
        T_camera_to_odom_,
        gobal::camera_intrinsic,
        gobal::camera_distortion
    );
    manual_r_runing_ = false;
}
void WustVision::calculationManualR(const cv::Mat& src_img) {
    manual_r_runing_ = true;
    clicked_points_.clear();
    cv::Mat img_show = src_img.clone();
    cv::cvtColor(img_show, img_show, cv::COLOR_BGR2RGB);
    cv::namedWindow("Manual R Box", cv::WINDOW_NORMAL);
    cv::resizeWindow("Manual R Box", 1280, 960);
    cv::setMouseCallback("Manual R Box", onMouse, nullptr);
    const int half_size = 5;
    while (true) {
        cv::Mat temp = img_show.clone();
        if (!clicked_points_.empty()) {
            manual_r_center_ = clicked_points_.front();
            float x = std::clamp(
                manual_r_center_.x,
                float(half_size),
                float(src_img.cols - half_size - 1)
            );
            float y = std::clamp(
                manual_r_center_.y,
                float(half_size),
                float(src_img.rows - half_size - 1)
            );
            manual_r_center_ = { x, y };
            manual_r_box_ = { {
                { x - half_size, y - half_size }, // 左上 → 对应点0
                { x - half_size, y + half_size }, // 左下 → 对应点1
                { x + half_size, y + half_size }, // 右下 → 对应点2
                { x + half_size, y - half_size } // 右上 → 对应点3
            } };
            cv::circle(temp, manual_r_center_, 3, cv::Scalar(0, 255, 0), -1);
            for (int i = 0; i < 4; ++i)
                cv::line(
                    temp,
                    manual_r_box_[i],
                    manual_r_box_[(i + 1) % 4],
                    cv::Scalar(255, 0, 0),
                    1
                );
        }
        cv::imshow("Manual R Box", temp);
        int key = cv::waitKey(30);
        if (key == 27) { // ESC 退出，不提交
            WUST_INFO("Manual R") << "Manual box canceled.";
            manual_r_init_ = false;
            break;
        }
        if (key == 13 || key == 10) { // 回车提交
            if (!clicked_points_.empty()) {
                manual_r_init_ = true;
                WUST_INFO("Manual R") << "Manual center: (" << manual_r_center_.x << ", "
                                      << manual_r_center_.y << ")";
                WUST_INFO("Manual R") << "Manual R Box Points Saved.";
            } else {
                WUST_ERROR("Manual R") << "No point to submit.";
                manual_r_init_ = false;
            }
            break;
        }
        if (key == 'b' || key == 8) {
            clicked_points_.clear();
            WUST_INFO("Manual R") << "Manual point cleared.";
        }
    }
    gobal::measure_tool->calcRTarget(
        manual_r_box_,
        T_r_,
        T_camera_to_odom_,
        gobal::camera_intrinsic,
        gobal::camera_distortion
    );
    cv::destroyWindow("Manual R Box");
    cv::destroyWindow("Manual R Box");
    cv::destroyWindow("Manual R Box");
    manual_r_runing_ = false;
}

std::vector<cv::Point2f> WustVision::clicked_points_;
void WustVision::onMouse(int event, int x, int y, int, void*) {
    if (event == cv::EVENT_LBUTTONDOWN) {
        clicked_points_.clear();
        clicked_points_.emplace_back(x, y);
        WUST_INFO("Manual R") << "Clicked Point: (" << x << ", " << y << ")";
    }
}
void WustVision::saveAutoLabelData(
    const std::vector<armor::ArmorObject>& objs,
    const CommonFrame& frame
) {
    static int save_counter = 0;

    for (const auto& obj: objs) {
        std::vector<float> csv_data;

        int number_ = formArmorNumber(obj.number);
        int color_ = formArmorColor(obj.color);

        const auto& pts = obj.is_ok ? obj.pts_binary : obj.pts;
        for (int i = 0; i < 4; ++i) {
            csv_data.push_back(pts[i].x);
            csv_data.push_back(pts[i].y);
        }

        csv_data.push_back(number_);
        csv_data.push_back(color_);

        save_counter++;
        if (save_counter % 10 == 0) {
            cv::Mat img_save;
            cv::cvtColor(frame.src_img, img_save, cv::COLOR_RGB2BGR);
            auto_labeler_->save(img_save, csv_data);
        }
    }
}
armor::Armors WustVision::visualizeTargetProjection(
    armor::Target armor_target_,
    std::vector<armor::OneTarget> one_armor_targets_
) {
    armor::Armors armor_data;
    armor_data.frame_id = "gimbal_odom";
    armor_data.timestamp = armor_target_.timestamp;

    if (armor_target_.tracking) {
        tf::Position pos = armor_target_.position_;
        tf::Position vel = armor_target_.velocity_;
        utils::addVelFromAccDt(vel, armor_target_.acceleration_, debug_show_dt_);
        utils::addPosFromVelDt(pos, vel, debug_show_dt_);
        if (pos.norm() > 0.5) {
            double yaw = armor_target_.yaw + armor_target_.v_yaw * debug_show_dt_;
            double r1 = armor_target_.radius_1;
            double r2 = armor_target_.radius_2;
            double d_za = armor_target_.d_za;
            double d_zc = armor_target_.d_zc;
            float xc = pos.x;
            float yc = pos.y;
            float zc = pos.z;
            bool is_current_pair = true;
            armor_data.armors.clear();
            size_t a_n = armor_target_.armors_num;
            armor_data.armors.reserve(a_n);
            for (size_t i = 0; i < a_n; ++i) {
                double tmp_yaw = yaw + i * (2 * M_PI / a_n);
                double cos_yaw = std::cos(tmp_yaw);
                double sin_yaw = std::sin(tmp_yaw);

                tf::Position pos;
                if (a_n == 4) {
                    double r = is_current_pair ? r1 : r2;
                    pos.z = zc + d_zc + (is_current_pair ? 0 : d_za);
                    pos.x = xc - r * cos_yaw;
                    pos.y = yc - r * sin_yaw;
                    is_current_pair = !is_current_pair;
                } else {
                    pos.z = zc;
                    pos.x = xc - r1 * cos_yaw;
                    pos.y = yc - r1 * sin_yaw;
                }

                tf::Quaternion ori;
                ori.setRPY(
                    M_PI / 2,
                    armor_target_.id == armor::ArmorNumber::OUTPOST ? -0.2618 : 0.2618,
                    tmp_yaw
                );

                armor_data.armors.emplace_back(armor::Armor { .type = armor_target_.type,
                                                              .pos = pos,
                                                              .ori = ori,
                                                              .is_ok = true,
                                                              .distance_to_image_center = 0.0f });
            }
        }
    }
    for (auto one_armor_target_: one_armor_targets_) {
        if (one_armor_target_.tracking) {
            tf::Position pos = one_armor_target_.position_;
            tf::Position vel = one_armor_target_.velocity_;
            utils::addVelFromAccDt(vel, one_armor_target_.acceleration_, debug_show_dt_);
            utils::addPosFromVelDt(pos, vel, debug_show_dt_);
            if (pos.norm() > 0.5) {
                double tmp_yaw = one_armor_target_.yaw + one_armor_target_.v_yaw * debug_show_dt_;
                tf::Quaternion ori;
                ori.setRPY(
                    M_PI / 2,
                    one_armor_target_.id == armor::ArmorNumber::OUTPOST ? -0.2618 : 0.2618,
                    tmp_yaw
                );

                armor_data.armors.emplace_back(armor::Armor { .type = one_armor_target_.type,
                                                              .pos = pos,
                                                              .ori = ori,
                                                              .is_ok = false,
                                                              .distance_to_image_center = 0.0f });
            }
        }
    }

    return armor_data;
}

void WustVision::reloadConfig(
) { //只有这个函数可以使用utils::tryGetValue，初始化必须保证参数完全加载
    using namespace std::chrono;
    static steady_clock::time_point last_reload_time = steady_clock::now() - seconds(2);
    static std::unordered_map<std::string, size_t> section_hashes;
    static int count = 0;

    auto now = steady_clock::now();
    if (duration_cast<seconds>(now - last_reload_time).count() < 2) {
        return;
    }
    last_reload_time = now;

    auto new_config = YAML::LoadFile(ROOT_CONFIG);
    if (!new_config) {
        std::cerr << "Failed to load config file or file empty." << std::endl;
        return;
    }

    auto compute_hash = [](const YAML::Node& node) -> size_t {
        if (!node || node.IsNull())
            return 0;
        YAML::Emitter emitter;
        emitter << node;
        std::hash<std::string> hasher;
        return hasher(emitter.c_str());
    };
    auto camera_config = new_config["camera"];
    size_t new_camera_hash = compute_hash(camera_config);
    if (new_camera_hash != section_hashes["camera"]) {
        if (camera_config && camera_ && count != 0) {
            int acquisition_frame_rate;
            utils::tryGetValue<int>(
                camera_config,
                "acquisition_frame_rate",
                acquisition_frame_rate
            );
            int exposure_time;
            utils::tryGetValue<int>(camera_config, "exposure_time", exposure_time);
            double gain;
            utils::tryGetValue<double>(camera_config, "gain", gain);
            double gamma;
            utils::tryGetValue<double>(camera_config, "gamma", gamma);
            std::string adc_bit_depth;
            utils::tryGetValue<std::string>(camera_config, "adc_bit_depth", adc_bit_depth);
            std::string pixel_format;
            utils::tryGetValue<std::string>(camera_config, "pixel_format", pixel_format);
            bool acquisition_frame_rate_enable;
            utils::tryGetValue<bool>(
                camera_config,
                "acquisition_frame_rate_enable",
                acquisition_frame_rate_enable
            );
            bool reverse_x;
            utils::tryGetValue<bool>(camera_config, "reverse_x", reverse_x);
            bool reverse_y;
            utils::tryGetValue<bool>(camera_config, "reverse_y", reverse_y);
            camera_->setParameters(
                acquisition_frame_rate,
                exposure_time,
                gain,
                gamma,
                adc_bit_depth,
                pixel_format,
                acquisition_frame_rate_enable,
                reverse_x,
                reverse_y
            );
        }
        section_hashes["camera"] = new_camera_hash;
    }

    auto shoot_config = new_config["shoot"];
    size_t new_shoot_hash = compute_hash(shoot_config);
    if (new_shoot_hash != section_hashes["shoot"]) {
        if (shoot_config && count != 0) {
            utils::tryGetValue<double>(shoot_config, "bullet_speed", gobal::velocity);
        }
        section_hashes["shoot"] = new_shoot_hash;
    }

    auto tracker_config = new_config["armor_tracker"];
    size_t new_tracker_hash = compute_hash(tracker_config);
    if (new_tracker_hash != section_hashes["armor_tracker"] && tracker_manager_) {
        if (tracker_config && tracker_manager_ && count != 0) {
            auto ekf_config = tracker_config["ekf"];
            if (ekf_config) {
                utils::tryGetValue<double>(ekf_config, "ys2qx_a", tracker_manager_->ys2qx_a_);
                utils::tryGetValue<double>(ekf_config, "ys2qy_a", tracker_manager_->ys2qy_a_);
                utils::tryGetValue<double>(ekf_config, "ys2qz_a", tracker_manager_->ys2qz_a_);
                utils::tryGetValue<double>(ekf_config, "ys2qyaw_a", tracker_manager_->ys2qyaw_a_);
                utils::tryGetValue<double>(ekf_config, "ys2qr_a", tracker_manager_->ys2qr_a_);
                utils::tryGetValue<double>(ekf_config, "ys2qd_zc_a", tracker_manager_->ys2qd_zc_a_);

                utils::tryGetValue<double>(ekf_config, "yr_y_a", tracker_manager_->yr_y_a_);
                utils::tryGetValue<double>(ekf_config, "yr_p_a", tracker_manager_->yr_p_a_);
                utils::tryGetValue<double>(
                    ekf_config,
                    "yr_d_front_a",
                    tracker_manager_->yr_d_front_a_
                );
                utils::tryGetValue<double>(
                    ekf_config,
                    "yr_d_side_a",
                    tracker_manager_->yr_d_side_a_
                );
                utils::tryGetValue<double>(
                    ekf_config,
                    "yr_yaw_front_a",
                    tracker_manager_->yr_yaw_front_a_
                );
                utils::tryGetValue<double>(
                    ekf_config,
                    "yr_yaw_side_a",
                    tracker_manager_->yr_yaw_side_a_
                );

                utils::tryGetValue<double>(ekf_config, "ys2qx_c", tracker_manager_->ys2qx_c_);
                utils::tryGetValue<double>(ekf_config, "ys2qy_c", tracker_manager_->ys2qy_c_);
                utils::tryGetValue<double>(ekf_config, "ys2qz_c", tracker_manager_->ys2qz_c_);
                utils::tryGetValue<double>(ekf_config, "ys2qyaw_c", tracker_manager_->ys2qyaw_c_);
                utils::tryGetValue<double>(ekf_config, "ys2qr_c", tracker_manager_->ys2qr_c_);
                utils::tryGetValue<double>(ekf_config, "ys2qd_zc_c", tracker_manager_->ys2qd_zc_c_);

                utils::tryGetValue<double>(ekf_config, "yr_y_c", tracker_manager_->yr_y_c_);
                utils::tryGetValue<double>(ekf_config, "yr_p_c", tracker_manager_->yr_p_c_);
                utils::tryGetValue<double>(
                    ekf_config,
                    "yr_d_front_c",
                    tracker_manager_->yr_d_front_c_
                );
                utils::tryGetValue<double>(
                    ekf_config,
                    "yr_d_side_c",
                    tracker_manager_->yr_d_side_c_
                );
                utils::tryGetValue<double>(
                    ekf_config,
                    "yr_yaw_front_c",
                    tracker_manager_->yr_yaw_front_c_
                );
                utils::tryGetValue<double>(
                    ekf_config,
                    "yr_yaw_side_c",
                    tracker_manager_->yr_yaw_side_c_
                );

                utils::tryGetValue<double>(ekf_config, "oys2qx", tracker_manager_->oys2qx_);
                utils::tryGetValue<double>(ekf_config, "oys2qy", tracker_manager_->oys2qy_);
                utils::tryGetValue<double>(ekf_config, "oys2qz", tracker_manager_->oys2qz_);
                utils::tryGetValue<double>(ekf_config, "oys2qyaw", tracker_manager_->oys2qyaw_);

                utils::tryGetValue<double>(ekf_config, "oyr_y", tracker_manager_->oyr_y_);
                utils::tryGetValue<double>(ekf_config, "oyr_p", tracker_manager_->oyr_p_);
                utils::tryGetValue<double>(
                    ekf_config,
                    "oyr_d_front",
                    tracker_manager_->oyr_d_front_
                );
                utils::tryGetValue<double>(ekf_config, "oyr_d_side", tracker_manager_->oyr_d_side_);
                utils::tryGetValue<double>(
                    ekf_config,
                    "oyr_yaw_front",
                    tracker_manager_->oyr_yaw_front_
                );
                utils::tryGetValue<double>(
                    ekf_config,
                    "oyr_yaw_side",
                    tracker_manager_->oyr_yaw_side_
                );

                utils::tryGetValue<double>(ekf_config, "r_v", tracker_manager_->r_v);
                utils::tryGetValue<double>(ekf_config, "q_v", tracker_manager_->q_v);
                utils::tryGetValue<double>(ekf_config, "q_a", tracker_manager_->q_a);
            }
        }
        section_hashes["armor_tracker"] = new_tracker_hash;
    }

    auto armor_solver_config = new_config["armor_solver"];
    size_t new_armor_solver_hash = compute_hash(armor_solver_config);
    if (new_armor_solver_hash != section_hashes["armor_solver"]) {
        if (armor_solver_config && armor_solver_ && count != 0) {
            utils::tryGetValue<double>(
                armor_solver_config,
                "small_shooting_range_w",
                armor_solver_->small_shooting_range_w_
            );
            utils::tryGetValue<double>(
                armor_solver_config,
                "small_shooting_range_h",
                armor_solver_->small_shooting_range_h_
            );
            utils::tryGetValue<double>(
                armor_solver_config,
                "big_shooting_range_w",
                armor_solver_->big_shooting_range_w_
            );
            utils::tryGetValue<double>(
                armor_solver_config,
                "big_shooting_range_h",
                armor_solver_->big_shooting_range_h_
            );
            utils::tryGetValue<double>(
                armor_solver_config,
                "max_tracking_v_yaw",
                armor_solver_->max_tracking_v_yaw_
            );
            utils::tryGetValue<double>(
                armor_solver_config,
                "prediction_delay",
                armor_solver_->prediction_delay_
            );
            utils::tryGetValue<double>(
                armor_solver_config,
                "side_angle",
                armor_solver_->side_angle_
            );
            utils::tryGetValue<double>(
                armor_solver_config,
                "min_switching_v_yaw",
                armor_solver_->min_switching_v_yaw_
            );
            utils::tryGetValue<double>(
                armor_solver_config,
                "gravity",
                armor_solver_->trajectory_compensator_->gravity_
            );
            utils::tryGetValue<double>(
                armor_solver_config,
                "resistance",
                armor_solver_->trajectory_compensator_->resistance_
            );
            utils::tryGetValue<int>(
                armor_solver_config,
                "iteration_times",
                armor_solver_->trajectory_compensator_->iteration_times_
            );
            utils::tryGetValue<double>(
                armor_solver_config,
                "oneswitch_position_thres",
                armor_solver_->oneswitch_position_thres_
            );
            utils::tryGetValue<double>(
                armor_solver_config,
                "oneswitch_angle_thres",
                armor_solver_->oneswitch_angle_thres_
            );
            utils::tryGetValue<double>(
                armor_solver_config,
                "oneswitch_hold_time",
                armor_solver_->oneswitch_hold_time_
            );
            //armor_solver_->manual_compensator_->updateMapFlow(utils::getOffsetEntry(armor_solver_config));
        }
        section_hashes["armor_solver"] = new_armor_solver_hash;
    }

    auto armor_optimize_config = new_config["armor_optimize"];
    size_t new_armor_optimize_hash = compute_hash(armor_optimize_config);
    if (new_armor_optimize_hash != section_hashes["armor_optimize"] && armor_pose_estimator_) {
        if (armor_optimize_config && armor_pose_estimator_ && count != 0) {
            utils::tryGetValue<int>(
                armor_optimize_config,
                "max_iter_R",
                armor_pose_estimator_->ba_solver_->max_iter_R_
            );
            utils::tryGetValue<int>(
                armor_optimize_config,
                "max_iter_t",
                armor_pose_estimator_->ba_solver_->max_iter_t_
            );
            utils::tryGetValue<int>(
                armor_optimize_config,
                "step_R",
                armor_pose_estimator_->ba_solver_->step_R_
            );
            utils::tryGetValue<int>(
                armor_optimize_config,
                "step_t",
                armor_pose_estimator_->ba_solver_->step_t_
            );
            utils::tryGetValue<double>(
                armor_optimize_config,
                "min_error_R",
                armor_pose_estimator_->ba_solver_->min_error_R_
            );
            utils::tryGetValue<double>(
                armor_optimize_config,
                "min_error_t",
                armor_pose_estimator_->ba_solver_->min_error_t_
            );
            utils::tryGetValue<double>(
                armor_optimize_config,
                "distance_fix_a2",
                armor_pose_estimator_->distance_fix_a2_
            );
        }
        section_hashes["armor_optimize"] = new_armor_optimize_hash;
    }

    auto rune_solver_config = new_config["rune_solver"];
    size_t new_rune_solver_hash = compute_hash(rune_solver_config);
    if (new_rune_solver_hash != section_hashes["rune_solver"] && rune_solver_) {
        if (rune_solver_config && rune_solver_ && count != 0) {
            utils::tryGetValue<double>(
                rune_solver_config,
                "gravity",
                rune_solver_->trajectory_compensator_->gravity_
            );
            utils::tryGetValue<double>(
                rune_solver_config,
                "resistance",
                rune_solver_->trajectory_compensator_->resistance_
            );
            utils::tryGetValue<int>(
                rune_solver_config,
                "iteration_times",
                rune_solver_->trajectory_compensator_->iteration_times_
            );
            //rune_solver_->manual_compensator_->updateMapFlow(utils::getOffsetEntry(rune_solver_config));
            auto ekf_config = rune_solver_config["ekf"];
            if (ekf_config) {
                utils::tryGetValue<std::vector<double>>(
                    ekf_config,
                    "q_ypdyaw",
                    rune_solver_->yq_vec_
                );
                utils::tryGetValue<std::vector<double>>(
                    ekf_config,
                    "r_ypdyaw",
                    rune_solver_->yr_vec_
                );
            }
        }
        section_hashes["rune_solver"] = new_rune_solver_hash;
    }

    gobal::config = new_config;
    count++;
}