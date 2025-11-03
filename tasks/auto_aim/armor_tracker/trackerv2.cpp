// #include "trackerv2.hpp"
// #include <algorithm>

// void TrackerV2::track(const armor::Armors& armors_msg) {
//     armor::Armors armors;
//     armors = armors_msg;
//     std::erase_if(armors.armors, [](const armor::Armor& a) { return !a.is_ok; });

//     // 排序
//     std::sort(
//         armors.armors.begin(),
//         armors.armors.end(),
//         [](const armor::Armor& a, const armor::Armor& b) {
//             return a.distance_to_image_center < b.distance_to_image_center;
//         }
//     );
//     bool found;
//     if (tracker_state == LOST) {
//         found = initTarget(armors);
//     } else {
//         found = updateTarget(armors);
//     }
//     if (tracker_state == DETECTING) {
//         if (found) {
//             detect_count_++;
//             if (detect_count_ > tracking_thres_) {
//                 detect_count_ = 0;
//                 tracker_state = TRACKING;
//             }
//         } else {
//             detect_count_ = 0;
//             tracker_state = LOST;
//         }
//     } else if (tracker_state == TRACKING) {
//         if (!found) {
//             tracker_state = TEMP_LOST;
//             lost_count_++;
//         }
//     } else if (tracker_state == TEMP_LOST) {
//         if (!found) {
//             lost_count_++;
//             if (lost_count_ > lost_thres_) {
//                 lost_count_ = 0;
//                 tracker_state = LOST;
//             }
//         } else {
//             tracker_state = TRACKING;
//             lost_count_ = 0;
//         }
//     }
// }
// bool TrackerV2::initTarget(const armor::Armors& armors) {
//     if (armors.armors.empty()) {
//         return false;
//     }
//     auto a = armors.armors.front();
//     double xa = a.target_pos.x();
//     double ya = a.target_pos.y();
//     double za = a.target_pos.z();
//     last_yaw_ = 0;
//     double yaw = orientationToYaw(a.target_ori);

//     target_state_ = Eigen::VectorXd::Zero(ypdv2armor_motion_model::X_N);
//     double r = 0.24;
//     double xc = xa + r * cos(yaw);
//     double yc = ya + r * sin(yaw);
//     double zc = za;
//     target_state_ << xc, 0, yc, 0, zc, 0, yaw, 0, r, 0, 0;
//     ekf_ypd_->setState(target_state_);
//     tracked_id_ = a.number;
//     if (tracked_id_ == armor::ArmorNumber::OUTPOST) {
//         armor_num_ = 3;
//     } else {
//         armor_num_ = 4;
//     }
//     type_ = a.type;
//     tracker_state = DETECTING;
//     return true;
// }
// bool TrackerV2::updateTarget(const armor::Armors& armors) {
//     int found_count = 0;
//     double min_x = 1e10;
//     Eigen::VectorXd ekf_prediction = ekf_ypd_->predict();
//     target_state_ = ekf_prediction;
//     for (const auto& armor: armors.armors) {
//         if (!armor::isSameTarget(armor.number, tracked_id_))
//             continue;
//         found_count++;
//     }
//     if (found_count == 0)
//         return false;
//     for (auto& armor: armors.armors) {
//         if (!armor::isSameTarget(armor.number, tracked_id_))
//             continue;
//         updateekf(armor, ekf_prediction);
//     }

//     return true;
// }
// Eigen::Vector3d TrackerV2::h_armor_xyz(const Eigen::VectorXd& x, int id) {
//     auto angle = angles::normalize_angle(x[6] + id * 2 * CV_PI / armor_num_);
//     auto use_l_h = (armor_num_ == 4) && (id == 1 || id == 3);

//     auto r = (use_l_h) ? x[8] + x[9] : x[8];
//     auto armor_x = x[0] - r * std::cos(angle);
//     auto armor_y = x[2] - r * std::sin(angle);
//     auto armor_z = (use_l_h) ? x[4] + x[10] : x[4];

//     return { armor_x, armor_y, armor_z };
// }
// std::vector<Eigen::Vector4d> TrackerV2::getArmorPosAndYaw(const Eigen::VectorXd& ekf_prediction) {
//     std::vector<Eigen::Vector4d> _armor_xyza_list;

//     for (int i = 0; i < armor_num_; i++) {
//         auto angle = angles::normalize_angle(ekf_prediction[6] + i * 2 * CV_PI / armor_num_);
//         Eigen::Vector3d xyz = h_armor_xyz(ekf_prediction, i);
//         _armor_xyza_list.push_back({ xyz[0], xyz[1], xyz[2], angle });
//     }
//     return _armor_xyza_list;
// }
// void TrackerV2::updateekf(const armor::Armor& armor, const Eigen::VectorXd& ekf_prediction) {
//     int id;
//     auto min_angle_error = 1e10;
//     const std::vector<Eigen::Vector4d>& xyza_list = getArmorPosAndYaw(ekf_prediction);

//     std::vector<std::pair<Eigen::Vector4d, int>> xyza_i_list;
//     for (int i = 0; i < armor_num_; i++) {
//         xyza_i_list.push_back({ xyza_list[i], i });
//     }

//     std::sort(
//         xyza_i_list.begin(),
//         xyza_i_list.end(),
//         [](const std::pair<Eigen::Vector4d, int>& a, const std::pair<Eigen::Vector4d, int>& b) {
//             Eigen::Vector3d ypd1 = utils::xyz2ypd(a.first.head(3));
//             Eigen::Vector3d ypd2 = utils::xyz2ypd(b.first.head(3));
//             return ypd1[2] < ypd2[2];
//         }
//     );

//     // 取前3个distance最小的装甲板
//     for (int i = 0; i < 3; i++) {
//         const auto& xyza = xyza_i_list[i].first;
//         Eigen::Vector3d ypd = utils::xyz2ypd(xyza.head(3));
//         auto angle_error =
//             std::abs(angles::normalize_angle(orientationToYaw(armor.target_ori) - xyza[3]))
//             + std::abs(angles::normalize_angle(orientationToYaw(armor.target_ori) - ypd[0]));

//         if (std::abs(angle_error) < std::abs(min_angle_error)) {
//             id = xyza_i_list[i].second;
//             min_angle_error = angle_error;
//         }
//     }

//     if (id != 0)
//         jumped = true;

//     if (id != last_id) {
//         is_switch_ = true;
//     } else {
//         is_switch_ = false;
//     }

//     if (is_switch_)
//         switch_count_++;

//     last_id = id;
//     update_count_++;
//     ekf_ypd_->setMeasureFunc(ypdv2armor_motion_model::Measure { id, armor_num_ });
//     auto p = armor.target_pos;
//     double measured_yaw = orientationToYaw(armor.target_ori);

//     double ypd_y = std::atan2(p.y(), p.x());
//     ypd_y = this->last_ypd_y + angles::shortest_angular_distance(this->last_ypd_y, ypd_y);
//     this->last_ypd_y = ypd_y;
//     double ypd_p = std::atan2(p.z(), std::sqrt(p.x() * p.x() + p.y() * p.y()));
//     double ypd_d = std::sqrt(p.x() * p.x() + p.y() * p.y() + p.z() * p.z());
//     measurement_ = Eigen::Vector4d(ypd_y, ypd_p, ypd_d, measured_yaw);
//     ekf_ypd_->update(measurement_);
// }