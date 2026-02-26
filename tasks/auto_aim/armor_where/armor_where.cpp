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

#include "armor_where.hpp"
#include "wust_vl/algorithm/pnp_solver.hpp"
#include <opencv2/core/eigen.hpp>
namespace wust_vision {
namespace auto_aim {

    struct ArmorWhere::Impl {
    public:
        Impl(const YAML::Node& config, const std::pair<cv::Mat, cv::Mat>& camera_info) {
            camera_info_ = camera_info;
            params_.load(config);
            pnp_solver_ = std::make_unique<wust_vl::algorithm::PnPSolver>(cv::SOLVEPNP_IPPE);
            pnp_solver_->setObjectPoints(
                "small",
                ArmorObject::buildObjectPoints<cv::Point3f>(SMALL_ARMOR_WIDTH, SMALL_ARMOR_HEIGHT)
            );
            pnp_solver_->setObjectPoints(
                "large",
                ArmorObject::buildObjectPoints<cv::Point3f>(LARGE_ARMOR_WIDTH, LARGE_ARMOR_HEIGHT)
            );
        }
        struct Params {
            enum class OptMode : int { GOLDEN = 0, CERES = 1, NONE = 2 } opt_mode;
            OptMode fromString(const std::string& mode) {
                if (mode == "golden" || mode == "GOLDEN") {
                    return OptMode::GOLDEN;
                } else if (mode == "none" || mode == "NONE") {
                    return OptMode::NONE;
                } else {
                    return OptMode::NONE;
                }
            }
            int ceres_max_iter = 40;
            int golden_search_side_deg = 60;
            double distance_fix_a2 = 0;
            void load(const YAML::Node& node) {
                opt_mode = fromString(node["yaw_opt"]["mode"].as<std::string>());
                ceres_max_iter = node["yaw_opt"]["ceres_max_iter"].as<int>();
                golden_search_side_deg = node["yaw_opt"]["golden_search_side_deg"].as<int>();
                distance_fix_a2 = node["distance_fix_a2"].as<double>();
            }
        } params_;

        std::vector<Armor> where(
            const std::vector<ArmorObject>& armors,
            Eigen::Matrix4d T_camera_to_odom
        ) const noexcept {
            std::vector<Armor> armors_msg;

            const Eigen::Matrix3d R_imu_cam = T_camera_to_odom.block<3, 3>(0, 0);
            auto makeArmor =
                [&](const ArmorObject& obj, const Eigen::Vector3d& t, const Eigen::Matrix3d& R) {
                    Armor msg;
                    msg.type = (obj.number == ArmorNumber::NO1 || obj.number == ArmorNumber::BASE)
                        ? "large"
                        : "small";
                    msg.number = obj.number;
                    Eigen::Quaterniond q(R);
                    Eigen::Quaterniond add_roll {
                        Eigen::AngleAxisd(M_PI / 2, Eigen::Vector3d::UnitX())
                    };
                    Eigen::Quaterniond new_q = q * add_roll;

                    auto [yaw, pitch, dist] = utils::xyz2ypd_rad(t.x(), t.y(), t.z());
                    dist += params_.distance_fix_a2 * dist * dist;
                    auto [x, y, z] = utils::ypd2xyz_rad(yaw, pitch, dist);

                    msg.pos = { x, y, z };
                    msg.ori = new_q;
                    auto_aim::transformArmorData(msg, T_camera_to_odom);
                    msg.distance_to_image_center =
                        pnp_solver_->calculateDistanceToCenter(obj.center, camera_info_.first);
                    msg.is_ok = obj.is_ok;
                    if (obj.color == ArmorColor::NONE || obj.color == ArmorColor::PURPLE) {
                        msg.is_none_purple = true;
                    } else {
                        msg.is_none_purple = false;
                    }
                    return msg;
                };

            for (auto const& a: armors) {
                cv::Mat rvec, tvec;
                std::string type = (a.number == ArmorNumber::NO1 || a.number == ArmorNumber::BASE)
                    ? "large"
                    : "small";

                if (!pnp_solver_->solvePnP(
                        a.landmarks(),
                        rvec,
                        tvec,
                        type,
                        camera_info_.first,
                        camera_info_.second
                    ))
                {
                    WUST_WARN("PNP") << "PNP failed";
                    continue;
                }
                cv::Mat R_cv;
                cv::Rodrigues(rvec, R_cv);
                Eigen::Matrix3d R = utils::cvToEigen(R_cv);
                Eigen::Vector3d t = utils::cvToEigen(tvec);
                if (params_.opt_mode != Params::OptMode::NONE) {
                    Eigen::Matrix3d R0 = R;
                    R = solveBa_R(a, t, R0, R_imu_cam, type);
                }

                armors_msg.push_back(makeArmor(a, t, R));
            }

            return armors_msg;
        }
        std::vector<Eigen::Vector2d> reprojectionArmor(
            double yaw,
            const std::vector<cv::Point3f>& object_points,
            const std::vector<cv::Point2f>& landmarks,
            const Eigen::Matrix3d& Rci,
            double pitch,
            double roll,
            const Eigen::Vector3d& t
        ) const noexcept {
            const Eigen::AngleAxisd ay(yaw, Eigen::Vector3d::UnitZ());
            const Eigen::AngleAxisd ap(pitch, Eigen::Vector3d::UnitY());
            const Eigen::AngleAxisd ar(roll, Eigen::Vector3d::UnitX());
            const Eigen::Matrix3d R = Rci * (ay * ap * ar).toRotationMatrix();

            cv::Mat rvec, R_cv;
            cv::eigen2cv(R, R_cv);
            cv::Rodrigues(R_cv, rvec);

            const cv::Mat tvec = (cv::Mat_<double>(3, 1) << t.x(), t.y(), t.z());

            std::vector<cv::Point2f> pts_2d;
            pts_2d.reserve(object_points.size());
            cv::projectPoints(
                object_points,
                rvec,
                tvec,
                camera_info_.first,
                camera_info_.second,
                pts_2d
            );

            std::vector<Eigen::Vector2d> image_points;
            image_points.reserve(pts_2d.size());

            for (const auto& p: pts_2d) {
                image_points.emplace_back(p.x, p.y);
            }

            return image_points;
        }

        double reprojectionErrorYaw(
            double yaw,
            const std::vector<cv::Point3f>& object_points,
            const std::vector<cv::Point2f>& landmarks,
            const std::vector<std::pair<int, int>>& sym_pairs,
            const Eigen::Matrix3d& Rci,
            double pitch,
            double roll,
            const Eigen::Vector3d& t
        ) const noexcept {
            const auto image_points =
                reprojectionArmor(yaw, object_points, landmarks, Rci, pitch, roll, t);
            double cost = 0.0;

            // for (size_t i = 0; i < image_points.size(); ++i) {
            //     Eigen::Vector2d obs(landmarks[i].x, landmarks[i].y);
            //     cost += (image_points[i] - obs).squaredNorm();
            // }

            for (auto& p: sym_pairs) {
                const Eigen::Vector2d mid = 0.5 * (image_points[p.first] + image_points[p.second]);

                const Eigen::Vector2d meas = 0.5
                    * (Eigen::Vector2d(landmarks[p.first].x, landmarks[p.first].y)
                       + Eigen::Vector2d(landmarks[p.second].x, landmarks[p.second].y));

                cost += (mid - meas).squaredNorm();
            }
            return cost;
        }

        double goldenYaw(
            double init,
            const std::vector<cv::Point3f>& obj,
            const std::vector<cv::Point2f>& lm,
            const std::vector<std::pair<int, int>>& sym_pairs,
            const Eigen::Matrix3d& Rci,
            double pitch,
            double roll,
            const Eigen::Vector3d& t
        ) const noexcept {
            constexpr double phi = 1.618033988749894848; //(1.0 + std::sqrt(5.0)) * 0.5;
            double l = init - params_.golden_search_side_deg * M_PI / 180.0;
            double r = init + params_.golden_search_side_deg * M_PI / 180.0;

            double y1 = r - (r - l) / phi;
            double y2 = l + (r - l) / phi;

            double f1 = reprojectionErrorYaw(y1, obj, lm, sym_pairs, Rci, pitch, roll, t);
            double f2 = reprojectionErrorYaw(y2, obj, lm, sym_pairs, Rci, pitch, roll, t);

            while (r - l > 0.0001) {
                if (f1 < f2) {
                    r = y2;
                    y2 = y1;
                    f2 = f1;
                    y1 = r - (r - l) / phi;
                    f1 = reprojectionErrorYaw(y1, obj, lm, sym_pairs, Rci, pitch, roll, t);
                } else {
                    l = y1;
                    y1 = y2;
                    f1 = f2;
                    y2 = l + (r - l) / phi;
                    f2 = reprojectionErrorYaw(y2, obj, lm, sym_pairs, Rci, pitch, roll, t);
                }
            }

            return 0.5 * (l + r);
        }

        Eigen::Matrix3d solveBa_R(
            const ArmorObject& armor,
            const Eigen::Vector3d& t_camera_armor,
            const Eigen::Matrix3d& R_camera_armor,
            const Eigen::Matrix3d& R_imu_camera,
            const std::string& type
        ) const noexcept {
            const Eigen::Matrix3d R_imu_armor = R_imu_camera * R_camera_armor;
            const Eigen::Matrix3d R_camera_imu = R_imu_camera.transpose();
            //double roll = std::atan2(R_imu_armor(2, 1), R_imu_armor(2, 2));
            const double roll = 0;
            // initial yaw
            const double yaw_init = std::atan2(-R_imu_armor(0, 1), R_imu_armor(1, 1));

            const double armor_pitch =
                (armor.number == ArmorNumber::OUTPOST) ? -FIFTTEN_DEGREE_RAD : FIFTTEN_DEGREE_RAD;

            const Eigen::Vector2d armor_size = (type == "large")
                ? Eigen::Vector2d { LARGE_ARMOR_WIDTH, LARGE_ARMOR_HEIGHT }
                : Eigen::Vector2d { SMALL_ARMOR_WIDTH, SMALL_ARMOR_HEIGHT };

            const auto objPts =
                ArmorObject::buildObjectPoints<cv::Point3f>(armor_size.x(), armor_size.y());
            const auto& lm = armor.landmarks();
            const auto& sym_pairs = ArmorObject::buildSymPairs<int>();
            double yaw = yaw_init;
            if (params_.opt_mode == Params::OptMode::GOLDEN) {
                yaw = goldenYaw(
                    yaw_init,
                    objPts,
                    lm,
                    sym_pairs,
                    R_camera_imu,
                    armor_pitch,
                    roll,
                    t_camera_armor
                );
            }

            // build yaw + pitch rotation
            const double cy = std::cos(yaw), sy = std::sin(yaw);
            Eigen::Matrix3d R_yaw;
            R_yaw << cy, -sy, 0, sy, cy, 0, 0, 0, 1;

            const double cp = std::cos(armor_pitch), sp = std::sin(armor_pitch);
            Eigen::Matrix3d R_pitch;
            R_pitch << cp, 0, sp, 0, 1, 0, -sp, 0, cp;

            const double cr = std::cos(roll), sr = std::sin(roll);
            Eigen::Matrix3d R_roll;
            R_roll << cr, -sr, 0, sr, cr, 0, 0, 0, 1;

            const Eigen::Matrix3d R_result = R_camera_imu * R_yaw * R_pitch * R_roll;
            return R_result;
        }

    private:
        std::pair<cv::Mat, cv::Mat> camera_info_;
        std::unique_ptr<wust_vl::algorithm::PnPSolver> pnp_solver_;
    };
    ArmorWhere::ArmorWhere(
        const YAML::Node& config,
        const std::pair<cv::Mat, cv::Mat>& camera_info
    ) {
        _impl = std::make_unique<Impl>(config, camera_info);
    }
    ArmorWhere::~ArmorWhere() {
        _impl.reset();
    }
    std::vector<Armor> ArmorWhere::where(
        const std::vector<ArmorObject>& armors,
        Eigen::Matrix4d T_camera_to_odom
    ) const noexcept {
        return _impl->where(armors, T_camera_to_odom);
    }
} // namespace auto_aim
} // namespace wust_vision