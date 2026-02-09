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

#include "armor_pose_estimator.hpp"
#include "tasks/auto_aim/armor_optimize/ba_solver.hpp"
#include "tasks/utils.hpp"
#include "wust_vl/algorithm/pnp_solver.hpp"
#include "wust_vl/common/utils/logger.hpp"
namespace wust_vision {
namespace auto_aim {
    struct ArmorPoseEstimator::Impl {
    public:
        Impl(const YAML::Node& config, std::pair<cv::Mat, cv::Mat> camera_info) {
            pnp_solver_ = std::make_unique<wust_vl::algorithm::PnPSolver>(cv::SOLVEPNP_IPPE);
            pnp_solver_->setObjectPoints(
                "small",
                ArmorObject::buildObjectPoints<cv::Point3f>(SMALL_ARMOR_WIDTH, SMALL_ARMOR_HEIGHT)
            );
            pnp_solver_->setObjectPoints(
                "large",
                ArmorObject::buildObjectPoints<cv::Point3f>(LARGE_ARMOR_WIDTH, LARGE_ARMOR_HEIGHT)
            );
            if (config["armor_optimize"]["enable"].as<bool>()) {
                ba_solver_ = std::make_unique<BaSolver>(config["armor_optimize"], camera_info);
            }
            distance_fix_a2_ = config["armor_optimize"]["distance_fix_a2"].as<double>();
        }
        std::vector<Armor> extractArmorPoses(
            const std::vector<ArmorObject>& armors,
            Eigen::Matrix4d T_camera_to_odom,
            const cv::Mat& camera_intrinsic,
            const cv::Mat& camera_distortion
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
                    dist += distance_fix_a2_ * dist * dist;
                    auto [x, y, z] = utils::ypd2xyz_rad(yaw, pitch, dist);

                    msg.pos = { x, y, z };
                    msg.ori = new_q;
                    transformArmorData(msg, T_camera_to_odom);
                    msg.distance_to_image_center =
                        pnp_solver_->calculateDistanceToCenter(obj.center, camera_intrinsic);
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
                        camera_intrinsic,
                        camera_distortion
                    )) {
                    WUST_WARN("PNP") << "PNP failed";
                    continue;
                }
                cv::Mat R_cv;
                cv::Rodrigues(rvec, R_cv);
                Eigen::Matrix3d R = utils::cvToEigen(R_cv);
                Eigen::Vector3d t = utils::cvToEigen(tvec);
                if (ba_solver_) {
                    Eigen::Matrix3d R0 = R;
                    R = ba_solver_->solveBa_R(a, t, R0, R_imu_cam, type);
                }

                armors_msg.push_back(makeArmor(a, t, R));
            }

            return armors_msg;
        }
        std::unique_ptr<BaSolver> ba_solver_;
        double distance_fix_a2_ = 0;
        std::unique_ptr<wust_vl::algorithm::PnPSolver> pnp_solver_;
    };
    ArmorPoseEstimator::ArmorPoseEstimator(
        const YAML::Node& config,
        std::pair<cv::Mat, cv::Mat> camera_info
    ) {
        _impl = std::make_unique<Impl>(config, camera_info);
    }
    ArmorPoseEstimator::~ArmorPoseEstimator() {
        _impl.reset();
    }
    std::vector<Armor> ArmorPoseEstimator::extractArmorPoses(
        const std::vector<ArmorObject>& armors,
        Eigen::Matrix4d T_camera_to_odom,
        const cv::Mat& camera_intrinsic,
        const cv::Mat& camera_distortion
    ) const noexcept {
        return _impl
            ->extractArmorPoses(armors, T_camera_to_odom, camera_intrinsic, camera_distortion);
    }
} // namespace auto_aim
} // namespace wust_vision