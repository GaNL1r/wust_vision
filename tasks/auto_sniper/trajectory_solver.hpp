#pragma once
#include "tasks/type_common.hpp"
#include <wust_vl/common/utils/manual_compensator.hpp>
#include <wust_vl/common/utils/trajectory_compensator.hpp>
#include <yaml-cpp/yaml.h>
namespace wust_vision {
namespace auto_sniper {

    class TrajectorySolver {
    public:
        TrajectorySolver(const YAML::Node& config) {
            std::string comp_type =
                config["trajectory_compensator"]["compenstator_type"].as<std::string>("ideal");
            double gravity_ = config["trajectory_compensator"]["gravity"].as<double>(10.0);
            double resistance_ = config["trajectory_compensator"]["resistance"].as<double>(0.092);
            int iteration_times_ = config["trajectory_compensator"]["iteration_times"].as<int>(20);
            auto trajectory_compensator =
                wust_vl::common::utils::CompensatorFactory::createCompensator(comp_type);
            trajectory_compensator->iteration_times_ = iteration_times_;
            trajectory_compensator->gravity_ = gravity_;
            trajectory_compensator->resistance_ = resistance_;

            manual_compensator_ = std::make_unique<wust_vl::common::utils::ManualCompensator>();
            std::vector<wust_vl::common::utils::OffsetEntry> entries;
            if (config["trajectory_offset"]) {
                for (const auto& node: config["trajectory_offset"]) {
                    wust_vl::common::utils::OffsetEntry e;
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
            target_position_ = Eigen::Vector3d(
                config["target_position"]["x"].as<double>(0.0),
                config["target_position"]["y"].as<double>(0.0),
                config["target_position"]["z"].as<double>(0.0)
            );
        }

        GimbalCmd solve(const Eigen::Vector3d& self_position, double bullet_speed) {
            Eigen::Vector3d relative_pos = target_position_ - self_position;
            auto offs = manual_compensator_->angleHardCorrect(
                relative_pos.head(2).norm(),
                relative_pos.z()
            );
            double raw_yaw = std::atan2(relative_pos.y(), relative_pos.x());
            double raw_pitch = std::atan2(relative_pos.z(), relative_pos.head(2).norm());
            trajectory_compensator_->compensate(relative_pos, raw_pitch, bullet_speed);
            GimbalCmd cmd;
            cmd.yaw = angles::normalize_angle((raw_yaw + offs[1]) * M_PI / 180.0) * 180.0 / M_PI;
            cmd.pitch = raw_pitch + offs[0];
            double angle = raw_yaw;
            Eigen::Vector4d target =
                Eigen::Vector4d(relative_pos.x(), relative_pos.y(), relative_pos.z(), angle);
            auto check = checkHit(
                target,
                cmd.yaw * M_PI / 180.0,
                cmd.pitch * M_PI / 180.0,
                bullet_speed,
                true,
                0
            );
            cmd.enable_yaw_diff = std::get<1>(check);
            cmd.enable_pitch_diff = std::get<2>(check);
            cmd.target_yaw = std::get<3>(check);
            cmd.target_pitch = std::get<4>(check);
            cmd.fire_advice = std::get<0>(check);
            return cmd;
        }
        std::tuple<bool, double, double, double, double> checkHit(
            Eigen::Vector4d maybe_hit,
            const double current_yaw,
            const double current_pitch,
            const double bullet_speed,
            bool use_off_fire,
            double gun_yaw_speed = 0.0
        ) {
            Eigen::Vector3d target_pos = maybe_hit.head<3>();
            double dx = target_pos.x();
            double dy = target_pos.y();
            double dz = target_pos.z();

            double distance = target_pos.norm();
            double shooting_range_yaw = std::abs(atan2(shooting_range_w_ / 2, distance));
            double shooting_range_pitch = std::abs(atan2(shooting_range_h_ / 2, distance));
            double yaw_factor = 1.0 * std::cos(maybe_hit[3]);
            double pitch_factor = 1.0;

            shooting_range_yaw *= yaw_factor;
            shooting_range_pitch *= pitch_factor;
            shooting_range_yaw = std::max(shooting_range_yaw, 0.5 * M_PI / 180);
            shooting_range_pitch = std::max(shooting_range_pitch, 0.5 * M_PI / 180);

            double target_yaw = angles::normalize_angle(std::atan2(dy, dx));
            double tmp_pitch = std::atan2(dz, std::sqrt(dx * dx + dy * dy));
            double target_pitch = tmp_pitch;

            if (!trajectory_compensator_->compensate(target_pos, target_pitch, bullet_speed))
                return { false,
                         shooting_range_yaw / M_PI * 180.0,
                         shooting_range_pitch / M_PI * 180.0,
                         target_yaw / M_PI * 180.0,
                         target_pitch / M_PI * 180.0 };

            if (use_off_fire) {
                auto offs = manual_compensator_->angleHardCorrect(
                    target_pos.head<2>().norm(),
                    target_pos.z()
                );
                target_yaw += offs[1] * M_PI / 180.0;
                target_pitch += offs[0] * M_PI / 180.0;
            }

            double t_f = distance / bullet_speed;
            target_yaw -= gun_yaw_speed * t_f;

            double yaw_diff = std::abs(angles::shortest_angular_distance(current_yaw, target_yaw));
            double pitch_diff =
                std::abs(angles::shortest_angular_distance(current_pitch, target_pitch));

            if (yaw_diff < shooting_range_yaw && pitch_diff < shooting_range_pitch) {
                return { true,
                         std::abs(shooting_range_yaw) / M_PI * 180.0,
                         std::abs(shooting_range_pitch) / M_PI * 180.0,
                         target_yaw / M_PI * 180.0,
                         target_pitch / M_PI * 180.0 };
            }

            return { false,
                     std::abs(shooting_range_yaw) / M_PI * 180.0,
                     std::abs(shooting_range_pitch) / M_PI * 180.0,
                     target_yaw / M_PI * 180.0,
                     target_pitch / M_PI * 180.0 };
        }

        Eigen::Vector3d target_position_;
        std::shared_ptr<wust_vl::common::utils::TrajectoryCompensator> trajectory_compensator_;
        std::unique_ptr<wust_vl::common::utils::ManualCompensator> manual_compensator_;
        double shooting_range_w_ = 0.135;
        double shooting_range_h_ = 0.135;
    };
} // namespace auto_sinper
} // namespace wust_vision