#include "type/type.hpp"
#include <fstream>
#include <iomanip> // for std::setprecision

double R_x = 0.0;
double R_y = 0.0;
double R_z = 0.0;
double R_yaw = 0.0;
double time_ = 0.0;

double s2qx = 0.0;
double s2qy = 0.0;
double s2qz = 0.0;
double s2qyaw = 0.0;

double s2qx_min = 0.1;
double s2qx_max = 100.0;
double s2qy_min = 0.1;
double s2qy_max = 100.0;
double s2qz_min = 0.1;
double s2qz_max = 100.0;
double s2qyaw_min = 0.1;
double s2qyaw_max = 100.0;
double last_x_ = 0.0, last_y_ = 0.0, last_z_ = 0.0, last_yaw_ = 0.0;
std::vector<armor::Armors> datas;
std::chrono::steady_clock::time_point last_time_ = std::chrono::steady_clock::now();

double orientationToYaw(const tf::Quaternion& orientation) {
    tf::Quaternion q(orientation.x, orientation.y, orientation.z, orientation.w);
    tf::Matrix3x3 m(q);
    double yaw, pitch, roll;
    m.getRPY(roll, pitch, yaw);
    return yaw;
}

void ex(double& a, double& min, double& max) {
    if (a < min) {
        min = a;
    }
    if (a > max) {
        max = a;
    }
}
void commandCallbacka(const armor::Armors& armors) {
    std::ofstream log_file("/tmp/calculation.txt", std::ios::trunc);
    log_file << "已经收集了" << datas.size() << "个数据" << std::endl;
    datas.push_back(armors);

    auto current_time = std::chrono::steady_clock::now();
    double delta_time =
        std::chrono::duration_cast<std::chrono::milliseconds>(current_time - last_time_).count()
        / 1000.0;

    if (!armors.armors.empty() && delta_time > 0.5) {
        last_time_ = current_time;

        const auto& pos = armors.armors[0].target_pos;
        double yaw = orientationToYaw(armors.armors[0].target_ori);

        if (datas.size() == 1) {
            last_x_ = pos.x;
            last_y_ = pos.y;
            last_z_ = pos.z;
            last_yaw_ = yaw;
        } else {
            double v_x = (pos.x - last_x_) / delta_time;
            double v_y = (pos.y - last_y_) / delta_time;
            double v_z = (pos.z - last_z_) / delta_time;
            double v_yaw = (yaw - last_yaw_) / delta_time;

            s2qx = std::exp(-(std::abs(v_x) + 0.5 * std::abs(v_yaw))) * (s2qx_max - s2qx_min)
                + s2qx_min;
            ex(s2qx, s2qx_min, s2qx_max);

            s2qy = std::exp(-(std::abs(v_y) + 0.5 * std::abs(v_yaw))) * (s2qy_max - s2qy_min)
                + s2qy_min;
            ex(s2qy, s2qy_min, s2qy_max);

            s2qz = std::exp(-(std::abs(v_z) + 0.5 * std::abs(v_yaw))) * (s2qz_max - s2qz_min)
                + s2qz_min;
            ex(s2qz, s2qz_min, s2qz_max);

            s2qyaw = std::exp(-(std::abs(v_x) + 0.5 * std::abs(v_z))) * (s2qyaw_max - s2qyaw_min)
                + s2qyaw_min;
            ex(s2qyaw, s2qyaw_min, s2qyaw_max);

            last_x_ = pos.x;
            last_y_ = pos.y;
            last_z_ = pos.z;
            last_yaw_ = yaw;
        }
    }

    if (datas.size() == 5000) {
        double all_yaw = 0.0, all_pitch = 0.0, all_dist = 0.0, all_ori_yaw = 0.0;
        time_ = 0;

        for (const auto& data: datas) {
            if (!data.armors.empty()) {
                time_++;
                const auto& armor = data.armors[0];
                double yaw = orientationToYaw(armor.target_ori);

                double armor_x = armor.target_pos.x;
                double armor_y = armor.target_pos.y;
                double armor_z = armor.target_pos.z;

                double xy_dist = std::sqrt(armor_x * armor_x + armor_y * armor_y);
                double dist = std::sqrt(xy_dist * xy_dist + armor_z * armor_z);
                double pitch = std::atan2(armor_z, xy_dist);

                all_yaw += yaw;
                all_pitch += pitch;
                all_dist += dist;
                all_ori_yaw += yaw;
            }
        }

        double mean_yaw = all_yaw / time_;
        double mean_pitch = all_pitch / time_;
        double mean_dist = all_dist / time_;
        double mean_ori_yaw = all_ori_yaw / time_;

        double var_yaw = 0.0, var_pitch = 0.0, var_dist = 0.0, var_ori_yaw = 0.0;

        for (const auto& data: datas) {
            if (!data.armors.empty()) {
                const auto& armor = data.armors[0];
                double yaw = orientationToYaw(armor.target_ori);

                double armor_x = armor.target_pos.x;
                double armor_y = armor.target_pos.y;
                double armor_z = armor.target_pos.z;

                double xy_dist = std::sqrt(armor_x * armor_x + armor_y * armor_y);
                double dist = std::sqrt(xy_dist * xy_dist + armor_z * armor_z);
                double pitch = std::atan2(armor_z, xy_dist);

                var_yaw += std::pow(yaw - mean_yaw, 2);
                var_pitch += std::pow(pitch - mean_pitch, 2);
                var_dist += std::pow(dist - mean_dist, 2);
                var_ori_yaw += std::pow(yaw - mean_ori_yaw, 2);
            }
        }

        R_x = var_yaw / time_;
        R_y = var_pitch / time_;
        R_z = var_dist / time_;
        R_yaw = var_ori_yaw / time_;
    }

    if (datas.size() > 5000) {
        std::ofstream log_file("/tmp/calculation.txt", std::ios::app);
        log_file << std::fixed << std::setprecision(10);
        log_file << "R_yaw: " << R_x << std::endl;
        log_file << "R_pitch: " << R_y << std::endl;
        log_file << "R_dist: " << R_z << std::endl;
        log_file << "R_ori_yaw: " << R_yaw << std::endl;

        log_file << "s2qx: " << s2qx << std::endl;
        log_file << "s2qy: " << s2qy << std::endl;
        log_file << "s2qz: " << s2qz << std::endl;
        log_file << "s2qyaw: " << s2qyaw << std::endl;
        log_file.close();
    }
}

void commandCallbackYpd(const armor::Armors& armors) {
    // 第一次写入（覆盖）
    if (datas.empty()) {
        std::ofstream log_file("/tmp/calculation.txt", std::ios::trunc);
        log_file << "开始收集数据..." << std::endl;
    }

    datas.push_back(armors);

    auto current_time = std::chrono::steady_clock::now();
    double delta_time =
        std::chrono::duration_cast<std::chrono::milliseconds>(current_time - last_time_).count()
        / 1000.0;

    if (!armors.armors.empty() && delta_time > 0.5) {
        last_time_ = current_time;

        const auto& pos = armors.armors[0].target_pos;
        double yaw = orientationToYaw(armors.armors[0].target_ori);

        if (datas.size() == 1) {
            last_x_ = pos.x;
            last_y_ = pos.y;
            last_z_ = pos.z;
            last_yaw_ = yaw;
        } else {
            double v_x = (pos.x - last_x_) / delta_time;
            double v_y = (pos.y - last_y_) / delta_time;
            double v_z = (pos.z - last_z_) / delta_time;
            double v_yaw = (yaw - last_yaw_) / delta_time;

            // 对应 EKF 的过程噪声估计
            s2qx = std::exp(-(std::abs(v_x) + 0.5 * std::abs(v_yaw))) * (s2qx_max - s2qx_min)
                + s2qx_min;
            ex(s2qx, s2qx_min, s2qx_max);

            s2qy = std::exp(-(std::abs(v_y) + 0.5 * std::abs(v_yaw))) * (s2qy_max - s2qy_min)
                + s2qy_min;
            ex(s2qy, s2qy_min, s2qy_max);

            s2qz = std::exp(-(std::abs(v_z) + 0.5 * std::abs(v_yaw))) * (s2qz_max - s2qz_min)
                + s2qz_min;
            ex(s2qz, s2qz_min, s2qz_max);

            s2qyaw = std::exp(-(std::abs(v_x) + 0.5 * std::abs(v_z))) * (s2qyaw_max - s2qyaw_min)
                + s2qyaw_min;
            ex(s2qyaw, s2qyaw_min, s2qyaw_max);

            last_x_ = pos.x;
            last_y_ = pos.y;
            last_z_ = pos.z;
            last_yaw_ = yaw;
        }
    }

    if (datas.size() == 5000) {
        double all_yaw = 0.0, all_pitch = 0.0, all_dist = 0.0, all_ori_yaw = 0.0;
        time_ = 0;

        for (const auto& data: datas) {
            if (!data.armors.empty()) {
                time_++;
                const auto& armor = data.armors[0];
                double ori_yaw = orientationToYaw(armor.target_ori);

                double x = armor.target_pos.x;
                double y = armor.target_pos.y;
                double z = armor.target_pos.z;

                double xy_dist = std::sqrt(x * x + y * y);
                double dist = std::sqrt(xy_dist * xy_dist + z * z);
                double pitch = std::atan2(z, xy_dist);
                double yaw = std::atan2(y, x);

                all_yaw += yaw;
                all_pitch += pitch;
                all_dist += dist;
                all_ori_yaw += ori_yaw;
            }
        }

        double mean_yaw = all_yaw / time_;
        double mean_pitch = all_pitch / time_;
        double mean_dist = all_dist / time_;
        double mean_ori_yaw = all_ori_yaw / time_;

        double var_yaw = 0.0, var_pitch = 0.0, var_dist = 0.0, var_ori_yaw = 0.0;

        for (const auto& data: datas) {
            if (!data.armors.empty()) {
                const auto& armor = data.armors[0];
                double ori_yaw = orientationToYaw(armor.target_ori);

                double x = armor.target_pos.x;
                double y = armor.target_pos.y;
                double z = armor.target_pos.z;

                double xy_dist = std::sqrt(x * x + y * y);
                double dist = std::sqrt(xy_dist * xy_dist + z * z);
                double pitch = std::atan2(z, xy_dist);
                double yaw = std::atan2(y, x);

                var_yaw += std::pow(yaw - mean_yaw, 2);
                var_pitch += std::pow(pitch - mean_pitch, 2);
                var_dist += std::pow(dist - mean_dist, 2);
                var_ori_yaw += std::pow(ori_yaw - mean_ori_yaw, 2);
            }
        }

        R_x = var_yaw / time_;
        R_y = var_pitch / time_;
        R_z = var_dist / time_;
        R_yaw = var_ori_yaw / time_;
    }

    if (datas.size() > 5000) {
        std::ofstream log_file("/tmp/calculation.txt", std::ios::app);
        log_file << std::fixed << std::setprecision(10);
        log_file << "R_yaw: " << R_x << std::endl;
        log_file << "R_pitch: " << R_y << std::endl;
        log_file << "R_dist: " << R_z << std::endl;
        log_file << "R_ori_yaw: " << R_yaw << std::endl;

        log_file << "s2qx: " << s2qx << std::endl;
        log_file << "s2qy: " << s2qy << std::endl;
        log_file << "s2qz: " << s2qz << std::endl;
        log_file << "s2qyaw: " << s2qyaw << std::endl;
        log_file.close();
    }
}
