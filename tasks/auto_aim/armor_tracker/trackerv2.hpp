#pragma once
#include "tasks/auto_aim/armor_tracker/motion_models/motion_modelypdv2.hpp"
#include "tasks/auto_aim/type.hpp"
#include "tasks/utils.hpp"

class TrackerV2 {
public:
    void track(const armor::Armors& armors_msg);
    enum State {
        LOST,
        DETECTING,
        TRACKING,
        TEMP_LOST,
    } tracker_state = LOST;
    bool initTarget(const armor::Armors& armors);
    bool updateTarget(const armor::Armors& armors);
    void updateekf(const armor::Armor& armor, const Eigen::VectorXd& ekf_prediction);
    std::vector<Eigen::Vector4d> getArmorPosAndYaw(const Eigen::VectorXd& ekf_prediction);
    Eigen::Vector3d h_armor_xyz(const Eigen::VectorXd& x, int id);
    std::unique_ptr<ypdv2armor_motion_model::RobotStateEKF> ekf_ypd_;
    int tracking_thres_;
    int lost_thres_;

    armor::ArmorNumber tracked_id_;
    std::string type_;
    Eigen::VectorXd measurement_ = Eigen::VectorXd::Zero(4);
    Eigen::VectorXd target_state_ = Eigen::VectorXd::Zero(11);
    double last_yaw_ = 0;
    double last_ypd_y = 0;
    int detect_count_ = 0;
    int lost_count_ = 0;
    int armor_num_ = 4;
    int switch_count_ = 0;
    int update_count_ = 0;

    bool is_switch_, is_converged_;
    bool jumped;
    int last_id; // debug only
    double orientationToYaw(const Eigen::Quaterniond& q) noexcept {
        double roll, pitch, yaw;
        Eigen::Vector3d euler = utils::quatToEuler(q, utils::EulerOrder::ZYX, false);
        yaw = euler[0];
        yaw = this->last_yaw_ + angles::shortest_angular_distance(this->last_yaw_, yaw);
        this->last_yaw_ = yaw;
        return yaw;
    }
};