#include "type_common.hpp"

namespace wust_vision {
std::string enemyColorToString(EnemyColor color) noexcept {
    switch (color) {
        case EnemyColor::RED:
            return "RED";
            break;
        case EnemyColor::BLUE:
            return "BLUE";
            break;
        case EnemyColor::WHITE:
            return "WHITE";
            break;
        default:
            return "UNKNOWN";
    }
}
AttackMode toAttackMode(int value) noexcept {
    switch (value) {
        case 0:
            return AttackMode::ARMOR;
        case 1:
            return AttackMode::SMALL_RUNE;
        case 2:
            return AttackMode::BIG_RUNE;
        default:
            return AttackMode::UNKNOWN;
    }
}

void AimTarget::predictSelf(double dt_sec) noexcept {
    if (!have_host)
        return;

    Eigen::Vector3d rel_pos = pos - host_pos;

    double theta = host_v_yaw * dt_sec;

    Eigen::Matrix3d R;
    R = Eigen::AngleAxisd(theta, Eigen::Vector3d::UnitZ());

    pos = host_pos + R * rel_pos + host_vel * dt_sec;
}
void AimTarget::tf(Eigen::Matrix4d T_camera_to_odom) noexcept {
    Eigen::Vector4d pos_camera(pos.x(), pos.y(), pos.z(), 1.0);
    Eigen::Vector4d pos_odom = T_camera_to_odom * pos_camera;

    pos.x() = pos_odom.x();
    pos.y() = pos_odom.y();
    pos.z() = pos_odom.z();
    Eigen::Matrix3d R_camera_to_odom = T_camera_to_odom.block<3, 3>(0, 0);
    Eigen::Quaterniond q_camera(ori.w(), ori.x(), ori.y(), ori.z());
    Eigen::Matrix3d R_ori_camera = q_camera.normalized().toRotationMatrix();

    Eigen::Matrix3d R_ori_odom = R_camera_to_odom * R_ori_camera;
    Eigen::Quaterniond q_odom(R_ori_odom);

    ori.w() = q_odom.w();
    ori.x() = q_odom.x();
    ori.y() = q_odom.y();
    ori.z() = q_odom.z();
}
std::vector<cv::Point2f>
AimTarget::toPts(const cv::Mat& camera_intrinsic, const cv::Mat& camera_distortion) noexcept {
    std::vector<cv::Point2f> pts;
    if (pos.norm() < 0.5) {
        return pts;
    }

    cv::Mat tvec = (cv::Mat_<double>(3, 1) << pos.x(), pos.y(), pos.z());
    Eigen::Matrix3d tf_rot = ori.toRotationMatrix();
    cv::Mat rot_mat =
        (cv::Mat_<double>(3, 3) << tf_rot(0, 0),
         tf_rot(0, 1),
         tf_rot(0, 2),
         tf_rot(1, 0),
         tf_rot(1, 1),
         tf_rot(1, 2),
         tf_rot(2, 0),
         tf_rot(2, 1),
         tf_rot(2, 2));

    // 旋转矩阵 -> 旋转向量
    cv::Mat rvec;
    cv::Rodrigues(rot_mat, rvec);

    cv::projectPoints(AIM_TARGET_BLOCK, rvec, tvec, camera_intrinsic, camera_distortion, pts);

    return pts;
}
} // namespace wust_vision