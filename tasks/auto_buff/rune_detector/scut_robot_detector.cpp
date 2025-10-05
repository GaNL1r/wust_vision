#include "scut_robot_detector.hpp"
#include "vc/camera/camera_param.h"
ScutRobotDetector::ScutRobotDetector(
    const std::pair<cv::Mat, cv::Mat>& camera_info,
    const YAML::Node& config
):
    camera_info_(camera_info) {
    CV_Assert(
        camera_info.first.rows == 3 && camera_info.first.cols == 3
        && camera_info.first.channels() == 1
    );
    cv::Mat camMat32;
    camera_info.first.convertTo(camMat32, CV_32F);

    camera_param.cameraMatrix = cv::Matx33f(
        camMat32.at<float>(0, 0),
        camMat32.at<float>(0, 1),
        camMat32.at<float>(0, 2),
        camMat32.at<float>(1, 0),
        camMat32.at<float>(1, 1),
        camMat32.at<float>(1, 2),
        camMat32.at<float>(2, 0),
        camMat32.at<float>(2, 1),
        camMat32.at<float>(2, 2)
    );
    CV_Assert(camera_info.second.total() == 5 && camera_info.second.channels() == 1);

    cv::Mat dist32;
    camera_info.second.convertTo(dist32, CV_32F);

    camera_param.distCoeff = cv::Matx<float, 5, 1>(
        dist32.at<float>(0),
        dist32.at<float>(1),
        dist32.at<float>(2),
        dist32.at<float>(3),
        dist32.at<float>(4)
    );

    std::array<double, 9> camera_matrix;
    CV_Assert(camera_info.first.rows == 3 && camera_info.first.cols == 3);
    CV_Assert(camera_info.first.type() == CV_64F);

    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
            camera_matrix[i * 3 + j] = camera_info.first.at<double>(i, j);
    ba_solver_ = std::make_unique<rune::BaSolver>(
        camera_matrix,
        config["rune_optimize"]["max_iter_R"].as<int>(),
        config["rune_optimize"]["max_iter_t"].as<int>(),
        config["rune_optimize"]["step_R"].as<int>(),
        config["rune_optimize"]["step_t"].as<int>(),
        config["rune_optimize"]["min_error_R"].as<double>(),
        config["rune_optimize"]["min_error_t"].as<double>()
    );
}
Eigen::Quaterniond scutRobot2us(Eigen::Quaterniond scut) {
    Eigen::Matrix3d R_img2gimbal;
    R_img2gimbal << 0, 0, 1, -1, 0, 0, 0, -1, 0;

    Eigen::Quaterniond q_gimbal(R_img2gimbal * scut.toRotationMatrix());

    Eigen::Vector3d euler_gimbal =
        utils::matrixToEuler(q_gimbal.toRotationMatrix(), utils::EulerOrder::ZXY);

    Eigen::Vector3d us_euler;
    us_euler[0] = euler_gimbal[2] + M_PI / 2;
    us_euler[1] = euler_gimbal[1];
    us_euler[2] = euler_gimbal[0] + M_PI / 2;

    Eigen::Matrix3d R_gimbal_new = utils::eulerToMatrix(us_euler, utils::EulerOrder::ZXY);

    Eigen::Matrix3d R_us = R_img2gimbal.transpose() * R_gimbal_new;
    Eigen::Quaterniond q_us(R_us);
    return q_us;
}

rune::RuneFan ScutRobotDetector::detect(
    const CommonFrame& frame,
    Eigen::Vector3d gimbal,
    Eigen::Matrix4d T_camera_to_odom,
    bool debug,
    cv::Mat& debug_img
) {
    if (debug) {
        DebugTools::get()->setImage(frame.src_img);
    }
    GyroData gyroData;
    gyroData.rotation.yaw = gimbal[0];
    gyroData.rotation.pitch = gimbal[1];
    gyroData.rotation.roll = gimbal[2];
    DetectorInput input;
    DetectorOutput output;
    input.setImage(frame.src_img);
    input.setTick(frame.timestamp);
    input.setGyroData(GyroData());
    input.setColor(PixChannel::RED);
    input.setFeatureNodes(rune_groups_);
    input.setDebug(debug);
    rune::RuneFan fan { .is_valid = false, .timestamp = frame.timestamp, .id = frame.id };
    static auto rune_detector = RuneDetector::make_detector();

    rune_detector->detect(input, output);
    rune_groups_ = output.getFeatureNodes();

    do {
        if (rune_groups_.empty())
            break;
        auto rune_group = RuneGroup::cast(rune_groups_.front());

        if (rune_group->childFeatures().empty())
            break;
        FeatureNode_cptr target_tracker = nullptr;

        for (auto tracker: rune_group->getTrackers()) {
            auto tracker_ = TrackingFeatureNode::cast(tracker);

            if (tracker_->getHistoryNodes().size() < 2)
                continue;
            auto type = RuneCombo::cast(tracker_->getHistoryNodes().front())->getRuneType();
            if (type == RuneType::PENDING_STRUCK) {
                target_tracker = tracker;
                break;
            }
        }

        if (!target_tracker)
            break;

        // 获取位姿信息
        auto pose = output.getPnpData();
        auto tvec = pose.tvec();
        tvec = tvec * 0.001;
        auto rvec = pose.rvec();
        Eigen::Vector3d t;
        Eigen::Quaterniond q_scut;
        utils::pnpToEigen(rvec, tvec, t, q_scut);

        auto q = scutRobot2us(q_scut);
        auto R = q.toRotationMatrix();
        const Eigen::Matrix3d R_imu_cam = T_camera_to_odom.block<3, 3>(0, 0);
        q = Eigen::Quaterniond(R);
        auto pos_odom = utils::transformPosition(t, T_camera_to_odom);
        auto q_odom = utils::transformOrientation(q, T_camera_to_odom);

        fan.target_pos = pos_odom;
        fan.target_ori = q_odom;

        fan.is_valid = true;
    } while (0);
    if (debug) {
        debug_img = DebugTools::get()->getImage().clone();
        auto debug_bin = output.getDebugImg();
        if (debug_bin.empty()) {
            std::cout << "debug_bin.empty()" << std::endl;
        }
        if (!debug_img.empty() && !debug_bin.empty()) {
            cv::Mat color_bin;
            if (debug_bin.channels() == 1)
                cv::cvtColor(debug_bin, color_bin, cv::COLOR_GRAY2BGR);
            else
                color_bin = debug_bin;

            cv::Mat small_bin;
            cv::resize(color_bin, small_bin, cv::Size(), 0.25, 0.25);

            int x = debug_img.cols - small_bin.cols - 10;
            int y = debug_img.rows - small_bin.rows - 10;

            small_bin.copyTo(debug_img(cv::Rect(x, y, small_bin.cols, small_bin.rows)));
        }
    }
    return fan;
}