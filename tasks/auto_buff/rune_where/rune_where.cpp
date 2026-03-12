#include "rune_where.hpp"
#include "tasks/utils/utils.hpp"
#include <Eigen/Dense>
#include <opencv2/core/eigen.hpp>

namespace wust_vision {
namespace auto_buff {
    struct RuneWhere::Impl {
    public:
        Impl(const YAML::Node& config, const std::pair<cv::Mat, cv::Mat>& camera_info) {
            camera_info_ = camera_info;
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

            int golden_search_side_deg = 60;
            void load(const YAML::Node& node) {
                opt_mode = fromString(node["roll_opt"]["mode"].as<std::string>());
                golden_search_side_deg = node["roll_opt"]["golden_search_side_deg"].as<int>();
            }
        } params_;
        auto_buff::RuneFan
        where(auto_buff::RuneFan f, Eigen::Matrix4d T_camera_to_odom) const noexcept {
            const Eigen::Matrix3d R_imu_cam = T_camera_to_odom.block<3, 3>(0, 0);
            for (auto& fan: f.fans) {
                cv::Mat rvec, tvec;
                cv::solvePnP(
                    fan.getObjs(),
                    fan.landmarks(),
                    camera_info_.first,
                    camera_info_.second,
                    rvec,
                    tvec,
                    false,
                    cv::SOLVEPNP_IPPE //平移更稳定，（旋转这里纯靠后面优化）
                );
                cv::Mat R_cv;
                cv::Rodrigues(rvec, R_cv);
                Eigen::Matrix3d R = utils::cvToEigen(R_cv);
                Eigen::Vector3d t = utils::cvToEigen(tvec);
                if (params_.opt_mode != Params::OptMode::NONE) {
                    R = solveBa_R(fan, t, R, R_imu_cam);
                }

                fan.ori = Eigen::Quaterniond(R);
                fan.pos = t;
                Eigen::Vector3d pos_camera = fan.pos;
                fan.target_pos = utils::transformPosition(pos_camera, T_camera_to_odom);

                const Eigen::Quaterniond
                    q_camera(fan.ori.w(), fan.ori.x(), fan.ori.y(), fan.ori.z());
                const Eigen::Quaterniond q_odom =
                    utils::transformOrientation(q_camera, T_camera_to_odom);
                fan.target_ori = q_odom;

                f.is_valid = true;
            }
            return f;
        }
        std::vector<Eigen::Vector2d> reprojection(
            double roll,
            const std::vector<cv::Point3f>& object_points,
            const std::vector<cv::Point2f>& landmarks,
            const Eigen::Matrix3d& Rci,
            double pitch,
            double yaw,
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
        double reprojectionErrorRoll(
            double roll,
            const std::vector<cv::Point3f>& obj,
            const std::vector<cv::Point2f>& lm,
            const Eigen::Matrix3d& Rci,
            double pitch,
            double yaw,
            const Eigen::Vector3d& t
        ) const noexcept {
            const auto image_points = reprojection(roll, obj, lm, Rci, pitch, yaw, t);
            double cost = 0.0;

            for (size_t i = 0; i < image_points.size(); ++i) {
                Eigen::Vector2d obs(lm[i].x, lm[i].y);
                cost += (image_points[i] - obs).squaredNorm();
            }
            return cost;
        }

        double goldenRoll(
            double init,
            const std::vector<cv::Point3f>& obj,
            const std::vector<cv::Point2f>& lm,
            const Eigen::Matrix3d& Rci,
            double pitch,
            double yaw,
            const Eigen::Vector3d& t
        ) const noexcept {
            constexpr double phi = 1.618033988749894848; // golden ratio
            double l = init - params_.golden_search_side_deg * M_PI / 180.0;
            double r = init + params_.golden_search_side_deg * M_PI / 180.0;

            double r1 = r - (r - l) / phi;
            double r2 = l + (r - l) / phi;

            double f1 = reprojectionErrorRoll(r1, obj, lm, Rci, pitch, yaw, t);
            double f2 = reprojectionErrorRoll(r2, obj, lm, Rci, pitch, yaw, t);

            while (r - l > 0.0001) { // 约 0.0057 度
                if (f1 < f2) {
                    r = r2;
                    r2 = r1;
                    f2 = f1;
                    r1 = r - (r - l) / phi;
                    f1 = reprojectionErrorRoll(r1, obj, lm, Rci, pitch, yaw, t);
                } else {
                    l = r1;
                    r1 = r2;
                    f1 = f2;
                    r2 = l + (r - l) / phi;
                    f2 = reprojectionErrorRoll(r2, obj, lm, Rci, pitch, yaw, t);
                }
            }

            return 0.5 * (l + r); // final best roll
        }

        Eigen::Matrix3d solveBa_R(
            const auto_buff::RuneFan::Simple& rune_fan,
            const Eigen::Vector3d& t_camera_armor,
            const Eigen::Matrix3d& R_camera_armor,
            const Eigen::Matrix3d& R_imu_camera
        ) const noexcept {
            Eigen::Matrix3d R_imu_armor = R_imu_camera * R_camera_armor;
            Eigen::Matrix3d R_camera_imu = R_imu_camera.transpose();
            double initial_roll = std::atan2(R_imu_armor(2, 1), R_imu_armor(2, 2));
            double roll = initial_roll;
            Eigen::Vector3d t_imu_armor = R_imu_camera * t_camera_armor;
            double yaw = std::atan2(t_imu_armor.y(), t_imu_armor.x());
            double pitch = 0;
            auto cv_points = rune_fan.getObjs();
            const auto& landmarks = rune_fan.landmarks();
            if (params_.opt_mode == Params::OptMode::GOLDEN) {
                roll = goldenRoll(
                    roll,
                    cv_points,
                    landmarks,
                    R_camera_imu,
                    pitch,
                    yaw,
                    t_camera_armor
                );
            }
            const Eigen::AngleAxisd ay(yaw, Eigen::Vector3d::UnitZ());
            const Eigen::AngleAxisd ap(pitch, Eigen::Vector3d::UnitY());
            const Eigen::AngleAxisd ar(roll, Eigen::Vector3d::UnitX());

            const Eigen::Matrix3d R_result = R_camera_imu * (ay * ap * ar).toRotationMatrix();
            return R_result;
        }

        std::pair<cv::Mat, cv::Mat> camera_info_;
    };
    RuneWhere::RuneWhere(const YAML::Node& config, const std::pair<cv::Mat, cv::Mat>& camera_info) {
        _impl = std::make_unique<Impl>(config, camera_info);
    }
    RuneWhere::~RuneWhere() {
        _impl.reset();
    }
    auto_buff::RuneFan RuneWhere::where(auto_buff::RuneFan f, Eigen::Matrix4d T_camera_to_odom) {
        return _impl->where(f, T_camera_to_odom);
    }
} // namespace auto_buff
} // namespace wust_vision