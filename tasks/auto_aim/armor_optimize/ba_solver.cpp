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

#include "ba_solver.hpp"
#include <ceres/autodiff_cost_function.h>
#include <ceres/local_parameterization.h>
#include <ceres/loss_function.h>
#include <ceres/problem.h>
#include <ceres/solver.h>
#include <opencv2/core/eigen.hpp>
namespace auto_aim {

struct BaSolver::Impl {
public:
    Impl(const YAML::Node& config, const std::pair<cv::Mat, cv::Mat>& camera_info) {
        camera_info_ = camera_info;
        cv::cv2eigen(camera_info.first, K_);
        dist_eigen_ = toEigenDist(camera_info.second);
        params_.load(config);
    }
    struct Params {
        enum class OptMode : int {
            GOLDEN = 0,
            CERES = 1

        } mode;
        OptMode fromString(const std::string& mode) {
            if (mode == "golden" || mode == "GOLDEN") {
                return OptMode::GOLDEN;
            } else if (mode == "ceres" || mode == "CERES") {
                return OptMode::CERES;
            } else {
                return OptMode::GOLDEN;
            }
        }
        int ceres_max_iter = 40;
        int golden_search_side_deg = 60;
        void load(const YAML::Node& node) {
            mode = fromString(node["mode"].as<std::string>());
            ceres_max_iter = node["ceres_max_iter"].as<int>();
            golden_search_side_deg = node["golden_search_side_deg"].as<int>();
        }
    } params_;
    std::vector<Eigen::Vector2d> reprojectionArmor(
        double yaw,
        const std::vector<Eigen::Vector3d>& object_points,
        const std::vector<cv::Point2f>& landmarks,
        const Eigen::Matrix3d& Rci,
        double pitch,
        double roll,
        const Eigen::Vector3d& t,
        const Eigen::Matrix3d& K,
        const cv::Mat& dist // 畸变参数
    ) const noexcept {
        // 组合旋转（相机系到目标系）
        const Eigen::AngleAxisd ay(yaw, Eigen::Vector3d::UnitZ());
        const Eigen::AngleAxisd ap(pitch, Eigen::Vector3d::UnitY());
        const Eigen::AngleAxisd ar(roll, Eigen::Vector3d::UnitX());
        const Eigen::Matrix3d R = Rci * (ay * ap * ar).toRotationMatrix();

        // 转成 OpenCV rvec/tvec
        cv::Mat rvec, R_cv;
        cv::eigen2cv(R, R_cv);
        cv::Rodrigues(R_cv, rvec);

        const cv::Mat tvec = (cv::Mat_<double>(3, 1) << t.x(), t.y(), t.z());

        // 取内参
        cv::Mat K_cv;
        cv::eigen2cv(K, K_cv);

        // 3D点转 cv::Point3f
        std::vector<cv::Point3f> obj_pts;
        obj_pts.reserve(object_points.size());
        for (const auto& p: object_points) {
            obj_pts.emplace_back(p.x(), p.y(), p.z());
        }

        // 投影（带畸变）
        std::vector<cv::Point2f> pts_2d;
        pts_2d.reserve(obj_pts.size());
        cv::projectPoints(obj_pts, rvec, tvec, K_cv, dist, pts_2d);

        // 输出到 Eigen，并做 NaN/越界/深度保护
        std::vector<Eigen::Vector2d> image_points;
        image_points.reserve(pts_2d.size());

        for (const auto& p: pts_2d) {
            if (!std::isfinite(p.x) || !std::isfinite(p.y)) {
                continue; // 过滤异常点
            }
            image_points.emplace_back(p.x, p.y);
        }

        return image_points;
    }

    double reprojectionErrorYaw(
        double yaw,
        const std::vector<Eigen::Vector3d>& object_points,
        const std::vector<cv::Point2f>& landmarks,
        const std::vector<std::pair<int, int>>& sym_pairs,
        const Eigen::Matrix3d& Rci,
        double pitch,
        double roll,
        const Eigen::Vector3d& t,
        const Eigen::Matrix3d& K,
        const cv::Mat& dist
    ) const noexcept {
        const auto image_points =
            reprojectionArmor(yaw, object_points, landmarks, Rci, pitch, roll, t, K, dist);
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
        const std::vector<Eigen::Vector3d>& obj,
        const std::vector<cv::Point2f>& lm,
        const std::vector<std::pair<int, int>>& sym_pairs,
        const Eigen::Matrix3d& Rci,
        double pitch,
        double roll,
        const Eigen::Vector3d& t,
        const Eigen::Matrix3d& K
    ) const noexcept {
        constexpr double phi = 1.618033988749894848; //(1.0 + std::sqrt(5.0)) * 0.5;
        double l = init - params_.golden_search_side_deg * M_PI / 180.0;
        double r = init + params_.golden_search_side_deg * M_PI / 180.0;

        double y1 = r - (r - l) / phi;
        double y2 = l + (r - l) / phi;

        double f1 = reprojectionErrorYaw(
            y1,
            obj,
            lm,
            sym_pairs,
            Rci,
            pitch,
            roll,
            t,
            K,
            camera_info_.second
        );
        double f2 = reprojectionErrorYaw(
            y2,
            obj,
            lm,
            sym_pairs,
            Rci,
            pitch,
            roll,
            t,
            K,
            camera_info_.second
        );

        while (r - l > 0.0001) {
            if (f1 < f2) {
                r = y2;
                y2 = y1;
                f2 = f1;
                y1 = r - (r - l) / phi;
                f1 = reprojectionErrorYaw(
                    y1,
                    obj,
                    lm,
                    sym_pairs,
                    Rci,
                    pitch,
                    roll,
                    t,
                    K,
                    camera_info_.second
                );
            } else {
                l = y1;
                y1 = y2;
                f1 = f2;
                y2 = l + (r - l) / phi;
                f2 = reprojectionErrorYaw(
                    y2,
                    obj,
                    lm,
                    sym_pairs,
                    Rci,
                    pitch,
                    roll,
                    t,
                    K,
                    camera_info_.second
                );
            }
        }

        return 0.5 * (l + r);
    }

    double ceresYaw(
        double initial_yaw,
        const std::vector<Eigen::Vector3d>& object_points,
        const std::vector<cv::Point2f>& landmarks,
        const std::vector<std::pair<int, int>>& sym_pairs,
        const Eigen::Matrix3d& R_camera_imu,
        double armor_pitch,
        double roll,
        const Eigen::Vector3d& t_camera_armor,
        const Eigen::Matrix3d& K
    ) const noexcept {
        double yaw = initial_yaw;

        const CameraProjector cam(R_camera_imu, armor_pitch, roll, t_camera_armor, K, dist_eigen_);

        ceres::Problem problem;
        problem.AddParameterBlock(&yaw, 1, new YawLocalParameterization());

        // for (size_t i = 0; i < object_points.size(); ++i) {
        //     ceres::CostFunction* cost =
        //         new ceres::AutoDiffCostFunction<ReprojectionError, 2, 1>(new ReprojectionError(
        //             Eigen::Vector2d(landmarks[i].x, landmarks[i].y),
        //             object_points[i],
        //             cam
        //         ));

        //     problem.AddResidualBlock(cost, new ceres::HuberLoss(1.0), &yaw);
        // }

        for (auto& p: sym_pairs) {
            Eigen::Vector2d meas = (Eigen::Vector2d(landmarks[p.first].x, landmarks[p.first].y)
                                    + Eigen::Vector2d(landmarks[p.second].x, landmarks[p.second].y))
                * 0.5;
            ceres::CostFunction* cost = new ceres::AutoDiffCostFunction<SymmetryError, 2, 1>(
                new SymmetryError(object_points[p.first], object_points[p.second], meas, cam)
            );
            problem.AddResidualBlock(cost, new ceres::HuberLoss(1.0), &yaw);
        }

        ceres::Solver::Options options;
        options.max_num_iterations = params_.ceres_max_iter;
        options.linear_solver_type = ceres::DENSE_QR;
        options.minimizer_progress_to_stdout = false;

        ceres::Solver::Summary summary;
        ceres::Solve(options, &problem, &summary);

        return yaw;
    }

    Eigen::Matrix3d solveBa_R(
        const ArmorObject& armor,
        const Eigen::Vector3d& t_camera_armor,
        const Eigen::Matrix3d& R_camera_armor,
        const Eigen::Matrix3d& R_imu_camera,
        const std::string& type
    ) const noexcept {
        const Eigen::Matrix3d K = K_;

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
            ArmorObject::buildObjectPoints<Eigen::Vector3d>(armor_size.x(), armor_size.y());
        const auto& lm = armor.landmarks();
        const auto& sym_pairs = ArmorObject::buildSymPairs<int>();
        double yaw = yaw_init;
        if (params_.mode == Params::OptMode::CERES) {
            yaw = ceresYaw(
                yaw_init,
                objPts,
                lm,
                sym_pairs,
                R_camera_imu,
                armor_pitch,
                roll,
                t_camera_armor,
                K
            );
        } else if (params_.mode == Params::OptMode::GOLDEN) {
            yaw = goldenYaw(
                yaw_init,
                objPts,
                lm,
                sym_pairs,
                R_camera_imu,
                armor_pitch,
                roll,
                t_camera_armor,
                K
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
    Eigen::Vector<double, 5> toEigenDist(const cv::Mat& dist) {
        Eigen::Vector<double, 5> d = Eigen::Vector<double, 5>::Zero();

        if (dist.total() < 5) {
            return d; // 如果参数不足，返回 0 避免异常
        }

        if (dist.type() == CV_32F) {
            const float* p = dist.ptr<float>();
            for (int i = 0; i < 5; i++)
                d[i] = static_cast<double>(p[i]);
        } else if (dist.type() == CV_64F) {
            const double* p = dist.ptr<double>();
            for (int i = 0; i < 5; i++)
                d[i] = p[i];
        } else {
            // 兼容其他类型，逐个读取
            for (int i = 0; i < 5; i++)
                d[i] = dist.at<double>(i, 0);
        }

        return d;
    }
    class YawLocalParameterization: public ceres::LocalParameterization {
    public:
        bool Plus(const double* x, const double* delta, double* x_plus_delta) const override {
            x_plus_delta[0] = x[0] + delta[0];
            return true;
        }
        bool ComputeJacobian(const double* x, double* jacobian) const override {
            jacobian[0] = 1.0;
            return true;
        }
        int GlobalSize() const override {
            return 1;
        }
        int LocalSize() const override {
            return 1;
        }
    };

    struct CameraProjector {
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW

        CameraProjector(
            const Eigen::Matrix3d& Rci,
            double pitch,
            double roll,
            const Eigen::Vector3d& t,
            const Eigen::Matrix3d& K,
            const Eigen::Vector<double, 5>& dist
        ):
            Rci_(Rci),
            pitch_(pitch),
            roll_(roll),
            t_(t),
            K_(K),
            dist_(dist) {}

        template<typename T, typename Derived>
        inline Eigen::Matrix<T, 2, 1>
        project(const T& yaw, const Eigen::MatrixBase<Derived>& Pw) const {
            const T cy = ceres::cos(yaw);
            const T sy = ceres::sin(yaw);
            Eigen::Matrix<T, 3, 3> R_yaw;
            R_yaw << cy, -sy, T(0), sy, cy, T(0), T(0), T(0), T(1);

            const T cp = ceres::cos(T(pitch_));
            const T sp = ceres::sin(T(pitch_));
            Eigen::Matrix<T, 3, 3> R_pitch;
            R_pitch << cp, T(0), sp, T(0), T(1), T(0), -sp, T(0), cp;

            const T cr = ceres::cos(T(roll_));
            const T sr = ceres::sin(T(roll_));
            Eigen::Matrix<T, 3, 3> R_roll;
            R_roll << cr, -sr, T(0), sr, cr, T(0), T(0), T(0), T(1);

            const Eigen::Matrix<T, 3, 3> R = Rci_.cast<T>() * R_yaw * R_pitch * R_roll;

            const Eigen::Matrix<T, 3, 1> Pc = R * Pw + t_.cast<T>();

            if (Pc.z() < T(0.3)) {
                return Eigen::Matrix<T, 2, 1>(T(0), T(0));
            }

            const T x = Pc.x() / Pc.z();
            const T y = Pc.y() / Pc.z();

            if (!ceres::isfinite(x) || !ceres::isfinite(y)) {
                return Eigen::Matrix<T, 2, 1>(T(0), T(0));
            }

            const T r2 = x * x + y * y;
            const T k1 = T(dist_(0));
            const T k2 = T(dist_(1));
            const T p1 = T(dist_(2));
            const T p2 = T(dist_(3));
            const T k3 = T(dist_(4));

            const T x_dist = x * (T(1) + k1 * r2 + k2 * r2 * r2 + k3 * r2 * r2 * r2)
                + T(2) * p1 * x * y + p2 * (r2 + T(2) * x * x);
            const T y_dist = y * (T(1) + k1 * r2 + k2 * r2 * r2 + k3 * r2 * r2 * r2)
                + p1 * (r2 + T(2) * y * y) + T(2) * p2 * x * y;

            if (!ceres::isfinite(x_dist) || !ceres::isfinite(y_dist)) {
                return Eigen::Matrix<T, 2, 1>(T(0), T(0));
            }

            T xu = x_dist;
            T yu = y_dist;

            for (int i = 0; i < 8; i++) {
                const T r2_u = xu * xu + yu * yu;
                const T radial = T(1) + k1 * r2_u + k2 * r2_u * r2_u + k3 * r2_u * r2_u * r2_u;
                const T dx = T(2) * p1 * xu * yu + p2 * (r2_u + T(2) * xu * xu);
                const T dy = p1 * (r2_u + T(2) * yu * yu) + T(2) * p2 * xu * yu;

                xu = (x_dist - dx) / radial;
                yu = (y_dist - dy) / radial;

                if (!ceres::isfinite(xu) || !ceres::isfinite(yu)) {
                    return Eigen::Matrix<T, 2, 1>(T(0), T(0));
                }
            }
            Eigen::Matrix<T, 2, 1> uv;
            uv(0) = T(K_(0, 0)) * xu + T(K_(0, 2));
            uv(1) = T(K_(1, 1)) * yu + T(K_(1, 2));
            return uv;
        }

        Eigen::Matrix3d Rci_;
        double pitch_;
        double roll_;
        Eigen::Vector3d t_;
        Eigen::Matrix3d K_;
        Eigen::Vector<double, 5> dist_;
    };

    struct ReprojectionError {
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW

        ReprojectionError(
            const Eigen::Vector2d& uv,
            const Eigen::Vector3d& pt3d,
            const CameraProjector& cam
        ):
            uv_(uv),
            pt3_(pt3d),
            cam_(cam) {}

        template<typename T>
        bool operator()(const T* const yaw, T* residuals) const {
            const Eigen::Matrix<T, 2, 1> uv_proj = cam_.project(yaw[0], pt3_.cast<T>());

            residuals[0] = uv_proj(0) - T(uv_(0));
            residuals[1] = uv_proj(1) - T(uv_(1));
            return true;
        }

        Eigen::Vector2d uv_;
        Eigen::Vector3d pt3_;
        CameraProjector cam_;
    };
    struct SymmetryError {
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW

        SymmetryError(
            const Eigen::Vector3d& p1,
            const Eigen::Vector3d& p2,
            const Eigen::Vector2d& measCenter,
            const CameraProjector& cam
        ):
            p1_(p1),
            p2_(p2),
            meas_(measCenter),
            cam_(cam) {}

        template<typename T>
        bool operator()(const T* const yaw, T* residuals) const {
            const Eigen::Matrix<T, 2, 1> uv1 = cam_.project(yaw[0], p1_.cast<T>());
            const Eigen::Matrix<T, 2, 1> uv2 = cam_.project(yaw[0], p2_.cast<T>());

            residuals[0] = (uv1(0) + uv2(0)) * T(0.5) - T(meas_(0));
            residuals[1] = (uv1(1) + uv2(1)) * T(0.5) - T(meas_(1));
            return true;
        }

        Eigen::Vector3d p1_;
        Eigen::Vector3d p2_;
        Eigen::Vector2d meas_;
        CameraProjector cam_;
    };

private:
    Eigen::Matrix3d K_;
    std::pair<cv::Mat, cv::Mat> camera_info_;
    Eigen::Vector<double, 5> dist_eigen_;
};
BaSolver::BaSolver(const YAML::Node& config, const std::pair<cv::Mat, cv::Mat>& camera_info) {
    _impl = std::make_unique<Impl>(config, camera_info);
}
BaSolver::~BaSolver() {
    _impl.reset();
}
Eigen::Matrix3d BaSolver::solveBa_R(
    const ArmorObject& armor,
    const Eigen::Vector3d& t_camera_armor,
    const Eigen::Matrix3d& R_camera_armor,
    const Eigen::Matrix3d& R_imu_camera,
    const std::string& type
) const noexcept {
    return _impl->solveBa_R(armor, t_camera_armor, R_camera_armor, R_imu_camera, type);
}
} // namespace auto_aim