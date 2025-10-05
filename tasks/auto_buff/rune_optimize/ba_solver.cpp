#include "ba_solver.hpp"
// std
#include <iostream>
#include <memory>
// g2o
#include <g2o/core/robust_kernel.h>
#include <g2o/core/robust_kernel_factory.h>
#include <g2o/core/robust_kernel_impl.h>
#include <g2o/types/slam3d/types_slam3d.h>
// 3rd party
#include <Eigen/Core>
#include <opencv2/core/eigen.hpp>
#include <sophus/se3.hpp>
#include <sophus/so3.hpp>
// project
#include "graph_optimizer.hpp"
#include "tasks/utils.hpp"
#include "wust_vl/common/utils/logger.hpp"
namespace rune {
G2O_USE_OPTIMIZATION_LIBRARY(dense)

BaSolver::BaSolver(
    std::array<double, 9>& camera_matrix,
    int max_iter_R,
    int max_iter_t,
    int step_R,
    int step_t,
    double min_error_R,
    double min_error_t
):
    max_iter_R_(max_iter_R),
    max_iter_t_(max_iter_t),
    step_R_(step_R),
    step_t_(step_t),
    min_error_R_(min_error_R),
    min_error_t_(min_error_t) {
    K_ = Eigen::Matrix3d::Identity();
    K_(0, 0) = camera_matrix[0];
    K_(1, 1) = camera_matrix[4];
    K_(0, 2) = camera_matrix[2];
    K_(1, 2) = camera_matrix[5];

    // Optimization information
    optimizer_.setVerbose(false);
    // Optimization method
    optimizer_.setAlgorithm(
        g2o::OptimizationAlgorithmFactory::instance()->construct("lm_dense", solver_property_)
    );
    // Initial step size
    lm_algorithm_ = dynamic_cast<g2o::OptimizationAlgorithmLevenberg*>(
        const_cast<g2o::OptimizationAlgorithm*>(optimizer_.algorithm())
    );
    lm_algorithm_->setUserLambdaInit(0.1);
}
Eigen::Matrix3d BaSolver::solveBa_R(
    const rune::RuneFan& rune_fan,
    const Eigen::Vector3d& t_camera_armor,
    const Eigen::Matrix3d& R_camera_armor, // camera <- armor
    const Eigen::Matrix3d& R_imu_camera, // imu -> camera
    const cv::Mat& camera_matrix,
    const cv::Mat& distort_coeffs
) noexcept {
    optimizer_.clear();

    // 坐标系变换
    Eigen::Matrix3d R_imu_armor = R_imu_camera * R_camera_armor;
    Sophus::SO3d R_camera_imu(R_imu_camera.transpose());

    // --- roll 初值估计 ---
    double initial_roll = std::atan2(R_imu_armor(2, 1), R_imu_armor(2, 2));

    // 固定 pitch
    double fixed_pitch = 0.0;
    Sophus::SO3d R_pitch = Sophus::SO3d::exp(Eigen::Vector3d(0, fixed_pitch, 0));

    // 固定 yaw
    double fixed_yaw = std::atan2(R_imu_armor(1, 0), R_imu_armor(0, 0));
    Sophus::SO3d R_yaw = Sophus::SO3d::exp(Eigen::Vector3d(0, 0, fixed_yaw));

    // 构建 3D 角点
    auto cv_points = rune_fan.getObjs();
    std::vector<Eigen::Vector3d> object_points;
    object_points.reserve(cv_points.size());
    for (const auto& p: cv_points) {
        object_points.emplace_back(p.x, p.y, p.z);
    }
    const auto& landmarks = rune_fan.landmarks();

    size_t id_counter = 0;
    // 添加 roll 顶点
    auto* v_roll = new VertexRoll();
    v_roll->setId(id_counter++);
    v_roll->setEstimate(initial_roll);
    optimizer_.addVertex(v_roll);

    // 添加投影边
    for (size_t i = 0; i < object_points.size(); ++i) {
        auto* v_pt = new g2o::VertexPointXYZ();
        v_pt->setId(id_counter++);
        v_pt->setEstimate(object_points[i]);
        v_pt->setFixed(true);
        optimizer_.addVertex(v_pt);

        auto* edge = new EdgeProjectionWithRoll(R_camera_imu, R_pitch, t_camera_armor, K_);
        edge->setId(id_counter++);
        edge->setVertex(0, v_roll);
        edge->setVertex(1, v_pt);
        edge->setMeasurement(Eigen::Vector2d(landmarks[i].x, landmarks[i].y));
        edge->setInformation(Eigen::Matrix2d::Identity());
        edge->setRobustKernel(new g2o::RobustKernelHuber);
        optimizer_.addEdge(edge);
    }
    // 优化
    optimizer_.initializeOptimization();
    int numEdges = optimizer_.edges().size();
    auto computeRMS = [&]() { return std::sqrt(optimizer_.chi2() / numEdges); };

    for (int k = 0; k < max_iter_R_ / step_R_; ++k) {
        optimizer_.optimize(step_R_);
        double rms = computeRMS();
        if (rms < min_error_R_)
            break;
    }

    double roll_opt = v_roll->estimate();
    if (!std::isfinite(roll_opt)) {
        std::cerr << "[BaSolver] Roll is invalid after g2o optimization. Reset to 0.\n";
        roll_opt = 0.0;
    }

    Sophus::SO3d R_roll = Sophus::SO3d::exp(Eigen::Vector3d(roll_opt, 0, 0));

    Eigen::Matrix3d R_final = (R_camera_imu * R_yaw * R_roll * R_pitch).matrix();
    if (!R_final.allFinite()) {
        std::cerr << "[BaSolver] R_final contains invalid elements, return identity\n";
        return Eigen::Matrix3d::Identity();
    }

    return R_final;
}

} // namespace rune