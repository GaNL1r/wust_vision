#include "3rdparty/angles.h"
//#ifdef USE_ROS2
#ifdef FUCK
    #include "auto_sniper.hpp"
    #include "ros2/tf.hpp"

    #include <Eigen/src/Core/Matrix.h>
    #include <atomic>
    #include <memory>
    #include <mutex>
    #include <open3d/utility/Eigen.h>
    #include <optional>
    #include <rclcpp/logger.hpp>
    #include <rclcpp/logging.hpp>
    #include <thread>
    #include <vector>

    #include <Eigen/Dense>

    #include <nav_msgs/msg/odometry.hpp>
    #include <open3d/Open3D.h>
    #include <rclcpp/node.hpp>
    #include <rclcpp/rclcpp.hpp>

    #include "k1_solver.hpp"
    #include "offset_helper.hpp"
    #include "tasks/type_common.hpp"
    #include "tasks/utils/config.hpp"
    #include "voxel_map.hpp"
    #include <sensor_msgs/msg/point_cloud2.hpp>
    #include <sensor_msgs/point_cloud2_iterator.hpp>
    #include <visualization_msgs/msg/marker.hpp>
    #include <yaml-cpp/yaml.h>
namespace wust_vision::auto_sniper {

struct AutoSniper::Impl {
    Impl(
        rclcpp::Node& node,
        std::shared_ptr<wust_vl::common::utils::MotionBufferGeneric<Motion, 1024>> motion_buffer
    ) {
        node_ = &node;
        motion_buffer_ = motion_buffer;
        tf_ = TF::create(*node_);
        auto config = YAML::LoadFile(AUTO_SNIPER_CONFIG);
        auto map_config = config["map"];
        auto min_pos_v = map_config["min_pos"].as<std::vector<double>>();
        auto min_pos = Eigen::Vector3d(min_pos_v[0], min_pos_v[1], min_pos_v[2]);
        auto max_pos_v = map_config["max_pos"].as<std::vector<double>>();
        auto max_pos = Eigen::Vector3d(max_pos_v[0], max_pos_v[1], max_pos_v[2]);
        voxel_map_ = std::make_shared<SlidingVoxelMap<3, Cell>>(
            map_config["voxel_size"].as<double>(),
            min_pos,
            max_pos,
            true
        );
        auto solver_config = config["solver"];
        vis_cloud_ = std::make_shared<open3d::geometry::PointCloud>();
        k1_solver_ = K1BallisticSolver::create(
            solver_config["k1"].as<double>(),
            solver_config["g"].as<double>()
        );
        target_armor_z_ = solver_config["target_armor_z"].as<double>();
        offset_helper_ = OffsetHelper::create(config["offset_helper"]);
        pointcloud_sub_ = node_->create_subscription<sensor_msgs::msg::PointCloud2>(
            "/cloud_registered",
            rclcpp::SensorDataQoS(),
            std::bind(&AutoSniper::Impl::pointCloudCallback, this, std::placeholders::_1)
        );

        odometry_sub_ = node_->create_subscription<nav_msgs::msg::Odometry>(
            "/Odometry",
            10,
            std::bind(&AutoSniper::Impl::odomCallback, this, std::placeholders::_1)
        );
        traj_pub_ =
            node_->create_publisher<visualization_msgs::msg::Marker>("bullet_trajectory", 10);
    }
    void start() {
        if (run_flag_) {
            return;
        }
        run_flag_ = true;
        vis_thread_ = wust_vl::common::concurrency::MonitoredThread::create(
            "AutoSniperVisualizer",
            [this](wust_vl::common::concurrency::MonitoredThread::Ptr self) {
                this->visualizeLoop(self);
            }
        );
        wust_vl::common::concurrency::ThreadManager::instance().registerThread(vis_thread_);
    }
    void doDebug() {}
    void pushInput(CommonFrame& frame) {}
    wust_vl::common::concurrency::MonitoredThread::Ptr getThread() {
        return vis_thread_;
    }
    GimbalCmd solve(double bullet_speed) noexcept {
        GimbalCmd cmd;
        if (!target_pos_in_map_.has_value()) {
            cmd.appera = false;
            return cmd;
        }
        Eigen::Vector3d target_pos_in_self = self_in_map_.inverse() * target_pos_in_map_.value();
        target_pos_in_self.z() = target_armor_z_;
        auto pitch = k1_solver_->solvePitch(target_pos_in_self, bullet_speed);
        if (!pitch.has_value()) {
            cmd.appera = false;
            std::cout << "no pitch" << std::endl;
            return cmd;
        }
        double yaw =
            angles::normalize_angle(std::atan2(target_pos_in_self.y(), target_pos_in_self.x()));
        auto traj = k1_solver_->computeTrajectory(
            self_in_map_.translation(),
            target_pos_in_map_.value(),
            bullet_speed,
            0.01
        );
        double yaw_deg = angles::to_degrees(yaw);
        double pitch_deg = angles::to_degrees(pitch.value());
        publishTrajectoryMarker(traj);

        auto control_pitch = pitch_deg + offset_helper_->getPitchOffset(target_pos_in_self.norm());

        auto control_yaw = yaw_deg + offset_helper_->getYawOffset(target_pos_in_self.norm());
        if (auto last_att = motion_buffer_->get_last()) {
            control_yaw += angles::to_degrees(last_att->data.yaw);
        }
        cmd.appera = true;
        cmd.target_pitch = control_pitch;
        cmd.target_yaw = control_yaw;
        cmd.appera = true;
        cmd.yaw = control_yaw;
        cmd.pitch = control_pitch;
        cmd.distance = target_pos_in_self.norm();
        cmd.enable_pitch_diff = 0.5;
        cmd.enable_yaw_diff = 0.5;
        static int count = 0;
        count++;
        if (count % 100 == 0) {
            std::cout << cmd.yaw << "  " << cmd.pitch << std::endl;
        }
        return cmd;
    }
    void publishTrajectoryMarker(const std::vector<Eigen::Vector3d>& traj) {
        static auto last_pub_time = std::chrono::steady_clock::now();
        auto now = std::chrono::steady_clock::now();
        auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(now - last_pub_time);
        if (elapsed.count() < 33) {
            return;
        }
        last_pub_time = now;

        if (traj.empty())
            return;
        visualization_msgs::msg::Marker marker;
        marker.header.frame_id = target_frame_;
        marker.header.stamp = node_->now();
        marker.ns = "bullet";
        marker.id = 0;
        marker.type = visualization_msgs::msg::Marker::LINE_STRIP;
        marker.action = visualization_msgs::msg::Marker::ADD;
        marker.scale.x = 0.1;
        marker.color.r = 0.0;
        marker.color.g = 1.0;
        marker.color.b = 1.0;
        marker.color.a = 1.0;
        marker.lifetime = rclcpp::Duration::from_seconds(1.0);
        for (auto& p: traj) {
            geometry_msgs::msg::Point pt;
            pt.x = p.x();
            pt.y = p.y();
            pt.z = p.z();
            marker.points.push_back(pt);
        }

        traj_pub_->publish(marker);
    }
    void odomCallback(const nav_msgs::msg::Odometry::SharedPtr msg) {
        auto T = tf_->getTransform(target_frame_, "gimbal_yaw", msg->header.stamp);

        if (!T.has_value()) {
            return;
        }
        self_in_map_ = T.value().cast<double>();
        // Eigen::Isometry3f
        // Eigen::Vector4f p(
        //     msg->pose.pose.position.x,
        //     msg->pose.pose.position.y,
        //     msg->pose.pose.position.z,
        //     1.0f
        // );

        // auto p_target = T.value() * p;
        // self_pos_ = Eigen::Vector3d(p_target.x(), p_target.y(), p_target.z());
    }

    void visualizeLoop(wust_vl::common::concurrency::MonitoredThread::Ptr self) {
        while (!self->isAlive()) {
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }

        auto vis_cloud = std::make_shared<open3d::geometry::PointCloud>();
        std::atomic<bool> picking = false;
        auto future = std::async(std::launch::async, [&, self]() {
            while (self->isAlive() && run_flag_) {
                picking = true;
                pickBlocking(vis_cloud);
                picking = false;
                std::this_thread::sleep_for(std::chrono::milliseconds(1000));
            }
        });

        while (self->isAlive() && run_flag_) {
            self->heartbeat();
            if (!picking) {
                std::lock_guard<std::mutex> lock(cloud_mutex_);
                *vis_cloud = *vis_cloud_;
            }

            std::this_thread::sleep_for(std::chrono::milliseconds(30));
        }
        if (future.valid()) {
            future.wait();
        }
    }

    void pickBlocking(std::shared_ptr<open3d::geometry::PointCloud> pcd) {
        open3d::visualization::VisualizerWithEditing vis;
        vis.CreateVisualizerWindow("Pick points (close window when done)", 1280, 720);
        vis.AddGeometry(pcd);
        vis.Run();

        auto picked = vis.GetPickedPoints();
        if (!picked.empty()) {
            Eigen::Vector3d picked_pos = pcd->points_[picked.back()];
            target_pos_in_map_ = Eigen::Vector3d(picked_pos);

            RCLCPP_INFO_STREAM(
                rclcpp::get_logger("awm"),
                " Target pos: " << picked_pos.transpose()
                                << " self pos: " << self_in_map_.translation().transpose()
            );
        }

        vis.DestroyVisualizerWindow();
    }

    void pointCloudCallback(const sensor_msgs::msg::PointCloud2::SharedPtr msg) {
        auto T = tf_->getTransform(target_frame_, msg->header.frame_id, msg->header.stamp);

        if (!T.has_value()) {
            return;
        }

        const size_t size = msg->width * msg->height;

        sensor_msgs::PointCloud2ConstIterator<float> iter_x(*msg, "x");
        sensor_msgs::PointCloud2ConstIterator<float> iter_y(*msg, "y");
        sensor_msgs::PointCloud2ConstIterator<float> iter_z(*msg, "z");

        std::vector<Eigen::Vector3d> new_points;
        new_points.reserve(size);

        for (size_t i = 0; i < size; ++i) {
            Eigen::Vector4f p(*iter_x, *iter_y, *iter_z, 1.0f);

            p = T.value() * p;

            if (std::isfinite(p.x()) && std::isfinite(p.y()) && std::isfinite(p.z())) {
                new_points.emplace_back(p.head<3>().cast<double>());
            }

            ++iter_x;
            ++iter_y;
            ++iter_z;
        }
        for (const auto& p: new_points) {
            auto idx = voxel_map_->worldToIndex(p);
            if (idx > 0) {
                voxel_map_->grid[idx].v = 1;
            }
        }
        std::vector<Eigen::Vector3d> pointcloud;
        for (int i = 0; i < voxel_map_->grid.size(); i++) {
            if (voxel_map_->grid[i].v == 1) {
                pointcloud.emplace_back(voxel_map_->indexToWorld(i));
            }
        }
        {
            std::lock_guard<std::mutex> lock(cloud_mutex_);

            vis_cloud_->points_ = pointcloud;
        }
    }

    std::string target_frame_ = "map";

    rclcpp::Node* node_;
    std::optional<Eigen::Vector3d> target_pos_in_map_ = std::nullopt;
    rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr pointcloud_sub_;
    rclcpp::Publisher<visualization_msgs::msg::Marker>::SharedPtr traj_pub_;
    rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr odometry_sub_;
    TF::Ptr tf_;
    std::shared_ptr<open3d::geometry::PointCloud> vis_cloud_;
    struct Cell {
        uint8_t v = 0;
    };
    SlidingVoxelMap<3, Cell>::Ptr voxel_map_;
    K1BallisticSolver::Ptr k1_solver_;
    OffsetHelper::Ptr offset_helper_;
    wust_vl::common::concurrency::MonitoredThread::Ptr vis_thread_;
    bool run_flag_ = false;
    std::mutex cloud_mutex_;
    double target_armor_z_ = 0.0;
    Eigen::Isometry3d self_in_map_ = Eigen::Isometry3d::Identity();
    std::shared_ptr<wust_vl::common::utils::MotionBufferGeneric<Motion, 1024>> motion_buffer_;
};
AutoSniper::AutoSniper(
    rclcpp::Node& node,
    std::shared_ptr<wust_vl::common::utils::MotionBufferGeneric<Motion, 1024>> motion_buffer
) {
    _impl = std::make_unique<Impl>(node, motion_buffer);
}
AutoSniper::~AutoSniper() {
    _impl.reset();
}
void AutoSniper::start() {
    _impl->start();
}
void AutoSniper::doDebug() {
    _impl->doDebug();
}
void AutoSniper::pushInput(CommonFrame& frame) {
    _impl->pushInput(frame);
}

wust_vl::common::concurrency::MonitoredThread::Ptr AutoSniper::getThread() {
    return _impl->getThread();
}
GimbalCmd AutoSniper::solve(double bullet_speed) {
    return _impl->solve(bullet_speed);
}
} // namespace wust_vision::auto_sniper
#endif