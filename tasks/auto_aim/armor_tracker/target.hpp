#pragma once
#include "tasks/auto_aim/armor_tracker/motion_models/motion_modelypdv2.hpp"
#include "tasks/auto_aim/type.hpp"
#include "wust_vl/common/utils/parameter.hpp"
#include <wust_vl/common/utils/timer.hpp>
namespace wust_vision {
namespace auto_aim {
    namespace MModel = ypdv2armor_motion_model;

    struct TargetConfig: wust_vl::common::utils::ParamGroup {
        static constexpr const char* kKey = "armor_tracker";
        const char* key() const override {
            return kKey;
        }
        GEN_PARAM(int, esekf_iter_num);
        GEN_PARAM(double, lost_time_thres);
        GEN_PARAM(int, tracking_thres);
        GEN_PARAM(double, max_yaw_diff_deg);
        GEN_PARAM(double, max_dis_diff);
        GEN_PARAM(double, match_gate);
        GEN_PARAM(double, qyaw_common);
        GEN_PARAM(double, qyaw_output);
        GEN_PARAM(double, q_r);
        GEN_PARAM(double, q_l);
        GEN_PARAM(double, q_h);
        GEN_PARAM(double, q_outpost_dz);
        GEN_PARAM(double, yp_r);
        GEN_PARAM(double, dis_r_front);
        GEN_PARAM(double, dis_r_side);
        GEN_PARAM(double, dis2_r_ratio);
        GEN_PARAM(double, yaw_r_base_front);
        GEN_PARAM(double, yaw_r_base_side);
        GEN_PARAM(double, yaw_r_log_ratio);
        GEN_PARAM(std::vector<double>, qxyz_common);
        GEN_PARAM(std::vector<double>, qxyz_output);
        Eigen::Vector3d qxyz_common = { 100, 100, 100 };
        Eigen::Vector3d qxyz_output = { 10, 10, 10 };
        using Ptr = std::shared_ptr<TargetConfig>;
        TargetConfig() {
            qxyz_output_param.onChange([this](auto o, auto n) {
                qxyz_common = Eigen::Vector3d(n[0], n[1], n[2]);
            });
            qxyz_output_param.onChange([this](auto o, auto n) {
                qxyz_output = Eigen::Vector3d(n[0], n[1], n[2]);
            });
        }
        static Ptr create() {
            return std::make_shared<TargetConfig>();
        }
        void loadSelf(const YAML::Node& node) override {
            esekf_iter_num_param.load(node);
            lost_time_thres_param.load(node);
            tracking_thres_param.load(node);
            max_yaw_diff_deg_param.load(node);
            max_dis_diff_param.load(node);
            match_gate_param.load(node);
            qyaw_common_param.load(node);
            qyaw_output_param.load(node);
            qxyz_common_param.load(node);
            qxyz_output_param.load(node);
            q_r_param.load(node);
            q_l_param.load(node);
            q_h_param.load(node);
            q_outpost_dz_param.load(node);
            yp_r_param.load(node);
            dis_r_front_param.load(node);
            dis_r_side_param.load(node);
            yaw_r_base_front_param.load(node);
            yaw_r_base_side_param.load(node);
            yaw_r_log_ratio_param.load(node);
        }
    };
    class Target {
    public:
        Target();
        Target(const Armor& armor, TargetConfig::Ptr target_config);
        MModel::Measure::MeasureCtx ctx_;
        ArmorNumber tracked_id_;
        std::string type_;
        MModel::VecZ measurement_ = Eigen::Matrix<double, MModel::Z_N, 1>::Zero();
        MModel::State target_state_ = MModel::State();
        double radius_pre_;

        int armor_num_ = 4;
        bool jumped = false;
        bool is_inited = false;
        bool is_tracking = false;
        std::chrono::steady_clock::time_point last_t_;
        std::chrono::steady_clock::time_point timestamp_;
        MModel::RobotStateESEKF esekf_ypd_;
        TargetConfig::Ptr target_config_;
        [[nodiscard]] cv::Rect expanded(
            Eigen::Matrix4d T_camera_to_odom,
            const cv::Mat& camera_intrinsic,
            const cv::Mat& camera_distortion,
            const cv::Size& image_size
        ) const noexcept;
        void predict(std::chrono::steady_clock::time_point t) noexcept;
        void predict(double dt) noexcept;
        void predictSimple(std::chrono::steady_clock::time_point t) noexcept;
        void predictSimple(double dt) noexcept;
        [[nodiscard]] MModel::Predict getPredictFunc(double dt) const noexcept;
        bool update(const std::pair<int, Armor>& armor) noexcept;
        [[nodiscard]] Eigen::Matrix<double, MModel::Z_N, MModel::Z_N>
        computeMeasurementCovariance(const Eigen::Matrix<double, MModel::Z_N, 1>& z) const noexcept;
        [[nodiscard]] Eigen::Matrix<double, MModel::X_N, MModel::X_N> computeProcessNoise(double dt
        ) const noexcept;
        [[nodiscard]] std::optional<ArmorNumber> getArmorNumber() const noexcept {
            if (!checkTargetAppear()) {
                return std::nullopt;
            }
            return tracked_id_;
        }

        [[nodiscard]] std::vector<double> getArmorYaws() const noexcept {
            std::vector<double> yaw_list;
            yaw_list.reserve(armor_num_);
            for (int i = 0; i < armor_num_; i++) {
                MModel::Measure::MeasureCtx _ctx(i, armor_num_);
                MModel::Measure measure(_ctx);
                yaw_list.push_back(measure.get_angle(target_state_.x.data()));
            }
            return yaw_list;
        }
        [[nodiscard]] std::vector<Eigen::Vector3d> getArmorPositions() const noexcept {
            std::vector<Eigen::Vector3d> armor_positions;
            armor_positions.reserve(armor_num_);
            for (int i = 0; i < armor_num_; i++) {
                MModel::Measure::MeasureCtx _ctx(i, armor_num_);
                MModel::Measure measure(_ctx);
                const Eigen::Vector4d xyza = measure.h_armor_xyza(target_state_.x);
                armor_positions.push_back(xyza.head<3>());
            }
            return armor_positions;
        }
        [[nodiscard]] std::vector<Eigen::Vector4d> getArmorPosAndYaw() const noexcept {
            std::vector<Eigen::Vector4d> pos_yaw;
            pos_yaw.reserve(armor_num_);
            for (int i = 0; i < armor_num_; ++i) {
                MModel::Measure::MeasureCtx _ctx(i, armor_num_);
                MModel::Measure measure(_ctx);
                const Eigen::Vector4d xyza = measure.h_armor_xyza(target_state_.x);
                pos_yaw.push_back(xyza);
            }
            return pos_yaw;
        }
        [[nodiscard]] double getMeanZ() const noexcept {
            double mean = 0;
            for (const auto& p: getArmorPositions()) {
                mean += p.z();
            }
            return mean / armor_num_;
        }
        [[nodiscard]] double getArmor2CenterXYDis(int id) const noexcept {
            const auto use_l_h = (armor_num_ == 4) && (id == 1 || id == 3);
            const auto r = (use_l_h) ? target_state_.r() + target_state_.l() : target_state_.r();
            return r;
        }
        [[nodiscard]] std::vector<std::pair<int, Armor>> match(const std::vector<Armor>& armors
        ) noexcept;

        [[nodiscard]] inline bool checkTargetAppear() const noexcept {
            const bool appear = is_tracking
                && wust_vl::common::utils::time_utils::durationSec(
                       timestamp_,
                       wust_vl::common::utils::time_utils::now()
                   ) < target_config_->lost_time_thres_param.get();
            return appear;
        }

        [[nodiscard]] Eigen::Matrix<double, MModel::Z_N, 1> getMeasure(const Armor& a) noexcept {
            const auto p = a.target_pos;
            const double measured_yaw = utils::orientationToYaw<Target>(a.target_ori);
            double ypd_y = std::atan2(p.y(), p.x());
            static double last_ypd_y = 0;
            ypd_y = last_ypd_y + angles::shortest_angular_distance(last_ypd_y, ypd_y);
            last_ypd_y = ypd_y;
            const double ypd_p = std::atan2(p.z(), std::sqrt(p.x() * p.x() + p.y() * p.y()));
            const double ypd_d = std::sqrt(p.x() * p.x() + p.y() * p.y() + p.z() * p.z());

            return Eigen::Vector4d(ypd_y, ypd_p, ypd_d, measured_yaw);
        }
    };
} // namespace auto_aim
} // namespace wust_vision