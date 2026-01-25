#include "ba_solver.hpp"
#include "tasks/utils.hpp"
#include <Eigen/Dense>
#include <ceres/autodiff_cost_function.h>
#include <ceres/jet.h>
#include <ceres/loss_function.h>
#include <ceres/problem.h>
#include <ceres/solver.h>
#include <opencv2/core.hpp>
namespace wust_vision {
namespace auto_buff {
    struct BaSolver::Impl {
    public:
        Impl(const YAML::Node& config, const cv::Mat& camera_matrix) {
            cv::cv2eigen(camera_matrix, K_);
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

        double reprojectionErrorRoll(
            double roll,
            const std::vector<Eigen::Vector3d>& obj,
            const std::vector<cv::Point2f>& lm,
            const Eigen::Matrix3d& Rci,
            double pitch,
            double yaw,
            const Eigen::Vector3d& t,
            const Eigen::Matrix3d& K
        ) const noexcept {
            const double fx = K(0, 0), fy = K(1, 1), cx = K(0, 2), cy = K(1, 2);

            // R_roll
            double cr = std::cos(roll);
            double sr = std::sin(roll);
            Eigen::Matrix3d R_roll;
            R_roll << 1, 0, 0, 0, cr, -sr, 0, sr, cr;

            // R_yaw
            double cyaw = std::cos(yaw);
            double syaw = std::sin(yaw);
            Eigen::Matrix3d R_yaw;
            R_yaw << cyaw, 0, syaw, 0, 1, 0, -syaw, 0, cyaw;

            // R_pitch
            double cp = std::cos(pitch);
            double sp = std::sin(pitch);
            Eigen::Matrix3d R_pitch;
            R_pitch << cp, 0, sp, 0, 1, 0, -sp, 0, cp;

            Eigen::Matrix3d R = Rci * R_yaw * R_roll * R_pitch;

            double mse = 0.0;
            int N = obj.size();
            constexpr double INF = std::numeric_limits<double>::infinity();

            for (int i = 0; i < N; i++) {
                Eigen::Vector3d Pc = R * obj[i] + t;
                if (Pc.z() < 1e-6)
                    return INF;

                double u = fx * (Pc.x() / Pc.z()) + cx;
                double v = fy * (Pc.y() / Pc.z()) + cy;

                double du = u - lm[i].x;
                double dv = v - lm[i].y;

                mse += du * du + dv * dv;
            }
            return mse / N;
        }

        double goldenRoll(
            double init,
            const std::vector<Eigen::Vector3d>& obj,
            const std::vector<cv::Point2f>& lm,
            const Eigen::Matrix3d& Rci,
            double pitch,
            double yaw,
            const Eigen::Vector3d& t,
            const Eigen::Matrix3d& K
        ) const noexcept {
            constexpr double phi = 1.618033988749894848; // golden ratio
            double l = init - params_.golden_search_side_deg * M_PI / 180.0;
            double r = init + params_.golden_search_side_deg * M_PI / 180.0;

            double r1 = r - (r - l) / phi;
            double r2 = l + (r - l) / phi;

            double f1 = reprojectionErrorRoll(r1, obj, lm, Rci, pitch, yaw, t, K);
            double f2 = reprojectionErrorRoll(r2, obj, lm, Rci, pitch, yaw, t, K);

            while (r - l > 0.0001) { // 约 0.0057 度
                if (f1 < f2) {
                    r = r2;
                    r2 = r1;
                    f2 = f1;
                    r1 = r - (r - l) / phi;
                    f1 = reprojectionErrorRoll(r1, obj, lm, Rci, pitch, yaw, t, K);
                } else {
                    l = r1;
                    r1 = r2;
                    f1 = f2;
                    r2 = l + (r - l) / phi;
                    f2 = reprojectionErrorRoll(r2, obj, lm, Rci, pitch, yaw, t, K);
                }
            }

            return 0.5 * (l + r); // final best roll
        }

        double ceresRoll(
            double init,
            const std::vector<Eigen::Vector3d>& obj,
            const std::vector<cv::Point2f>& lm,
            const Eigen::Matrix3d& Rci,
            double pitch,
            double yaw,
            const Eigen::Vector3d& t,
            const Eigen::Matrix3d& K
        ) const noexcept {
            double roll = init;
            ceres::Problem problem;

            for (size_t i = 0; i < obj.size(); ++i) {
                ceres::CostFunction* costFn =
                    new ceres::AutoDiffCostFunction<RollProjectionError, 2, 1>(
                        new RollProjectionError(
                            Eigen::Vector2d(lm[i].x, lm[i].y),
                            obj[i],
                            Rci,
                            pitch,
                            yaw,
                            t,
                            K_
                        )
                    );

                problem.AddResidualBlock(costFn, new ceres::HuberLoss(3.0), &roll);
            }

            ceres::Solver::Options options;
            options.max_num_iterations = params_.ceres_max_iter;
            options.linear_solver_type = ceres::DENSE_QR;
            ceres::Solver::Summary summary;
            ceres::Solve(options, &problem, &summary);

            return roll;
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
            std::vector<Eigen::Vector3d> object_points;
            object_points.reserve(cv_points.size());
            for (const auto& p: cv_points) {
                object_points.emplace_back(p.x, p.y, p.z);
            }
            const auto& landmarks = rune_fan.landmarks();
            if (params_.mode == Params::OptMode::CERES) {
                roll = ceresRoll(
                    roll,
                    object_points,
                    landmarks,
                    R_camera_imu,
                    pitch,
                    yaw,
                    t_camera_armor,
                    K_
                );
            } else if (params_.mode == Params::OptMode::GOLDEN) {
                roll = goldenRoll(
                    roll,
                    object_points,
                    landmarks,
                    R_camera_imu,
                    pitch,
                    yaw,
                    t_camera_armor,
                    K_
                );
            }

            double cr = cos(roll), sr = sin(roll);
            Eigen::Matrix3d R_roll;
            R_roll << 1, 0, 0, 0, cr, -sr, 0, sr, cr;

            double cyaw = cos(yaw), syaw = sin(yaw);
            Eigen::Matrix3d R_yaw;
            R_yaw << cyaw, 0, syaw, 0, 1, 0, -syaw, 0, cyaw;

            Eigen::Matrix3d R_pitch;
            R_pitch << 1, 0, 0, 0, 1, 0, 0, 0, 1;

            return R_camera_imu * R_yaw * R_roll * R_pitch;
        }

    private:
        Eigen::Matrix3d K_;
        struct RollProjectionError {
            RollProjectionError(
                const Eigen::Vector2d& uv,
                const Eigen::Vector3d& pt_3d,
                const Eigen::Matrix3d& Rci,
                double pitch,
                double yaw,
                const Eigen::Vector3d& t,
                const Eigen::Matrix3d& K
            ):
                uv_(uv),
                pt3_(pt_3d),
                Rci_(Rci),
                pitch_(pitch),
                yaw_(yaw),
                t_(t),
                K_(K) {}

            template<typename T>
            bool operator()(const T* const roll, T* residuals) const {
                Eigen::Matrix<T, 3, 3> R_roll;
                R_roll << T(1), T(0), T(0), T(0), ceres::cos(*roll), -ceres::sin(*roll), T(0),
                    ceres::sin(*roll), ceres::cos(*roll);

                // yaw 和 pitch 常量
                T cy = ceres::cos(T(yaw_)), sy = ceres::sin(T(yaw_));
                Eigen::Matrix<T, 3, 3> R_yaw;
                R_yaw << cy, T(0), sy, T(0), T(1), T(0), -sy, T(0), cy;

                T cp = ceres::cos(T(pitch_)), sp = ceres::sin(T(pitch_));
                Eigen::Matrix<T, 3, 3> R_pitch;
                // Note: consistent with your previous Rpitch = exp([0, pitch, 0])
                R_pitch << cp, T(0), sp, T(0), T(1), T(0), -sp, T(0), cp;

                // R = Rci * R_yaw * R_pitch
                Eigen::Matrix<T, 3, 3> R = Rci_.cast<T>() * R_yaw * R_pitch;

                Eigen::Matrix<T, 3, 1> Pc = R * pt3_.cast<T>() + t_.cast<T>();
                // project (assumes fx == K(0,0), fy == K(1,1), cx, cy)
                T u = T(K_(0, 0)) * Pc.x() / Pc.z() + T(K_(0, 2));
                T v = T(K_(1, 1)) * Pc.y() / Pc.z() + T(K_(1, 2));

                residuals[0] = u - T(uv_(0));
                residuals[1] = v - T(uv_(1));
                return true;
            }

            const Eigen::Vector2d uv_;
            const Eigen::Vector3d pt3_;
            const Eigen::Matrix3d Rci_;
            const double pitch_, yaw_;
            const Eigen::Vector3d t_;
            const Eigen::Matrix3d K_;
        };
    };
    BaSolver::BaSolver(const YAML::Node& config, const cv::Mat& camera_matrix) {
        _impl = std::make_unique<Impl>(config, camera_matrix);
    }
    BaSolver::~BaSolver() {
        _impl.reset();
    }
    Eigen::Matrix3d BaSolver::solveBa_R(
        const auto_buff::RuneFan::Simple& rune_fan,
        const Eigen::Vector3d& t_camera_armor,
        const Eigen::Matrix3d& R_camera_armor,
        const Eigen::Matrix3d& R_imu_camera
    ) const noexcept {
        return _impl->solveBa_R(rune_fan, t_camera_armor, R_camera_armor, R_imu_camera);
    }
} // namespace auto_buff
} // namespace wust_vision