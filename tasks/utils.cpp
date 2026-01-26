#include "utils.hpp"
#include <iomanip>
#include <pwd.h>
#include <regex>
#include <opencv2/core/eigen.hpp>
// util functions
namespace wust_vision {
namespace utils {
    double limit_rad(double angle) noexcept {
        while (angle > M_PI)
            angle -= 2.0 * M_PI;
        while (angle < -M_PI)
            angle += 2.0 * M_PI;
        return angle;
    }

    Eigen::Vector3d
    quatToEuler(const Eigen::Quaterniond& q, int axis0, int axis1, int axis2, bool extrinsic) {
        if (!extrinsic)
            std::swap(axis0, axis2);

        auto i = axis0, j = axis1, k = axis2;
        bool is_proper = (i == k);
        if (is_proper)
            k = 3 - i - j;
        int sign = (i - j) * (j - k) * (k - i) / 2;

        double a, b, c, d;
        const Eigen::Vector4d xyzw = q.coeffs(); // [x,y,z,w]
        if (is_proper) {
            a = xyzw[3];
            b = xyzw[i];
            c = xyzw[j];
            d = xyzw[k] * sign;
        } else {
            a = xyzw[3] - xyzw[j];
            b = xyzw[i] + xyzw[k] * sign;
            c = xyzw[j] + xyzw[3];
            d = xyzw[k] * sign - xyzw[i];
        }

        Eigen::Vector3d eulers;
        const double n2 = a * a + b * b + c * c + d * d;
        eulers[1] = std::acos(2 * (a * a + b * b) / n2 - 1);

        const double half_sum = std::atan2(b, a);
        const double half_diff = std::atan2(-d, c);

        const double eps = 1e-7;
        const bool safe1 = std::abs(eulers[1]) >= eps;
        const bool safe2 = std::abs(eulers[1] - M_PI) >= eps;
        bool safe = safe1 && safe2;

        if (safe) {
            eulers[0] = half_sum + half_diff;
            eulers[2] = half_sum - half_diff;
        } else {
            if (!extrinsic) {
                eulers[0] = 0;
                eulers[2] = !safe1 ? 2 * half_sum : -2 * half_diff;
            } else {
                eulers[2] = 0;
                eulers[0] = !safe1 ? 2 * half_sum : 2 * half_diff;
            }
        }

        for (int idx = 0; idx < 3; idx++)
            eulers[idx] = limit_rad(eulers[idx]);

        if (!is_proper) {
            eulers[2] *= sign;
            eulers[1] -= M_PI / 2;
        }

        if (!extrinsic)
            std::swap(eulers[0], eulers[2]);

        return eulers;
    }

    Eigen::Quaterniond
    eulerToQuat(const Eigen::Vector3d& euler, int axis0, int axis1, int axis2, bool extrinsic) {
        const double rz = euler[0], ry = euler[1], rx = euler[2];
        const Eigen::Quaterniond qx(Eigen::AngleAxisd(rx, Eigen::Vector3d::UnitX()));
        const Eigen::Quaterniond qy(Eigen::AngleAxisd(ry, Eigen::Vector3d::UnitY()));
        const Eigen::Quaterniond qz(Eigen::AngleAxisd(rz, Eigen::Vector3d::UnitZ()));

        if (!extrinsic)
            std::swap(axis0, axis2);
        Eigen::Quaterniond q;

        // 生成四元数
        if (axis0 == 0 && axis1 == 1 && axis2 == 2)
            q = qx * qy * qz;
        else if (axis0 == 0 && axis1 == 2 && axis2 == 1)
            q = qx * qz * qy;
        else if (axis0 == 1 && axis1 == 0 && axis2 == 2)
            q = qy * qx * qz;
        else if (axis0 == 1 && axis1 == 2 && axis2 == 0)
            q = qy * qz * qx;
        else if (axis0 == 2 && axis1 == 0 && axis2 == 1)
            q = qz * qx * qy;
        else if (axis0 == 2 && axis1 == 1 && axis2 == 0)
            q = qz * qy * qx;
        else
            throw std::invalid_argument("Unsupported axis order");

        return q;
    }

    Eigen::Matrix3d
    eulerToMatrix(const Eigen::Vector3d& euler, int axis0, int axis1, int axis2, bool extrinsic) {
        return eulerToQuat(euler, axis0, axis1, axis2, extrinsic).toRotationMatrix();
    }

    Eigen::Vector3d
    matrixToEuler(const Eigen::Matrix3d& R, int axis0, int axis1, int axis2, bool extrinsic) {
        Eigen::Quaterniond q(R);
        return quatToEuler(q, axis0, axis1, axis2, extrinsic);
    }

    Eigen::Vector3d quatToEuler(const Eigen::Quaterniond& q, EulerOrder order, bool extrinsic) {
        switch (order) {
            case EulerOrder::XYZ:
                return quatToEuler(q, 0, 1, 2, extrinsic);
            case EulerOrder::XZY:
                return quatToEuler(q, 0, 2, 1, extrinsic);
            case EulerOrder::YXZ:
                return quatToEuler(q, 1, 0, 2, extrinsic);
            case EulerOrder::YZX:
                return quatToEuler(q, 1, 2, 0, extrinsic);
            case EulerOrder::ZXY:
                return quatToEuler(q, 2, 0, 1, extrinsic);
            case EulerOrder::ZYX:
                return quatToEuler(q, 2, 1, 0, extrinsic);
            default:
                throw std::invalid_argument("Unsupported EulerOrder");
        }
    }

    Eigen::Quaterniond eulerToQuat(const Eigen::Vector3d& euler, EulerOrder order, bool extrinsic) {
        switch (order) {
            case EulerOrder::XYZ:
                return eulerToQuat(euler, 0, 1, 2, extrinsic);
            case EulerOrder::XZY:
                return eulerToQuat(euler, 0, 2, 1, extrinsic);
            case EulerOrder::YXZ:
                return eulerToQuat(euler, 1, 0, 2, extrinsic);
            case EulerOrder::YZX:
                return eulerToQuat(euler, 1, 2, 0, extrinsic);
            case EulerOrder::ZXY:
                return eulerToQuat(euler, 2, 0, 1, extrinsic);
            case EulerOrder::ZYX:
                return eulerToQuat(euler, 2, 1, 0, extrinsic);
            default:
                throw std::invalid_argument("Unsupported EulerOrder");
        }
    }

    Eigen::Matrix3d eulerToMatrix(const Eigen::Vector3d& euler, EulerOrder order, bool extrinsic) {
        return eulerToQuat(euler, order, extrinsic).toRotationMatrix();
    }

    Eigen::Vector3d matrixToEuler(const Eigen::Matrix3d& R, EulerOrder order, bool extrinsic) {
        return quatToEuler(Eigen::Quaterniond(R), order, extrinsic);
    }

    Eigen::MatrixXd cvToEigen(const cv::Mat& cv_mat) noexcept {
        Eigen::MatrixXd eigen_mat = Eigen::MatrixXd::Zero(cv_mat.rows, cv_mat.cols);
        cv::cv2eigen(cv_mat, eigen_mat);
        return eigen_mat;
    }
    /// 将相机坐标系下的位置转换到 odom 坐标系
    Eigen::Vector3d transformPosition(
        const Eigen::Vector3d& pos_camera,
        const Eigen::Matrix4d& T_camera_to_odom
    ) noexcept {
        Eigen::Vector4d pos_homo(pos_camera.x(), pos_camera.y(), pos_camera.z(), 1.0);
        Eigen::Vector4d pos_odom = T_camera_to_odom * pos_homo;
        return pos_odom.head<3>();
    }

    /// 将相机坐标系下的姿态转换到 odom 坐标系
    Eigen::Quaterniond transformOrientation(
        const Eigen::Quaterniond& q_camera,
        const Eigen::Matrix4d& T_camera_to_odom
    ) noexcept {
        Eigen::Matrix3d R_camera_to_odom = T_camera_to_odom.block<3, 3>(0, 0);
        Eigen::Matrix3d R_ori_camera = q_camera.normalized().toRotationMatrix();
        Eigen::Matrix3d R_ori_odom = R_camera_to_odom * R_ori_camera;
        return Eigen::Quaterniond(R_ori_odom).normalized();
    }
    void pnpToEigen(
        const cv::Mat& rvec,
        const cv::Mat& tvec,
        Eigen::Vector3d& t_out,
        Eigen::Quaterniond& q_out
    ) noexcept {
        // 平移
        cv::cv2eigen(tvec, t_out);

        // 旋转
        cv::Mat R_cv;
        cv::Rodrigues(rvec, R_cv);
        Eigen::Matrix3d R;
        cv::cv2eigen(R_cv, R);

        q_out = Eigen::Quaterniond(R).normalized();
    }
    void pnpToEigen(
        const cv::Vec3d& rvec,
        const cv::Vec3d& tvec,
        Eigen::Vector3d& t_out,
        Eigen::Quaterniond& q_out
    ) noexcept {
        // 平移
        t_out = Eigen::Vector3d(tvec[0], tvec[1], tvec[2]);

        // Rodrigues 旋转向量转矩阵
        cv::Mat rvec_mat(rvec); // cv::Vec3d -> 3x1 Mat
        cv::Mat R_cv;
        cv::Rodrigues(rvec_mat, R_cv);

        Eigen::Matrix3d R;
        cv::cv2eigen(R_cv, R);

        q_out = Eigen::Quaterniond(R).normalized();
    }

    cv::Point2f computeCenter(const std::vector<cv::Point2f>& points) noexcept {
        if (points.empty()) {
            return cv::Point2f(0.f, 0.f);
        }

        float sum_x = 0.f;
        float sum_y = 0.f;
        for (const auto& pt: points) {
            sum_x += pt.x;
            sum_y += pt.y;
        }
        return cv::Point2f(sum_x / points.size(), sum_y / points.size());
    }
    bool isStateValid(const Eigen::VectorXd& state) noexcept {
        return state.allFinite(); // 所有元素都不是 NaN 或 Inf
    }
    Eigen::Matrix4d computeCameraToOdomTransform(
        const Eigen::Matrix3d& R_gimbal2odom,
        const Eigen::Matrix3d& R_camera_to_gimbal,
        const Eigen::Vector3d& t_camera_to_gimbal
    ) noexcept {
        Eigen::Matrix4d T_gimbal_to_odom = Eigen::Matrix4d::Identity();
        T_gimbal_to_odom.block<3, 3>(0, 0) = R_gimbal2odom;

        Eigen::Matrix4d T_camera_to_gimbal = Eigen::Matrix4d::Identity();
        T_camera_to_gimbal.block<3, 3>(0, 0) = R_camera_to_gimbal;
        T_camera_to_gimbal.block<3, 1>(0, 3) = t_camera_to_gimbal;

        const Eigen::Matrix4d T_camera_to_odom = T_gimbal_to_odom * T_camera_to_gimbal;

        return T_camera_to_odom;
    }

    void addVelFromAccDt(Eigen::Vector3d& vel, const Eigen::Vector3d& acc, double dt) noexcept {
        vel.x() += acc.x() * dt;
        vel.y() += acc.y() * dt;
        vel.z() += acc.z() * dt;
    }
    void addPosFromVelDt(Eigen::Vector3d& pos, const Eigen::Vector3d& vel, double dt) noexcept {
        pos.x() += vel.x() * dt;
        pos.y() += vel.y() * dt;
        pos.z() += vel.z() * dt;
    }
    template<typename T>
    bool tryGetValue(const YAML::Node& node, const char* key, T& out_val) {
        if (!node[key]) {
            return false;
        }
        try {
            out_val = node[key].template as<T>();
            return true;
        } catch (...) {
            return false;
        }
    }
    void changeFileOwner(const std::string& filepath, const std::string& username) {
        struct passwd* pwd = getpwnam(username.c_str());
        if (pwd == nullptr) {
            perror("getpwnam failed");
            return;
        }
        const uid_t uid = pwd->pw_uid;
        const gid_t gid = pwd->pw_gid;

        if (chown(filepath.c_str(), uid, gid) != 0) {
            perror("chown failed");
        }
    }
    std::string getOriginalUsername() {
        const char* sudo_user = std::getenv("SUDO_USER");
        if (sudo_user) {
            return std::string(sudo_user);
        }
        const uid_t uid = getuid();
        struct passwd* pw = getpwuid(uid);
        if (pw) {
            return std::string(pw->pw_name);
        }
        return "";
    }
    bool setThreadAffinityAndPriority(
        std::thread& thread,
        int cpu_id,
        int priority,
        bool use_sched_fifo
    ) {
#ifdef __linux__
        pthread_t native = thread.native_handle();

        cpu_set_t cpuset;
        CPU_ZERO(&cpuset);
        CPU_SET(cpu_id, &cpuset);
        if (pthread_setaffinity_np(native, sizeof(cpu_set_t), &cpuset) != 0) {
            perror("pthread_setaffinity_np failed");
            return false;
        }

        sched_param sch_params;
        sch_params.sched_priority = priority;
        int policy = use_sched_fifo ? SCHED_FIFO : SCHED_RR;
        if (pthread_setschedparam(native, policy, &sch_params) != 0) {
            perror("pthread_setschedparam failed");
            return false;
        }

        return true;

#elif defined(_WIN32) || defined(_WIN64)
        HANDLE native = (HANDLE)thread.native_handle();

        DWORD_PTR affinityMask = 1ULL << cpu_id;
        if (SetThreadAffinityMask(native, affinityMask) == 0) {
            return false;
        }

        int win_priority = THREAD_PRIORITY_HIGHEST; // you can map `priority` if needed
        if (!SetThreadPriority(native, win_priority)) {
            return false;
        }

        return true;

#else
        // Unsupported platform
        (void)thread;
        (void)cpu_id;
        (void)priority;
        (void)use_sched_fifo;
        return false;
#endif
    }
    double rad2deg(double rad) noexcept {
        return rad * 180.0 / M_PI;
    }
    double deg2rad(double deg) noexcept {
        return deg * M_PI / 180.0;
    }

    std::tuple<double, double, double> xyz2ypd_rad(double x, double y, double z) noexcept {
        const double distance = std::sqrt(x * x + y * y + z * z);
        const double yaw = std::atan2(y, x);
        const double pitch = std::atan2(z, std::sqrt(x * x + y * y));
        return std::make_tuple(yaw, pitch, distance);
    }

    std::tuple<double, double, double>
    ypd2xyz_rad(double yaw, double pitch, double distance) noexcept {
        const double x = distance * std::cos(pitch) * std::cos(yaw);
        const double y = distance * std::cos(pitch) * std::sin(yaw);
        const double z = distance * std::sin(pitch);
        return std::make_tuple(x, y, z);
    }

    std::tuple<double, double, double> xyz2ypd_deg(double x, double y, double z) noexcept {
        const double distance = std::sqrt(x * x + y * y + z * z);
        const double yaw = std::atan2(y, x);
        const double pitch = std::atan2(z, std::sqrt(x * x + y * y));
        return std::make_tuple(rad2deg(yaw), rad2deg(pitch), distance);
    }

    std::tuple<double, double, double>
    ypd2xyz_deg(double yaw_deg, double pitch_deg, double distance) noexcept {
        const double yaw = deg2rad(yaw_deg);
        const double pitch = deg2rad(pitch_deg);
        const double x = distance * std::cos(pitch) * std::cos(yaw);
        const double y = distance * std::cos(pitch) * std::sin(yaw);
        const double z = distance * std::sin(pitch);
        return std::make_tuple(x, y, z);
    }
    Eigen::Vector3d xyz2ypd(const Eigen::Vector3d& xyz) noexcept {
        const auto x = xyz[0], y = xyz[1], z = xyz[2];
        const auto yaw = std::atan2(y, x);
        const auto pitch = std::atan2(z, std::sqrt(x * x + y * y));
        const auto distance = std::sqrt(x * x + y * y + z * z);
        return { yaw, pitch, distance };
    }
    template<typename Point>
    Point getCenter(const std::vector<Point>& points) noexcept {
        if (points.empty())
            return Point();
        return std::accumulate(points.begin(), points.end(), Point())
            / static_cast<float>(points.size());
    }
    bool segmentIntersection(
        const cv::Point2f& a1,
        const cv::Point2f& a2,
        const cv::Point2f& b1,
        const cv::Point2f& b2,
        cv::Point2f& intersection
    ) {
        const cv::Point2f r = a2 - a1;
        const cv::Point2f s = b2 - b1;
        const float rxs = r.x * s.y - r.y * s.x;
        const float qpxr = (b1 - a1).x * r.y - (b1 - a1).y * r.x;

        if (fabs(rxs) < 1e-6)
            return false; // 平行或重叠，无唯一交点

        const float t = ((b1 - a1).x * s.y - (b1 - a1).y * s.x) / rxs;
        const float u = qpxr / rxs;

        if (t >= 0 && t <= 1 && u >= 0 && u <= 1) {
            intersection = a1 + t * r;
            return true;
        }
        return false;
    }

    /// 主函数：计算线段与 RotatedRect 的交点
    std::vector<cv::Point2f> intersectLineRotatedRect(
        const cv::RotatedRect& rect,
        const cv::Point2f& line_p1,
        const cv::Point2f& line_p2
    ) {
        std::vector<cv::Point2f> intersections;

        // 1. 获取旋转矩形四个角点
        cv::Point2f vertices[4];
        rect.points(vertices);

        // 2. 遍历矩形四条边，检测交点
        for (int i = 0; i < 4; i++) {
            cv::Point2f p1 = vertices[i];
            cv::Point2f p2 = vertices[(i + 1) % 4];
            cv::Point2f inter;

            if (segmentIntersection(line_p1, line_p2, p1, p2, inter)) {
                intersections.push_back(inter);
            }
        }

        return intersections;
    }
    std::string makeTimestampedFileName() {
        using namespace std::chrono;
        auto now = system_clock::now();
        auto time_t_now = system_clock::to_time_t(now);
        auto ms = duration_cast<milliseconds>(now.time_since_epoch()) % 1000;

        std::tm tm_now {};
#if defined(_MSC_VER)
        localtime_s(&tm_now, &time_t_now);
#else
        localtime_r(&time_t_now, &tm_now);
#endif

        std::ostringstream oss;
        oss << std::put_time(&tm_now, "%Y%m%d_%H%M%S") << "_" << std::setfill('0') << std::setw(3)
            << ms.count();
        return oss.str();
    }

    std::string expandEnv(const std::string& s) {
        std::regex env_re(R"(\$\{([^}]+)\})");
        std::smatch match;
        std::string result = s;
        while (std::regex_search(result, match, env_re)) {
            const char* env = std::getenv(match[1].str().c_str());
            std::string val = env ? env : "";
            result.replace(match.position(0), match.length(0), val);
        }
        return result;
    }
    double computeBrightness(const cv::Mat& frame) {
        cv::Mat gray;
        cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);
        return cv::mean(gray)[0];
    }
    cv::Mat letterbox(
        const cv::Mat& img,
        Eigen::Matrix3f& transform_matrix,
        const int new_shape_w,
        const int new_shape_h
    ) noexcept {
        const int img_h = img.rows;
        const int img_w = img.cols;

        const float scale = std::min((float)new_shape_h / img_h, (float)new_shape_w / img_w);
        const int resize_h = int(img_h * scale + 0.5f);
        const int resize_w = int(img_w * scale + 0.5f);

        const int pad_h = new_shape_h - resize_h;
        const int pad_w = new_shape_w - resize_w;
        const int top = pad_h / 2;
        const int left = pad_w / 2;

        cv::Mat resized;
        cv::resize(img, resized, cv::Size(resize_w, resize_h), 0, 0, cv::INTER_LINEAR);

        cv::Mat out;
        cv::copyMakeBorder(
            resized,
            out,
            top,
            pad_h - top,
            left,
            pad_w - left,
            cv::BORDER_CONSTANT,
            cv::Scalar(114, 114, 114)
        );

        const float inv_scale = 1.0f / scale;

        transform_matrix << inv_scale, 0, -left * inv_scale, 0, inv_scale, -top * inv_scale, 0, 0,
            1;

        return out;
    }

} // namespace utils
} // namespace wust_vision