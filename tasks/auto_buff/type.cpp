#include "type.hpp"
#include "tasks/utils/utils.hpp"
namespace wust_vision {
namespace auto_buff {
    void RunePan::draw(cv::Mat& img, const cv::Point2f& offset) const {
        if (!is_valid || corners.size() < 3)
            return;

        std::vector<cv::Point2f> sorted_corners = corners;
        for (auto& pt: sorted_corners) {
            pt += offset;
        }
        // 画边
        for (size_t i = 0; i < sorted_corners.size(); ++i) {
            cv::line(
                img,
                sorted_corners[i],
                sorted_corners[(i + 1) % sorted_corners.size()],
                cv::Scalar(0, 255, 255),
                2
            );
        }

        // 画中心点
        cv::circle(img, center, 3, cv::Scalar(255, 0, 255), -1);
        if (has_refer) {
            // 画角点编号
            for (size_t i = 0; i < sorted_corners.size(); ++i) {
                cv::Point2f p = sorted_corners[i];

                // 绘制角点位置
                cv::circle(img, p, 3, cv::Scalar(0, 0, 255), -1);

                // 让数字稍微往右下偏移，避免盖到角点
                cv::Point2f text_pos = p + cv::Point2f(5, -5);

                cv::putText(
                    img,
                    std::to_string(i),
                    text_pos,
                    cv::FONT_HERSHEY_SIMPLEX,
                    0.5,
                    cv::Scalar(0, 255, 0),
                    1
                );
            }
        }
    }
    double RunePan::getArea() const {
        if (corners.size() < 3)
            return 0.0;
        std::vector<cv::Point2f> sorted_corners = corners;
        std::sort(
            sorted_corners.begin(),
            sorted_corners.end(),
            [this](const cv::Point2f& a, const cv::Point2f& b) {
                double angA = std::atan2(a.y - center.y, a.x - center.x);
                double angB = std::atan2(b.y - center.y, b.x - center.x);
                return angA < angB;
            }
        );
        return cv::contourArea(sorted_corners);
    }

    void RunePan::addReferRuneCenter(const RuneCenter& rc) {
        if (!rc.is_valid || !is_valid)
            return;
        if (corners.size() != 4)
            return;

        cv::Point2f down_vec = rc.center - center;
        float norm = std::sqrt(down_vec.x * down_vec.x + down_vec.y * down_vec.y);
        if (norm < 1e-6f)
            return;
        has_refer = true;
        float angle_ref = std::atan2(down_vec.y, down_vec.x);

        // 获取4个点在旋转后的角度
        struct Node {
            float ang;
            cv::Point2f p;
        };
        std::vector<Node> arr;
        arr.reserve(4);

        for (auto& p: corners) {
            cv::Point2f v = p - center;

            // 旋转坐标，使 down_vec 对齐 angle=0
            float ang = std::atan2(v.y, v.x) - angle_ref;

            // 归一化到 (-π, π]
            while (ang <= -CV_PI)
                ang += 2 * CV_PI;
            while (ang > CV_PI)
                ang -= 2 * CV_PI;

            arr.push_back({ ang, p });
        }

        // 按角度排序（从 -π 到 π）
        std::sort(arr.begin(), arr.end(), [](const Node& a, const Node& b) {
            return a.ang < b.ang;
        });

        // 准备象限变量并标记
        cv::Point2f lu(0, 0), ru(0, 0), rd(0, 0), ld(0, 0);
        bool has_lu = false, has_ru = false, has_rd = false, has_ld = false;

        for (const auto& n: arr) {
            float a = n.ang;

            if (a > CV_PI / 2 && a <= CV_PI) {
                lu = n.p;
                has_lu = true;
            } else if (a > 0 && a <= CV_PI / 2) {
                ru = n.p;
                has_ru = true;
            } else if (a > -CV_PI / 2 && a <= 0) {
                rd = n.p;
                has_rd = true;
            } else { // a > -CV_PI && a <= -CV_PI/2
                ld = n.p;
                has_ld = true;
            }
        }

        std::array<cv::Point2f, 4> ordered;

        if (has_lu && has_ru && has_rd && has_ld) {
            ordered[0] = lu;
            ordered[1] = ru;
            ordered[2] = rd;
            ordered[3] = ld;
            corners.assign(ordered.begin(), ordered.end());
            return;
        }

        float target = 3.0f * CV_PI / 4.0f; // 135°
        int best_idx = 0;
        float best_diff = std::numeric_limits<float>::max();
        for (int i = 0; i < (int)arr.size(); ++i) {
            float d = std::fabs(angles::shortest_angular_distance(target, arr[i].ang)
            ); // 如果没有 angles::shortest_angular_distance，可以用下面替代
            if (d < best_diff) {
                best_diff = d;
                best_idx = i;
            }
        }

        for (int i = 0; i < 4; ++i) {
            int idx = (best_idx + i) % 4;
            ordered[i] = arr[idx].p;
        }

        corners.assign(ordered.begin(), ordered.end());
    }

    void RuneFan::Simple::addOther(const Simple& other) {
        auto l1 = points2d[0] - points2d[5];
        auto l2 = other.points2d[0] - other.points2d[5];
        float a1 = std::atan2(l1.y, l1.x);
        float a2 = std::atan2(l2.y, l2.x);

        float d = a1 - a2;
        d = normalizeAngle0to2pi(d);

        int id = 0;
        double min_err = 1e9;
        for (int i = 0; i < angle_diffs.size(); i++) {
            double err = std::abs(angle_diffs[i] - d);
            if (err < min_err) {
                min_err = err;
                id = i;
            }
        }

        if (id < 1) {
            return;
        }
        has_other++;
        points2d.push_back(other.points2d[1]);
        points2d.push_back(other.points2d[2]);
        points2d.push_back(other.points2d[3]);
        points2d.push_back(other.points2d[4]);

        double roll = -angle_diffs[id];

        points3d.push_back(rotateX(points3d[1], roll));
        points3d.push_back(rotateX(points3d[2], roll));
        points3d.push_back(rotateX(points3d[3], roll));
        points3d.push_back(rotateX(points3d[4], roll));
    }

    void RuneFan::Simple::drawLandmarks(cv::Mat& image) const {
        std::vector<cv::Point2f> lm = landmarks();
        for (size_t i = 0; i < lm.size(); i++) {
            cv::circle(image, lm[i], 3, cv::Scalar(255, 255, 255), -1);
            if (i == 0) {
                cv::putText(
                    image,
                    "R",
                    lm[i],
                    cv::FONT_HERSHEY_SIMPLEX,
                    1.5,
                    cv::Scalar(40, 255, 40),
                    2
                );
            } else {
                cv::putText(
                    image,
                    std::to_string(i),
                    lm[i],
                    cv::FONT_HERSHEY_SIMPLEX,
                    0.5,
                    cv::Scalar(255, 255, 255),
                    2
                );
            }
        }
    }

    void RuneFan::addOffset(const cv::Point2f& offset) {
        for (auto& fan: fans) {
            for (auto& point: fan.points2d) {
                point += offset;
            }
        }
    }
    void RuneFan::transform(const Eigen::Matrix<float, 3, 3>& transform_matrix) {
        for (auto& fan: fans) {
            for (auto& pt: fan.points2d) {
                pt = utils::transformPoint2D(transform_matrix, pt);
            }
        }
    }

    void PowerRune::Pose::tf(Eigen::Matrix4d T_camera_to_odom) {
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

    std::vector<cv::Point2f> PowerRune::Pose::toPts(
        const cv::Mat& camera_intrinsic,
        const cv::Mat& camera_distortion,
        const std::vector<cv::Point3f>& obj_points
    ) const {
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

        cv::projectPoints(obj_points, rvec, tvec, camera_intrinsic, camera_distortion, pts);

        return pts;
    }
    void PowerRune::Pose::draw(
        cv::Mat& img,
        const cv::Mat& camera_intrinsic,
        const cv::Mat& camera_distortion,
        const std::vector<cv::Point3f>& obj_points,
        cv::Scalar color
    ) const {
        auto pts = toPts(camera_intrinsic, camera_distortion, obj_points);
        if (!pts.empty()) {
            for (int i = 0; i < 4; i++)
                cv::line(img, pts[i], pts[(i + 1) % 4], color, 2);

            // 后表面
            for (int i = 4; i < 8; i++)
                cv::line(img, pts[i], pts[4 + (i + 1) % 4], color, 2);

            // 侧边
            for (int i = 0; i < 4; i++)
                cv::line(img, pts[i], pts[i + 4], color, 2);
            cv::Point2f center(0.f, 0.f);
            for (auto pt: pts) {
                center += pt;
            }
            center *= 1.0 / pts.size();
        }
    }

    void PowerRune::tf(Eigen::Matrix4d T_camera_to_odom) {
        center.tf(T_camera_to_odom);
        for (auto& fan: fans)
            fan.tf(T_camera_to_odom);
    }
    void
    PowerRune::draw(cv::Mat& img, const cv::Mat& camera_intrinsic, const cv::Mat& camera_distortion)
        const {
        center.draw(img, camera_intrinsic, camera_distortion);
        for (int i = 0; i < fans.size(); i++) {
            if (i == hit_id)
                fans[i].draw(
                    img,
                    camera_intrinsic,
                    camera_distortion,
                    FAN_BLOCK,
                    cv::Scalar(40, 255, 40)
                );
            else
                fans[i].draw(img, camera_intrinsic, camera_distortion, FAN_BLOCK);
        }
    }
} // namespace auto_buff
} // namespace wust_vision