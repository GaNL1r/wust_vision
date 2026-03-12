#include "rune_detector.hpp"
#include "tasks/utils.hpp"
namespace wust_vision {
namespace auto_buff {
    struct RuneDetectorCV::Impl {
    public:
        Impl(const YAML::Node& node) {
            params_.load(node);
        }

        void setCallback(DetectorCallback callback) {
            callback_ = callback;
        }
        cv::Mat preProcess(const cv::Mat& src, bool use_red) {
            cv::Mat bin;
            cv::cvtColor(src, bin, cv::COLOR_RGB2GRAY);
            cv::threshold(bin, bin, params_.bin_threshold, 255, cv::THRESH_BINARY);
            return bin;
        }
        inline auto_buff::RuneCenter getRuneCenter(
            const std::vector<std::vector<cv::Point>>& contours,
            const std::vector<cv::Vec4i>& hierarchy,
            cv::Size image_size,
            cv::Point2f offset,
            cv::Mat& debug_img,
            std::vector<bool>& used_flags
        ) {
            auto_buff::RuneCenter result;
            struct Node {
                cv::Point2f center;
                int idx;
                cv::RotatedRect rr;
            };

            std::vector<Node> nodes;

            for (int i = 0; i < contours.size(); i++) {
                if (used_flags[i])
                    continue;
                if (hierarchy[i][3] != -1)
                    continue;

                double area = cv::contourArea(contours[i]);
                if (area < params_.rune_center_min_area || area > params_.rune_center_max_area)
                    continue;

                cv::RotatedRect rr = cv::minAreaRect(contours[i]);
                float w = rr.size.width;
                float h = rr.size.height;

                if (w < 5 || h < 5)
                    continue;

                double ratio = (w > h ? w / h : h / w);
                if (ratio - 1.0 > params_.rune_center_1x1ratio_tol)
                    continue;

                double rect_area = w * h;
                if (rect_area <= 1e-5)
                    continue;

                double fill_ratio = area / rect_area;
                if (fill_ratio < params_.rune_center_fill_ratio_min)
                    continue;

                nodes.push_back({ rr.center, i, rr });

                if (!debug_img.empty()) {
                    cv::Point2f pts[4];
                    rr.points(pts);
                    for (size_t k = 0; k < 4; k++) {
                        pts[k] += offset;
                    }
                    for (int k = 0; k < 4; k++) {
                        cv::line(debug_img, pts[k], pts[(k + 1) % 4], cv::Scalar(0, 255, 0), 2);
                    }
                }
            }

            if (nodes.empty())
                return result;

            cv::Point2f img_center(image_size.width * 0.5f, image_size.height * 0.5f);

            double best_dist = 1e18;
            int best_idx = -1;
            cv::RotatedRect best_rr;

            for (auto& n: nodes) {
                double dx = n.center.x - img_center.x;
                double dy = n.center.y - img_center.y;
                double dist2 = dx * dx + dy * dy;

                if (dist2 < best_dist) {
                    best_dist = dist2;
                    best_idx = n.idx;
                    best_rr = n.rr;
                }
            }

            if (!debug_img.empty()) {
                cv::circle(
                    debug_img,
                    img_center + offset,
                    5,
                    cv::Scalar(0, 255, 255),
                    -1
                ); // 图像中心

                cv::Point2f pts[4];
                best_rr.points(pts);
                for (size_t k = 0; k < 4; k++) {
                    pts[k] += offset;
                }
                for (int k = 0; k < 4; k++) {
                    cv::line(debug_img, pts[k], pts[(k + 1) % 4], cv::Scalar(0, 0, 255), 2);
                }
            }

            return auto_buff::RuneCenter(best_rr);
        }

        inline int findTopParent(int idx, const std::vector<cv::Vec4i>& hierarchy) {
            int p = hierarchy[idx][3]; // parent
            while (p != -1 && hierarchy[p][3] != -1) {
                p = hierarchy[p][3]; // 一直追溯到最顶层 parent
            }
            return p; // 若 p == -1 表示 contour 本身就是顶层轮廓
        }

        inline std::vector<auto_buff::RunePan> markRuneTarget(
            const std::vector<std::vector<cv::Point>>& contours,
            const std::vector<cv::Vec4i>& hierarchy,
            std::vector<bool>& used_flags
        ) {
            std::vector<auto_buff::RunePan> results;
            if (hierarchy.empty())
                return results;

            struct Node {
                int idx;
                cv::Point2f center;
                int parent_top_id;
            };

            std::vector<Node> candidates;

            for (int i = 0; i < contours.size(); i++) {
                if (used_flags[i])
                    continue;

                const auto& cnt = contours[i];

                double contour_area = cv::contourArea(cnt);
                if (contour_area < params_.rune_target_min_area
                    || contour_area > params_.rune_target_max_area)
                    continue;

                cv::Moments m = cv::moments(cnt);
                if (m.m00 == 0)
                    continue;

                cv::Point2f center(m.m10 / m.m00, m.m01 / m.m00);
                int top_parent = findTopParent(i, hierarchy);
                candidates.push_back({ i, center, top_parent });
            }

            if (candidates.size() < 3)
                return results;

            std::unordered_map<int, std::vector<int>> groups;
            for (int i = 0; i < candidates.size(); i++) {
                groups[candidates[i].parent_top_id].push_back(i);
            }

            for (auto& [parent_top_id, idx_list]: groups) {
                int M = idx_list.size();
                if (M < 3 || M > 7)
                    continue;

                std::vector<int> cluster_id(M, -1);
                int cluster_count = 0;

                for (int i = 0; i < M; i++) {
                    if (cluster_id[i] != -1)
                        continue;

                    cluster_id[i] = cluster_count;

                    std::queue<int> q;
                    q.push(i);

                    while (!q.empty()) {
                        int u = q.front();
                        q.pop();

                        for (int v = 0; v < M; v++) {
                            if (cluster_id[v] != -1)
                                continue;

                            auto& cu = candidates[idx_list[u]].center;
                            auto& cv = candidates[idx_list[v]].center;

                            double dx = cu.x - cv.x;
                            double dy = cu.y - cv.y;
                            double dist = std::sqrt(dx * dx + dy * dy);

                            if (dist <= params_.rune_target_cluster_radius) {
                                cluster_id[v] = cluster_count;
                                q.push(v);
                            }
                        }
                    }
                    cluster_count++;
                }

                std::vector<int> cluster_size(cluster_count, 0);
                for (int id: cluster_id)
                    cluster_size[id]++;

                std::vector<std::vector<cv::Point2f>> cluster_points(cluster_count);

                for (int i = 0; i < M; i++) {
                    int cid = cluster_id[i];

                    if (cluster_size[cid] >= 3) {
                        int contour_index = candidates[idx_list[i]].idx;
                        used_flags[contour_index] = true;
                        cluster_points[cid].push_back(candidates[idx_list[i]].center);
                    }
                }

                for (int cid = 0; cid < cluster_count; cid++) {
                    if (cluster_points[cid].size() < 3)
                        continue;

                    cv::RotatedRect rr = cv::minAreaRect(cluster_points[cid]);
                    double w = rr.size.width;
                    double h = rr.size.height;

                    if (w < 1 || h < 1)
                        continue;

                    double ratio = (w > h ? w / h : h / w);
                    if (ratio > params_.rune_target_max_square_ratio)
                        continue;

                    std::vector<std::pair<double, cv::Point2f>> dist_list;
                    dist_list.reserve(cluster_points[cid].size());

                    for (auto& p: cluster_points[cid]) {
                        double dx = p.x - rr.center.x;
                        double dy = p.y - rr.center.y;
                        double dist = dx * dx + dy * dy;
                        dist_list.emplace_back(dist, p);
                    }

                    std::sort(dist_list.begin(), dist_list.end(), [](auto& a, auto& b) {
                        return a.first > b.first;
                    });

                    std::vector<cv::Point2f> corner_points;
                    for (int i = 0; i < 4 && i < dist_list.size(); i++)
                        corner_points.push_back(dist_list[i].second);

                    auto_buff::RunePan pan;
                    pan.center = rr.center;
                    pan.corners = corner_points;
                    if (corner_points.size() > 3)
                        pan.is_valid = true;

                    results.push_back(pan);
                }
            }

            return results;
        }

        inline void markInvalidContours(
            cv::Mat& color,
            cv::Mat& debug_img,
            const std::vector<std::vector<cv::Point>>& contours,
            std::vector<bool>& used_flags,
            const cv::Rect& valid_rect,
            bool filter_red,
            double diff_thresh
        ) {
            used_flags.assign(contours.size(), false);

            for (int i = 0; i < contours.size(); i++) {
                cv::Rect r = cv::boundingRect(contours[i]);
                if (r.width < 5 || r.height < 5)
                    continue;

                cv::Rect rr = r & cv::Rect(0, 0, color.cols, color.rows);
                if (rr.width < 2 || rr.height < 2)
                    continue;

                const cv::Mat roi = color(rr);
                const cv::Scalar avg = cv::mean(roi);

                const double B = avg[0], G = avg[1], R = avg[2];

                const double diff_RB = R - B;
                const double diff_BR = B - R;

                const bool is_red = (diff_RB > diff_thresh);
                const bool is_blue = (diff_BR > diff_thresh);

                bool invalid = false;

                if (filter_red) {
                    if (is_red)
                        invalid = true;
                } else {
                    if (is_blue)
                        invalid = true;
                }
                cv::Rect inter = r & valid_rect;
                bool inside_region = (inter.area() > 0);

                used_flags[i] = !invalid || !inside_region;

                if (!used_flags[i]) {
                    if (!debug_img.empty())
                        cv::drawContours(debug_img, contours, i, cv::Scalar(255, 0, 0), 2);
                }
            }
        }
        static bool isUpscaled(const cv::Rect& roi, int model_w, int model_h) {
            float scale = std::min(model_w / float(roi.width), model_h / float(roi.height));

            return scale > 1.0f;
        }
        void pushInput(CommonFrame& frame, bool is_big, bool debug) {
            frame.id = current_id_++;
            auto_buff::RuneFan fan {
                .is_valid = false,
                .id = frame.id,
                .is_big = is_big,
                .timestamp = frame.img_frame.timestamp,

            };
            cv::Mat debug_img;
            if (debug) {
                debug_img = frame.img_frame.src_img.clone();
            }
            cv::Mat roi = frame.img_frame.src_img(frame.expanded);

            cv::Mat processed_img = preProcess(roi, frame.detect_color);

            std::vector<std::vector<cv::Point>> contours;
            std::vector<cv::Vec4i> hierarchy;

            cv::findContours(
                processed_img,
                contours,
                hierarchy,
                cv::RETR_TREE,
                cv::CHAIN_APPROX_SIMPLE
            );
            std::vector<bool> used_flags;
            used_flags.assign(contours.size(), false);
            markInvalidContours(
                roi,
                debug_img,
                contours,
                used_flags,
                cv::Rect(0, 0, roi.cols, roi.rows),
                frame.detect_color,
                params_.color_diff_threshold
            );
            auto rune_center =
                getRuneCenter(contours, hierarchy, roi.size(), frame.offset, debug_img, used_flags);
            std::vector<auto_buff::RunePan> rune_pans =
                markRuneTarget(contours, hierarchy, used_flags);
            for (auto& rune_pan: rune_pans) {
                if (rune_center.is_valid) {
                    rune_pan.addReferRuneCenter(rune_center);
                }
                if (rune_pan.is_valid && rune_pan.has_refer) {
                    auto_buff::RuneFan::Simple simple;
                    simple.points2d.push_back(rune_center.center);
                    for (auto& pt: rune_pan.corners) {
                        simple.points2d.push_back(pt);
                    }
                    simple.points2d.push_back(rune_pan.center);
                    fan.fans.push_back(simple);
                }
                if (!debug_img.empty())
                    rune_pan.draw(debug_img, frame.offset);
            }
            auto_buff::RuneFan tmp = fan;
            for (int i = 0; i < tmp.fans.size(); i++) {
                for (int j = 0; j < tmp.fans.size(); j++) {
                    if (i == j)
                        continue;

                    fan.fans[i].addOther(tmp.fans[j]);
                }
            }
            fan.addOffset(frame.offset);
            if (callback_) {
                callback_(fan, frame, debug_img);
            }
        }

        DetectorCallback callback_;
        cv::Mat tmp_R_;
        int current_id_ = 0;
        struct Params {
            double rune_center_min_area = 100.0;
            double rune_center_max_area = 2000.0;
            double rune_center_1x1ratio_tol = 0.7;
            double rune_center_fill_ratio_min = 0.7;

            double rune_target_min_area = 100.0;
            double rune_target_max_area = 3000.0;
            double rune_target_max_square_ratio = 1.3;
            double rune_target_cluster_radius = 70.0;

            double bin_threshold = 150.0;
            double color_diff_threshold = 40.0;

            int target_width = 416;
            int target_height = 416;

            void load(const YAML::Node& node) {
                // center params
                rune_center_min_area = node["rune_center_min_area"]
                    ? node["rune_center_min_area"].as<double>()
                    : rune_center_min_area;
                rune_center_max_area = node["rune_center_max_area"]
                    ? node["rune_center_max_area"].as<double>()
                    : rune_center_max_area;
                rune_center_1x1ratio_tol = node["rune_center_1x1ratio_tol"]
                    ? node["rune_center_1x1ratio_tol"].as<double>()
                    : rune_center_1x1ratio_tol;
                rune_center_fill_ratio_min = node["rune_center_fill_ratio_min"]
                    ? node["rune_center_fill_ratio_min"].as<double>()
                    : rune_center_fill_ratio_min;

                // target params
                rune_target_min_area = node["rune_target_min_area"]
                    ? node["rune_target_min_area"].as<double>()
                    : rune_target_min_area;
                rune_target_max_area = node["rune_target_max_area"]
                    ? node["rune_target_max_area"].as<double>()
                    : rune_target_max_area;
                rune_target_max_square_ratio = node["rune_target_max_square_ratio"]
                    ? node["rune_target_max_square_ratio"].as<double>()
                    : rune_target_max_square_ratio;
                rune_target_cluster_radius = node["rune_target_cluster_radius"]
                    ? node["rune_target_cluster_radius"].as<double>()
                    : rune_target_cluster_radius;

                bin_threshold =
                    node["bin_threshold"] ? node["bin_threshold"].as<double>() : bin_threshold;

                color_diff_threshold = node["color_diff_threshold"]
                    ? node["color_diff_threshold"].as<double>()
                    : color_diff_threshold;

                target_width = node["target_width"] ? node["target_width"].as<int>() : target_width;
                target_height =
                    node["target_height"] ? node["target_height"].as<int>() : target_height;
            }
        } params_;
    };
    RuneDetectorCV::RuneDetectorCV(const YAML::Node& node) {
        _impl = std::make_unique<Impl>(node);
    }
    RuneDetectorCV::~RuneDetectorCV() {
        _impl.reset();
    }
    void RuneDetectorCV::pushInput(CommonFrame& frame, bool is_big, bool debug) {
        _impl->pushInput(frame, is_big, debug);
    }
    void RuneDetectorCV::setCallback(DetectorCallback callback) {
        _impl->setCallback(callback);
    }
} // namespace auto_buff
} // namespace wust_vision