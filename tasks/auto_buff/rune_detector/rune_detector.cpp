#include "rune_detector.hpp"
RuneDetectorCV::RuneDetectorCV(const YAML::Node& node) {
    params_.load(node);
}
cv::Mat RuneDetectorCV::preProcess(const cv::Mat& src, bool use_red) {
    cv::Mat bin;
    cv::cvtColor(src, bin, cv::COLOR_RGB2GRAY);
    cv::threshold(bin, bin, params_.bin_threshold, 255, cv::THRESH_BINARY);

    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3));

    // cv::dilate(bin, bin, kernel, cv::Point(-1, -1), 1);

    // cv::morphologyEx(bin, bin, cv::MORPH_CLOSE, kernel, cv::Point(-1, -1), 1);
    // cv::morphologyEx(bin, bin, cv::MORPH_OPEN, kernel, cv::Point(-1, -1), 1);
    return bin;
}
// cv::Mat RuneDetectorCV::preProcess(const cv::Mat& src, bool use_red) {
//     cv::Mat channel, bin;

//
//     if (use_red) {
//
//         cv::extractChannel(src, channel, 2); // 2 = R
//     } else {
//
//         cv::extractChannel(src, channel, 0); // 0 = B
//     }

//
//     cv::threshold(channel, bin, 50, 255, cv::THRESH_BINARY);

//
//     cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3));

//
//     cv::dilate(bin, bin, kernel, cv::Point(-1, -1), 1);
//     cv::morphologyEx(bin, bin, cv::MORPH_CLOSE, kernel, cv::Point(-1, -1), 1);
//     cv::morphologyEx(bin, bin, cv::MORPH_OPEN, kernel, cv::Point(-1, -1), 1);

//     return bin;
// }

inline rune::RuneCenter RuneDetectorCV::getRuneCenter(
    const std::vector<std::vector<cv::Point>>& contours,
    const std::vector<cv::Vec4i>& hierarchy,
    cv::Mat& debug_img,
    std::vector<bool>& used_flags
) {
    rune::RuneCenter result;

    auto inSearchROI = [](const cv::Point2f& pt, const cv::Point2f& ctr, float half) {
        return fabs(pt.x - ctr.x) <= half && fabs(pt.y - ctr.y) <= half;
    };

    if (!debug_img.empty() && last_center.x > 0 && last_center.y > 0) {
        cv::Point2f tl(last_center.x - search_half_size, last_center.y - search_half_size);
        cv::Point2f br(last_center.x + search_half_size, last_center.y + search_half_size);
        cv::rectangle(debug_img, tl, br, cv::Scalar(255, 0, 0), 2); // 蓝色 ROI
    }

    struct Node {
        cv::Point2f center;
        int idx;
        cv::RotatedRect rr;
    };

    std::vector<Node> nodes;
    nodes.reserve(20);

    for (int i = 0; i < contours.size(); i++) {
        if (used_flags[i])
            continue;
        if (hierarchy[i][3] != -1)
            continue;

        if (contours[i].empty())
            continue;

        cv::RotatedRect rr = cv::minAreaRect(contours[i]);
        cv::Point2f center = rr.center;

        if (last_center.x > 0 && last_center.y > 0) {
            if (!inSearchROI(center, last_center, search_half_size))
                continue;
        }

        double area = cv::contourArea(contours[i]);
        if (area < params_.rune_center_min_area || area > params_.rune_center_max_area)
            continue;

        float w = rr.size.width;
        float h = rr.size.height;
        if (w < 5 || h < 5)
            continue;

        double ratio = (w > h) ? w / h : h / w;
        if (ratio - 1.0 > params_.rune_center_1x1ratio_tol)
            continue;

        double rect_area = w * h;
        if (rect_area <= 1e-5)
            continue;

        double fill_ratio = area / rect_area;
        if (fill_ratio < params_.rune_center_fill_ratio_min)
            continue;

        nodes.push_back({ center, i, rr });

        if (!debug_img.empty()) {
            cv::Point2f pts[4];
            rr.points(pts);
            for (int k = 0; k < 4; k++)
                cv::line(debug_img, pts[k], pts[(k + 1) % 4], cv::Scalar(0, 255, 0), 2);
        }
    }

    if (nodes.empty()) {
        search_half_size += 50;
        if (search_half_size > SEARCH_MAX)
            search_half_size = SEARCH_MAX;

        return result;
    }

    cv::Point2f global_center(0, 0);
    for (auto& n: nodes) {
        global_center.x += n.center.x;
        global_center.y += n.center.y;
    }
    global_center.x /= nodes.size();
    global_center.y /= nodes.size();

    double best_dist = 1e18;
    cv::RotatedRect best_rr;
    int best_idx = -1;

    for (auto& n: nodes) {
        double dx, dy, dist2;

        // ---- 优先使用 ROI 中心（last_center） ----
        if (last_center.x > 0 && last_center.y > 0) {
            dx = n.center.x - last_center.x;
            dy = n.center.y - last_center.y;
        } else {
            // ---- 第一帧或没有上一次数据 → fallback 到原来 global_center ----
            dx = n.center.x - global_center.x;
            dy = n.center.y - global_center.y;
        }

        dist2 = dx * dx + dy * dy;

        if (dist2 < best_dist) {
            best_dist = dist2;
            best_idx = n.idx;
            best_rr = n.rr;
        }
    }

    if (!debug_img.empty()) {
        cv::circle(debug_img, global_center, 4, cv::Scalar(0, 255, 255), -1);

        cv::Point2f pts[4];
        best_rr.points(pts);
        for (int k = 0; k < 4; k++)
            cv::line(debug_img, pts[k], pts[(k + 1) % 4], cv::Scalar(0, 0, 255), 2);
    }

    if (best_idx != -1) {
        last_center = best_rr.center;
        cv::Rect2f br = best_rr.boundingRect();

        float bbox_half = std::max(br.width, br.height) * 0.5f;
        SEARCH_MIN = bbox_half * 2.0f;

        if (SEARCH_MIN < SEARCH_MIN_REAL)
            SEARCH_MIN = SEARCH_MIN_REAL;

        search_half_size -= 30;
        if (search_half_size < SEARCH_MIN)
            search_half_size = SEARCH_MIN;

        if (search_half_size > SEARCH_MAX)
            search_half_size = SEARCH_MAX;

        return rune::RuneCenter(best_rr);
    }

    search_half_size += 50;
    if (search_half_size > SEARCH_MAX)
        search_half_size = SEARCH_MAX;

    return result;
}

inline int findTopParent(int idx, const std::vector<cv::Vec4i>& hierarchy) {
    int p = hierarchy[idx][3]; // parent
    while (p != -1 && hierarchy[p][3] != -1) {
        p = hierarchy[p][3]; // 一直追溯到最顶层 parent
    }
    return p; // 若 p == -1 表示 contour 本身就是顶层轮廓
}

inline std::vector<rune::RunePan> RuneDetectorCV::markRuneTarget(
    const std::vector<std::vector<cv::Point>>& contours,
    const std::vector<cv::Vec4i>& hierarchy,
    std::vector<bool>& used_flags
) {
    std::vector<rune::RunePan> results;
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

            rune::RunePan pan;
            pan.center = rr.center;
            pan.corners = corner_points;
            if (corner_points.size() > 3)
                pan.is_valid = true;

            results.push_back(pan);
        }
    }

    return results;
}

inline void RuneDetectorCV::markInvalidContours(
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

        cv::Mat roi = color(rr);
        cv::Scalar avg = cv::mean(roi);

        double B = avg[0], G = avg[1], R = avg[2];

        double diff_RB = R - B;
        double diff_BR = B - R;

        bool is_red = (diff_RB > diff_thresh);
        bool is_blue = (diff_BR > diff_thresh);

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

        used_flags[i] = !invalid||!inside_region;

        if (!used_flags[i]) {
            if (!debug_img.empty())
                cv::drawContours(debug_img, contours, i, cv::Scalar(255, 0, 0), 2);
        }
    }
} 

void RuneDetectorCV::pushInput(CommonFrame& frame, bool is_big) {
    frame.id = current_id_++;
    rune::RuneFan fan { .is_valid = false,
                        .timestamp = frame.timestamp,
                        .id = frame.id,
                        .is_big = is_big };

    cv::Mat processed_img = preProcess(frame.src_img, frame.detect_color);
    cv::Mat debug_img;
    debug_img = frame.src_img.clone();

    std::vector<std::vector<cv::Point>> contours;
    std::vector<cv::Vec4i> hierarchy;

    cv::findContours(processed_img, contours, hierarchy, cv::RETR_TREE, cv::CHAIN_APPROX_SIMPLE);
    std::vector<bool> used_flags;
    used_flags.assign(contours.size(), false);
    markInvalidContours(frame.src_img, debug_img, contours, used_flags,frame.expanded, frame.detect_color);
    auto rune_center = getRuneCenter(contours, hierarchy, debug_img, used_flags);
    std::vector<rune::RunePan> rune_pans = markRuneTarget(contours, hierarchy, used_flags);
    double avg_pan_area = 0.0;
    for (auto& rune_pan: rune_pans) {
        if (rune_center.is_valid) {
            rune_pan.addReferRuneCenter(rune_center);
        }
        if (rune_pan.is_valid && rune_pan.has_refer) {
            rune::RuneFan::Simple simple;
            simple.points2d.push_back(rune_center.center);
            for (auto& pt: rune_pan.corners) {
                simple.points2d.push_back(pt);
            }
            simple.points2d.push_back(rune_pan.center);
            fan.fans.push_back(simple);
        }
        if (!debug_img.empty())
            rune_pan.draw(debug_img);
    }
    rune::RuneFan tmp = fan;
    for (int i = 0; i < tmp.fans.size(); i++) {
        for (int j = 0; j < tmp.fans.size(); j++) {
            if (i == j)
                continue;

            fan.fans[i].addOther(tmp.fans[j]);
        }
    }

    if (callback_) {
        // cv::Mat img_copy = processed_img.clone();
        callback_(fan, frame, debug_img);
    }
}