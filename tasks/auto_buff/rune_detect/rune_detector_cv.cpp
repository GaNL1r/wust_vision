
#include "rune_detector_cv.hpp"
#include <numeric>
#include <vector>
#include <execution>
#include <mutex>
#include <cmath>

// ---------------- 图像预处理 ----------------
cv::Mat preprocessROI(const cv::Mat& input_roi) {
    cv::Mat roi;
    input_roi.copyTo(roi);
    if (roi.channels() > 1)
        cv::cvtColor(roi, roi, cv::COLOR_BGR2GRAY);

    cv::threshold(roi, roi, 100, 255, cv::THRESH_BINARY);
    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(3, 3));
    cv::morphologyEx(roi, roi, cv::MORPH_CLOSE, kernel, cv::Point2f(-1, -1), 2);
    return roi;
}

// ---------------- 霍夫圆检测 ----------------
cv::Vec3f detectLargestCircle(const cv::Mat& roi) {
    std::vector<cv::Vec3f> circles;
    cv::HoughCircles(roi, circles, cv::HOUGH_GRADIENT, 1, roi.rows / 8, 40, 20, 5, 50);
    if (circles.empty())
        return cv::Vec3f(0, 0, 0);

    cv::Vec3f max_circle = circles[0];
    for (size_t i = 1; i < circles.size(); i++)
        if (circles[i][2] > max_circle[2])
            max_circle = circles[i];

    
    return max_circle;
}

cv::Vec3f detectAdaptiveCircle(
    const cv::Mat& roi, 
    const cv::Mat& template_circle,
    double min_match_score = 0.2,
    double coarse_angle_step = 45.0,
    double fine_angle_range = 10.0,
    double fine_angle_step = 5.0,
    double scale_min = 0.5,
    double scale_max = 1.5,
    double scale_step = 0.1
)
{
    if (roi.empty() || template_circle.empty())
        return cv::Vec3f(0,0,0);

    // -------- 1. 使用 HoughCircles 检测粗圆 --------
    cv::Vec3f coarse_circle = detectLargestCircle(roi);
    if (coarse_circle[2] <= 0)
        return cv::Vec3f(0,0,0);

    int cx = static_cast<int>(coarse_circle[0]);
    int cy = static_cast<int>(coarse_circle[1]);
    int r = static_cast<int>(coarse_circle[2]);

    // ROI 扩展 2 倍半径
    int roi_x1 = std::max(cx - 2*r, 0);
    int roi_y1 = std::max(cy - 2*r, 0);
    int roi_x2 = std::min(cx + 2*r, roi.cols - 1);
    int roi_y2 = std::min(cy + 2*r, roi.rows - 1);
    cv::Mat local_roi = roi(cv::Range(roi_y1, roi_y2), cv::Range(roi_x1, roi_x2));

    static double last_angle = 0.0;

    cv::Point best_loc;
    double best_score = -1;
    double best_angle = last_angle;
    double best_scale = 1.0;

    std::mutex mtx;

    // -------- 2. 多尺度模板匹配（粗略 + 并行旋转） --------
    for(double scale = scale_min; scale <= scale_max; scale += scale_step)
    {
        cv::Mat scaled_template;
        cv::resize(template_circle, scaled_template, cv::Size(), scale, scale, cv::INTER_LINEAR);

        // 粗略旋转角度列表
        std::vector<double> angles;
        for(double angle = last_angle; angle < last_angle + 360.0; angle += coarse_angle_step)
            angles.push_back(fmod(angle,360.0));

        std::for_each(std::execution::par, angles.begin(), angles.end(),
            [&](double cur_angle){
                cv::Mat rot_template;
                cv::Mat rot_mat = cv::getRotationMatrix2D(
                    cv::Point2f(scaled_template.cols/2.0f, scaled_template.rows/2.0f),
                    cur_angle, 1.0);
                cv::warpAffine(scaled_template, rot_template, rot_mat, scaled_template.size(),
                               cv::INTER_LINEAR, cv::BORDER_CONSTANT, cv::Scalar(0));

                int result_cols = local_roi.cols - rot_template.cols + 1;
                int result_rows = local_roi.rows - rot_template.rows + 1;
                if(result_cols <= 0 || result_rows <= 0) return;

                cv::Mat result(result_rows, result_cols, CV_32FC1);
                cv::matchTemplate(local_roi, rot_template, result, cv::TM_CCOEFF_NORMED);

                double minVal, maxVal;
                cv::Point minLoc, maxLoc;
                cv::minMaxLoc(result, &minVal, &maxVal, &minLoc, &maxLoc);

                std::lock_guard<std::mutex> lock(mtx);
                if(maxVal > best_score)
                {
                    best_score = maxVal;
                    best_loc = maxLoc;
                    best_angle = cur_angle;
                    best_scale = scale;
                }
            }
        );
    }

    if(best_score < min_match_score)
        return cv::Vec3f(0,0,0);

    last_angle = best_angle;

    // -------- 3. 局部精细优化 --------
    double fine_best_score = best_score;
    cv::Point fine_best_loc = best_loc;
    double fine_best_angle = best_angle;

    std::vector<double> fine_angles;
    for(double angle = best_angle - fine_angle_range; angle <= best_angle + fine_angle_range; angle += fine_angle_step)
        fine_angles.push_back(fmod(angle + 360.0,360.0));

    std::for_each(std::execution::par, fine_angles.begin(), fine_angles.end(),
        [&](double cur_angle){
            cv::Mat scaled_template;
            cv::resize(template_circle, scaled_template, cv::Size(), best_scale, best_scale, cv::INTER_LINEAR);

            cv::Mat rot_template;
            cv::Mat rot_mat = cv::getRotationMatrix2D(
                cv::Point2f(scaled_template.cols/2.0f, scaled_template.rows/2.0f),
                cur_angle, 1.0);
            cv::warpAffine(scaled_template, rot_template, rot_mat, scaled_template.size(),
                           cv::INTER_LINEAR, cv::BORDER_CONSTANT, cv::Scalar(0));

            int result_cols = local_roi.cols - rot_template.cols + 1;
            int result_rows = local_roi.rows - rot_template.rows + 1;
            if(result_cols <= 0 || result_rows <= 0) return;

            cv::Mat result(result_rows, result_cols, CV_32FC1);
            cv::matchTemplate(local_roi, rot_template, result, cv::TM_CCOEFF_NORMED);

            double minVal, maxVal;
            cv::Point minLoc, maxLoc;
            cv::minMaxLoc(result, &minVal, &maxVal, &minLoc, &maxLoc);

            std::lock_guard<std::mutex> lock(mtx);
            if(maxVal > fine_best_score)
            {
                fine_best_score = maxVal;
                fine_best_loc = maxLoc;
                fine_best_angle = cur_angle;
            }
        }
    );

    last_angle = fine_best_angle;

    // -------- 4. 计算最终圆心与半径 --------
    int final_cx = fine_best_loc.x + roi_x1 + static_cast<int>(template_circle.cols*best_scale/2);
    int final_cy = fine_best_loc.y + roi_y1 + static_cast<int>(template_circle.rows*best_scale/2);
    float final_radius = template_circle.cols/2.0f * best_scale;

    return cv::Vec3f(final_cx, final_cy, final_radius);
}







// ---------------- r_tag 修正 ----------------
std::tuple<cv::Point2f, cv::Size2f, float>
refineRTag(const cv::Mat& roi, const cv::Point2f& circle_center, const cv::Point2f& r_tag_guess) {
    cv::Mat mask = roi.clone();
    if (mask.channels() != 1) {
        cv::cvtColor(mask, mask, cv::COLOR_BGR2GRAY);
    }
    cv::threshold(mask, mask, 100, 255, cv::THRESH_BINARY);

    double r_radius = cv::norm(r_tag_guess - circle_center) * 15.0 / 65.0;
    cv::Mat circle_mask = cv::Mat::zeros(mask.size(), CV_8UC1);
    cv::circle(circle_mask, r_tag_guess, static_cast<int>(r_radius), cv::Scalar(255), -1);
    cv::bitwise_and(mask, circle_mask, mask);

    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(mask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_NONE);

    cv::Point2f best_r_tag = r_tag_guess;
    cv::Size2f best_size(0, 0);
    float best_angle = 0.0f;
    double best_score = std::numeric_limits<double>::infinity();

    for (auto& cnt: contours) {
        if (cnt.size() < 10)
            continue;

        auto rect = cv::minAreaRect(cnt);
        double ratio = rect.size.height > rect.size.width ? rect.size.height / rect.size.width
                                                          : rect.size.width / rect.size.height;

        if (ratio < best_score) {
            best_score = ratio;
            best_r_tag = rect.center;
            best_size = rect.size;
            best_angle = rect.angle;
        }
    }

    return std::make_tuple(best_r_tag, best_size, best_angle);
}

// ---------------- 射线端点检测 ----------------
std::vector<cv::Point2f>
detectRayEndpoints(const cv::Mat& roi, const cv::Point2f& center, const cv::Point2f& r_tag,double max_dist) {
    std::vector<cv::Point2f> endpoints;
    cv::Point2f dir = r_tag - center;
    std::vector<cv::Point2f> directions = { dir,
                                            cv::Point2f(-dir.y, dir.x),
                                            -dir,
                                            cv::Point2f(dir.y, -dir.x) };
    double r_tag_dist = cv::norm(r_tag - center);
    double min_dist = r_tag_dist * 5.0 / 65.0;

    for (auto d: directions) {
        double len = std::sqrt(d.x * d.x + d.y * d.y);
        if (len == 0)
            len = 1.0;
        cv::Point2f unit(d.x / len, d.y / len);

        cv::Point2f best_pt = center;  // 默认没有找到亮点时返回中心

        // 从外向内扫描
        for (double r = max_dist; r >= min_dist; r--) {
            int x = cvRound(center.x + unit.x * r);
            int y = cvRound(center.y + unit.y * r);

            // 防止越界
            if (x < 0 || x >= roi.cols || y < 0 || y >= roi.rows)
                continue;

            if (roi.at<uchar>(y, x) > 20) {  // 找到亮点
                best_pt = cv::Point2f(x, y);
                break;  // 找到最远亮点后立即退出
            }
        }
        endpoints.push_back(best_pt);
    }
    return endpoints;
}


// ---------------- 圆形合理性检查 ----------------
bool likeCircleCheckAndFix(std::vector<cv::Point2f>& pts, const cv::Point2f& center) {
    if (pts.size() != 4)
        return false;

    std::vector<double> dists, angles;
    for (auto& pt: pts) {
        dists.push_back(cv::norm(pt - center));
        angles.push_back(std::atan2(pt.y - center.y, pt.x - center.x));
    }

    double mean_dist = std::accumulate(dists.begin(), dists.end(), 0.0) / dists.size();

    // 找出偏差最大点
    int bad_idx = -1;
    double max_dev = 0;
    for (int i = 0; i < 4; i++) {
        double dev = std::abs(dists[i] - mean_dist);
        if (dev > max_dev) {
            max_dev = dev;
            bad_idx = i;
        }
    }

    // 如果偏差太大，就修正坏点
    if (bad_idx != -1 && max_dev > 0.2 * mean_dist) {
        // 找相对点（角度相差最接近 π 的点）
        int opp_idx = -1;
        double best_angle_diff = 1e9;
        for (int i = 0; i < 4; i++) {
            if (i == bad_idx)
                continue;
            double diff = std::fabs(std::fabs(angles[i] - angles[bad_idx]) - M_PI);
            if (diff < best_angle_diff) {
                best_angle_diff = diff;
                opp_idx = i;
            }
        }

        if (opp_idx != -1) {
            // 相对点
            cv::Point2f opp = pts[opp_idx];
            // 中心对称点：p' = 2*center - opp
            cv::Point2f corrected = 2 * center - opp;
            pts[bad_idx] = corrected;
        }
    }

    return true;
}

cv::Mat warpImg(const cv::Mat& input_img, const rune::RuneFan& result) {
    if (result.fan_kpoints.size() != 4)
        return input_img;

    const int warp_height = 1000;
    const int warp_width = 360;
    const int r_base = 50;
    const int r2center = 650;
    const int radius = 150;
    // 只取 4 个点 (例如 r_tag 上方，左右，底部)
    std::vector<cv::Point2f> fan_vertices = { result.fan_kpoints[0],
                                              result.fan_kpoints[3],
                                              result.fan_kpoints[1],
                                              result.fan_kpoints[2] };

    std::vector<cv::Point2f> target_vertices = {
        cv::Point2f(warp_width / 2.0, r_base + r2center - radius),
        cv::Point2f(warp_width / 2.0 - radius, r_base + r2center),
        cv::Point2f(warp_width / 2.0 + radius, r_base + r2center),
        cv::Point2f(warp_width / 2.0, r_base + r2center + radius)
    };

    cv::Mat warp_matrix = cv::getPerspectiveTransform(fan_vertices, target_vertices);
    cv::Mat warp_img;
    cv::warpPerspective(input_img, warp_img, warp_matrix, cv::Size(warp_width, warp_height));

    cv::imshow("warp_img", warp_img);
    cv::waitKey(1);
    return warp_img;
}

std::tuple<cv::Point2f, cv::Size2f, float> detectOuterRect(
    const cv::Mat& roi,
    const cv::Point2f& circle_center,
    const cv::Point2f& r_tag,
    const double radius,
    float min_ratio = 15.0f / 65.0f,
    float max_ratio = 55.0f / 65.0f
) {
    cv::Mat mask;
    if (roi.channels() != 1)
        cv::cvtColor(roi, mask, cv::COLOR_BGR2GRAY);
    else
        mask = roi.clone();

    // ---------------- CR 方向 ----------------
    cv::Point2f CR = r_tag - circle_center;
    float CR_len = cv::norm(CR);
    if (CR_len == 0)
        return std::make_tuple(r_tag, cv::Size2f(0, 0), 0.0f);
    cv::Point2f CR_unit = CR / CR_len;

    // ---------------- 两个边界点 ----------------
    float r_min = CR_len * min_ratio;
    float r_max = CR_len * max_ratio;

    cv::Point2f inner_pt = circle_center + CR_unit * r_min;
    cv::Point2f outer_pt = circle_center + CR_unit * r_max;

    // ---------------- 构造狭长矩形 ROI ----------------
    // CR 垂直方向
    cv::Point2f CR_perp(-CR_unit.y, CR_unit.x);
    float half_width = radius;

    std::vector<cv::Point> poly = { inner_pt + CR_perp * half_width,
                                    inner_pt - CR_perp * half_width,
                                    outer_pt - CR_perp * half_width,
                                    outer_pt + CR_perp * half_width };

    cv::Mat region_mask = cv::Mat::zeros(mask.size(), CV_8UC1);
    cv::fillConvexPoly(region_mask, poly, 255);

    // ---------------- 与原图相交，只保留中间部分 ----------------
    cv::bitwise_and(mask, region_mask, mask);

    // ---------------- 开闭运算 ----------------
    int colse_kernel_size = 15;
    int open_kernel_size = 5;
    cv::morphologyEx(
        mask,
        mask,
        cv::MORPH_OPEN,
        cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(open_kernel_size, open_kernel_size))
    );
    cv::morphologyEx(
        mask,
        mask,
        cv::MORPH_CLOSE,
        cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(colse_kernel_size, colse_kernel_size))
    );

    // ---------------- 查找轮廓 ----------------
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(mask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_NONE);

    cv::Point2f best_center = r_tag;
    cv::Size2f best_size(0, 0);
    float best_angle = 0;
    double max_area = 0;

    for (auto& cnt: contours) {
        if (cnt.size() < 10)
            continue;

        std::vector<cv::Point> hull;
        cv::convexHull(cnt, hull);
        if (hull.size() < 5)
            continue;

        auto rect = cv::minAreaRect(hull);
        cv::Size2f size = rect.size;
        float angle = rect.angle;
        float rect_long_axis_angle = size.width > size.height ? angle : angle + 90.0f;

        float CR_angle = std::atan2(CR_unit.y, CR_unit.x) * 180.0f / CV_PI;
        float diff_angle = std::fabs(rect_long_axis_angle - CR_angle);
        if (diff_angle > 90.0f)
            diff_angle = 180.0f - diff_angle;

        if (diff_angle > 45.0f)
            continue;

        double contour_area = cv::contourArea(cnt);
        double rect_area = size.width * size.height;
        if (rect_area < 1e-3)
            continue;

        if (rect_area > max_area) {
            max_area = rect_area;
            best_center = rect.center;
            best_size = size;
            best_angle = rect.angle;
        }
    }

    return std::make_tuple(best_center, best_size, best_angle);
}

// ---------------- 主函数 ----------------
rune::RuneFan
RuneDetectorCV::detect(const cv::Mat& input_roi, const cv::Point2f& r_tag, cv::Mat& debug_img,bool debug) {
    rune::RuneFan result;
    cv::Mat roi = preprocessROI(input_roi);
    
    cv::Vec3f max_circle = detectAdaptiveCircle(roi,template_mat);
    if (max_circle[2] == 0)
        return result;

    result.fan_center = cv::Point2f(cvRound(max_circle[0]), cvRound(max_circle[1]));
    result.radius = cvRound(max_circle[2]);
    cv::Size2f rect_size;
    float angle;

    std::tie(result.r_tag, rect_size, angle) = refineRTag(roi, result.fan_center, r_tag);

    result.fan_kpoints = detectRayEndpoints(roi, result.fan_center, result.r_tag,result.radius);

    bool isCircleLike = likeCircleCheckAndFix(result.fan_kpoints, result.fan_center);
    result.is_valid = isCircleLike;
    cv::Point2f outer_center;
    cv::Size2f outer_size;
    float outer_angle;

    std::tie(outer_center, outer_size, outer_angle) =
        detectOuterRect(roi, result.fan_center, result.r_tag, result.radius);
    result.mid_rect = cv::RotatedRect(outer_center, outer_size, outer_angle);
    if (debug) {
    cv::Mat roi_color;
    cv::cvtColor(roi, roi_color, cv::COLOR_GRAY2BGR);
    cv::circle(roi_color, result.fan_center, result.radius, cv::Scalar(0, 255, 0), 2);
    cv::circle(roi_color, result.fan_center, 2, cv::Scalar(0, 0, 255), 3);
    cv::line(roi_color, result.fan_center, result.r_tag, cv::Scalar(0, 255, 255), 2);
    for (auto& pt: result.fan_kpoints) {
        cv::line(roi_color, result.fan_center, pt, cv::Scalar(255, 0, 0), 1);
        cv::circle(roi_color, pt, 3, cv::Scalar(0, 0, 255), -1);
    }
    cv::putText(
        roi_color,
        isCircleLike ? "Valid Circle" : "Invalid Circle",
        cv::Point2f(10, 20),
        cv::FONT_HERSHEY_SIMPLEX,
        0.6,
        isCircleLike ? cv::Scalar(0, 255, 0) : cv::Scalar(0, 0, 255),
        2
    );

    cv::Point2f pts[4];
    result.mid_rect.points(pts);
    for (int i = 0; i < 4; i++)
        cv::line(roi_color, pts[i], pts[(i + 1) % 4], cv::Scalar(0, 255, 255), 2);

    // ===== 将 roi_color 覆盖到 debug_img 的右下角 =====
    if (!debug_img.empty() &&
        debug_img.cols >= roi_color.cols &&
        debug_img.rows >= roi_color.rows) 
    {
        cv::Rect roi_dst(
            debug_img.cols - roi_color.cols,  // x 起点
            debug_img.rows - roi_color.rows,  // y 起点
            roi_color.cols,
            roi_color.rows
        );
        roi_color.copyTo(debug_img(roi_dst));
    }
}


    return result;
}
