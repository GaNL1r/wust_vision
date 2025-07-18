#include "common/camera_calibrator.hpp"
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <opencv2/imgcodecs.hpp>
#include <yaml-cpp/emitter.h>
#include <yaml-cpp/emittermanip.h>
using namespace cv;

CameraCalibrator::CameraCalibrator(
    Size boardSize,
    float squareSize,
    int requiredViews,
    float minShift
):
    boardSize_(boardSize),
    squareSize_(squareSize),
    requiredViews_(requiredViews),
    minCornerShift_(minShift) {}

std::vector<Point3f> CameraCalibrator::generate3DPoints() const {
    std::vector<Point3f> objp;
    for (int i = 0; i < boardSize_.height; ++i)
        for (int j = 0; j < boardSize_.width; ++j)
            objp.emplace_back(j * squareSize_, i * squareSize_, 0);
    return objp;
}

bool CameraCalibrator::isDifferent(const std::vector<Point2f>& c1, const std::vector<Point2f>& c2)
    const {
    if (c1.empty() || c2.empty() || c1.size() != c2.size())
        return true;
    double sum = 0;
    for (size_t i = 0; i < c1.size(); ++i)
        sum += norm(c1[i] - c2[i]);
    return (sum / c1.size()) > minCornerShift_;
}

void CameraCalibrator::updateCoverage(const std::vector<Point2f>& corners, Size size) {
    auto w = size.width, h = size.height;
    Point2f center(w / 2.f, h / 2.f);
    for (auto& pt: corners) {
        if (pt.x < center.x && pt.y < center.y)
            coverageFlags_[0] = true;
        if (pt.x > center.x && pt.y < center.y)
            coverageFlags_[1] = true;
        if (pt.x < center.x && pt.y > center.y)
            coverageFlags_[2] = true;
        if (pt.x > center.x && pt.y > center.y)
            coverageFlags_[3] = true;
        if (norm(pt - center) < std::min(w, h) * 0.1f)
            coverageFlags_[4] = true;
    }
}

void CameraCalibrator::calibrate(const Size& imageSize) {
    std::vector<Mat> rvecs, tvecs;
    calibrateCamera(
        objectPoints_,
        imagePoints_,
        imageSize,
        cameraMatrix_,
        distCoeffs_,
        rvecs,
        tvecs
    );

    double totalError = 0;
    int totalPoints = 0;
    for (size_t i = 0; i < objectPoints_.size(); ++i) {
        std::vector<Point2f> projected;
        projectPoints(objectPoints_[i], rvecs[i], tvecs[i], cameraMatrix_, distCoeffs_, projected);
        totalError += norm(projected, imagePoints_[i], NORM_L2SQR);
        totalPoints += projected.size();
    }
    double rmse = std::sqrt(totalError / totalPoints);
    std::cout << "[✓] 标定完成，重投影误差 RMSE = " << rmse << " pixels" << std::endl;
    calibrated_ = true;
}

bool CameraCalibrator::processFrame(const Mat& frame) {
    if (calibrated_)
        return false;
    imageSize_.width = frame.cols;
    imageSize_.height = frame.rows;

    Mat vis = frame.clone();
    std::vector<Point2f> corners;
    Mat gray;
    cvtColor(frame, gray, COLOR_BGR2GRAY);
    bool found = findChessboardCornersSB(
        gray,
        boardSize_,
        corners,
        cv::CALIB_CB_NORMALIZE_IMAGE | cv::CALIB_CB_EXHAUSTIVE
    );

    if (found) {
        cornerSubPix(
            gray,
            corners,
            { 11, 11 },
            { -1, -1 },
            TermCriteria(TermCriteria::EPS + TermCriteria::MAX_ITER, 30, 0.01)
        );

        if (isDifferent(corners, lastCorners_)) {
            updateCoverage(corners, gray.size());
            imagePoints_.push_back(corners);
            objectPoints_.push_back(generate3DPoints());
            lastCorners_ = corners;

            std::cout << "    当前采样数: " << imagePoints_.size() << "/" << requiredViews_
                      << std::endl;
            std::cout << "    分布覆盖 (UL,UR,LL,LR,CENTER): ";
            for (auto f: coverageFlags_)
                std::cout << (f ? "[✓]" : "[×]") << " ";
            std::cout << std::endl;
        }

        // +++ 添加角点编号和详细评估信息 +++
        if (!corners.empty()) {
            // 计算角点间距统计信息
            std::vector<double> distX, distY;
            for (int i = 0; i < boardSize_.height; i++) {
                for (int j = 0; j < boardSize_.width - 1; j++) {
                    int idx = i * boardSize_.width + j;
                    distX.push_back(norm(corners[idx] - corners[idx + 1]));
                }
            }
            for (int i = 0; i < boardSize_.height - 1; i++) {
                for (int j = 0; j < boardSize_.width; j++) {
                    int idx = i * boardSize_.width + j;
                    distY.push_back(norm(corners[idx] - corners[idx + boardSize_.width]));
                }
            }

            double minDist = std::min(
                *min_element(distX.begin(), distX.end()),
                *min_element(distY.begin(), distY.end())
            );
            double maxDist = std::max(
                *max_element(distX.begin(), distX.end()),
                *max_element(distY.begin(), distY.end())
            );
            double avgDist = (mean(distX)[0] + mean(distY)[0]) / 2.0;

            // 在图像上绘制角点编号和间距信息
            for (size_t i = 0; i < corners.size(); ++i) {
                putText(
                    vis,
                    std::to_string(i),
                    corners[i],
                    FONT_HERSHEY_PLAIN,
                    1.0,
                    Scalar(255, 0, 0),
                    1
                );

                // 在第一个角点显示间距统计
                if (i == 0) {
                    std::ostringstream oss;
                    oss << "Size: " << std::fixed << std::setprecision(1) << avgDist
                        << "px (min:" << minDist << " max:" << maxDist << ")";
                    putText(
                        vis,
                        oss.str(),
                        corners[i] + Point2f(0, -20),
                        FONT_HERSHEY_SIMPLEX,
                        0.5,
                        Scalar(0, 200, 255),
                        1
                    );
                }
            }

            // +++ 实时估算当前误差 +++
            if (imagePoints_.size() >= 3) {
                std::vector<Mat> rvecs, tvecs;
                Mat tmpK = Mat::eye(3, 3, CV_64F);
                Mat tmpD = Mat::zeros(1, 5, CV_64F);
                calibrateCamera(objectPoints_, imagePoints_, gray.size(), tmpK, tmpD, rvecs, tvecs);

                std::vector<Point2f> projected;
                projectPoints(
                    generate3DPoints(),
                    rvecs.back(),
                    tvecs.back(),
                    tmpK,
                    tmpD,
                    projected
                );

                // 计算详细的误差统计
                double totalErr = 0;
                double maxErr = 0;
                double errX = 0, errY = 0;
                std::vector<double> errors;

                for (size_t i = 0; i < corners.size(); ++i) {
                    double dx = projected[i].x - corners[i].x;
                    double dy = projected[i].y - corners[i].y;
                    double err = std::sqrt(dx * dx + dy * dy);

                    errors.push_back(err);
                    totalErr += err;
                    errX += std::abs(dx);
                    errY += std::abs(dy);

                    if (err > maxErr)
                        maxErr = err;

                    // 在误差较大的角点上绘制误差向量
                    if (err > 1.0) {
                        arrowedLine(vis, corners[i], projected[i], Scalar(0, 0, 255), 2);
                    }
                }

                double meanErr = totalErr / corners.size();
                errX /= corners.size();
                errY /= corners.size();

                // 计算误差分布
                std::sort(errors.begin(), errors.end());
                double medianErr = errors[errors.size() / 2];
                double p95Err = errors[errors.size() * 95 / 100];

                // 在图像上显示详细误差信息
                std::ostringstream oss;
                oss << "RMSE: " << std::fixed << std::setprecision(2) << meanErr << "px "
                    << "(X:" << errX << " Y:" << errY << ")";
                putText(
                    vis,
                    oss.str(),
                    Point(10, vis.rows - 60),
                    FONT_HERSHEY_SIMPLEX,
                    0.6,
                    Scalar(0, 255, 255),
                    2
                );

                oss.str("");
                oss << "Max: " << std::fixed << std::setprecision(2) << maxErr << "px "
                    << "P95: " << p95Err << "px";
                putText(
                    vis,
                    oss.str(),
                    Point(10, vis.rows - 30),
                    FONT_HERSHEY_SIMPLEX,
                    0.6,
                    Scalar(0, 255, 255),
                    2
                );

                // 在控制台输出详细统计信息
                std::cout << "    Error Stats: mean=" << meanErr << "px, "
                          << "median=" << medianErr << "px, p95=" << p95Err << "px, "
                          << "max=" << maxErr << "px\n";
                std::cout << "    Axis Errors: X=" << errX << "px, Y=" << errY << "px\n";
            }
        }
    }

    if (!corners.empty())
        drawChessboardCorners(vis, boardSize_, corners, true);

    // 缩小图像显示
    Mat show_img;
    resize(vis, show_img, Size(vis.cols / 2, vis.rows / 2));

    // 添加状态信息
    putText(
        show_img,
        "Press 'c' to calibrate, ESC to quit",
        Point(10, 30),
        FONT_HERSHEY_SIMPLEX,
        0.7,
        Scalar(0, 255, 0),
        2
    );

    std::ostringstream status;
    status << "Views: " << imagePoints_.size() << "/" << requiredViews_ << "  | Coverage: ";
    for (bool flag: coverageFlags_)
        status << (flag ? "✓ " : "× ");
    putText(
        show_img,
        status.str(),
        Point(10, 60),
        FONT_HERSHEY_SIMPLEX,
        0.5,
        Scalar(0, 255, 255),
        1
    );

    imshow("Camera Calibration", show_img);
    int key = waitKey(1);
    if (key == 27) {
        std::cout << "用户退出采集。" << std::endl;
        return false;
    }

    if (key == 'c') {
        if (imagePoints_.size() < requiredViews_) {
            std::cout << "[!] 样本不足，至少需要 " << requiredViews_ << " 张" << std::endl;
        } else if (!std::all_of(coverageFlags_.begin(), coverageFlags_.end(), [](bool v) {
                       return v;
                   })) {
            std::cout << "[!] 图像分布不足，采集更多不同角度图像" << std::endl;
        } else {
            calibrate(gray.size());
            std::cout << "标定完成！" << std::endl;
        }
    }

    return true;
}

void CameraCalibrator::saveToFile(const std::filesystem::path& filename) const {
    if (!calibrated_) {
        std::cerr << "尚未完成标定，无法保存参数。" << std::endl;
        return;
    }

    YAML::Emitter out;
    out << YAML::BeginMap;
    out << YAML::Key << "image_width" << YAML::Value << imageSize_.width;
    out << YAML::Key << "image_height" << YAML::Value << imageSize_.height;
    out << YAML::Key << "camera_name" << YAML::Value << "narrow_stereo";

    out << YAML::Key << "camera_matrix" << YAML::Value << YAML::BeginMap;
    out << YAML::Key << "rows" << YAML::Value << 3;
    out << YAML::Key << "cols" << YAML::Value << 3;
    out << YAML::Key << "data" << YAML::Value << YAML::Flow << YAML::BeginSeq;
    for (int i = 0; i < 9; ++i)
        out << cameraMatrix_.at<double>(i / 3, i % 3);
    out << YAML::EndSeq << YAML::EndMap;

    out << YAML::Key << "distortion_model" << YAML::Value << "plumb_bob";
    out << YAML::Key << "distortion_coefficients" << YAML::Value << YAML::BeginMap;
    out << YAML::Key << "rows" << YAML::Value << 1;
    out << YAML::Key << "cols" << YAML::Value << (int)distCoeffs_.cols;
    out << YAML::Key << "data" << YAML::Value << YAML::Flow << YAML::BeginSeq;
    for (int i = 0; i < distCoeffs_.cols; ++i)
        out << distCoeffs_.at<double>(0, i);
    out << YAML::EndSeq << YAML::EndMap;

    out << YAML::Key << "rectification_matrix" << YAML::Value << YAML::BeginMap;
    out << YAML::Key << "rows" << YAML::Value << 3;
    out << YAML::Key << "cols" << YAML::Value << 3;
    out << YAML::Key << "data" << YAML::Value << YAML::Flow << YAML::BeginSeq;
    for (int i = 0; i < 9; ++i)
        out << (i % 4 == 0 ? 1.0 : 0.0);
    out << YAML::EndSeq << YAML::EndMap;

    out << YAML::Key << "projection_matrix" << YAML::Value << YAML::BeginMap;
    out << YAML::Key << "rows" << YAML::Value << 3;
    out << YAML::Key << "cols" << YAML::Value << 4;
    out << YAML::Key << "data" << YAML::Value << YAML::Flow << YAML::BeginSeq;
    out << cameraMatrix_.at<double>(0, 0) << 0 << cameraMatrix_.at<double>(0, 2) << 0 << 0
        << cameraMatrix_.at<double>(1, 1) << cameraMatrix_.at<double>(1, 2) << 0 << 0 << 0 << 1
        << 0;
    out << YAML::EndSeq << YAML::EndMap;

    out << YAML::EndMap;

    std::ofstream fout(filename);
    fout << out.c_str();
    std::cout << "标定参数已保存为 YAML 至: " << filename << std::endl;
}
