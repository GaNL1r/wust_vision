#include "guidance_detector_opencv.hpp"
#include "tasks/utils.hpp"
bool initializing = true;

int lowH = 35, highH = 85;
int lowS = 50, highS = 255;
int lowV = 80, highV = 255;

static void onTrackbar(int, void*) {
    if (initializing)
        return; // 初始化阶段不更新

    lowH = cv::getTrackbarPos("LowH", "mask");
    highH = cv::getTrackbarPos("HighH", "mask");
    lowS = cv::getTrackbarPos("LowS", "mask");
    highS = cv::getTrackbarPos("HighS", "mask");
    lowV = cv::getTrackbarPos("LowV", "mask");
    highV = cv::getTrackbarPos("HighV", "mask");
}

void initGUI() {
    cv::namedWindow("mask", cv::WINDOW_NORMAL); // 允许调整大小
    cv::resizeWindow("mask", 400, 600); // 设置初始窗口大小

    cv::createTrackbar("LowH", "mask", nullptr, 179, onTrackbar);
    cv::createTrackbar("HighH", "mask", nullptr, 179, onTrackbar);
    cv::createTrackbar("LowS", "mask", nullptr, 255, onTrackbar);
    cv::createTrackbar("HighS", "mask", nullptr, 255, onTrackbar);
    cv::createTrackbar("LowV", "mask", nullptr, 255, onTrackbar);
    cv::createTrackbar("HighV", "mask", nullptr, 255, onTrackbar);

    cv::setTrackbarPos("LowH", "mask", lowH);
    cv::setTrackbarPos("HighH", "mask", highH);
    cv::setTrackbarPos("LowS", "mask", lowS);
    cv::setTrackbarPos("HighS", "mask", highS);
    cv::setTrackbarPos("LowV", "mask", lowV);
    cv::setTrackbarPos("HighV", "mask", highV);

    initializing = false;
}

namespace auto_guidance {
GuidanceDetectorOpenCV::GuidanceDetectorOpenCV(const YAML::Node& config_gobal, bool debug) {
    debug_ = debug;
    auto config = config_gobal["opencv"];
    lowH = config["HSV"]["lowH"].as<int>();
    highH = config["HSV"]["highH"].as<int>();
    lowS = config["HSV"]["lowS"].as<int>();
    highS = config["HSV"]["highS"].as<int>();
    highV = config["HSV"]["highV"].as<int>();
    lowV = config["HSV"]["lowV"].as<int>();
    max_area_ = config["contours"]["max_area"].as<double>();
    min_area_ = config["contours"]["min_area"].as<double>();
    min_aspect_ratio = config["contours"]["min_aspect_ratio"].as<double>();
    min_fill_ratio_ = config["contours"]["min_fill_ratio"].as<double>();
    use_gui_ = config["gui"].as<bool>();
    if (debug_ && use_gui_) {
        initGUI();
    }
}
GuidanceDetectorOpenCV::~GuidanceDetectorOpenCV() {}
void GuidanceDetectorOpenCV::setCallback(DetectorCallback callback) {
    infer_callback_ = callback;
}

void GuidanceDetectorOpenCV::processCallback(const CommonFrame& frame) {
    std::vector<GreenLight> lights;
    cv::Mat img = frame.src_img.clone();

    cv::Mat hsv;
    cv::cvtColor(img, hsv, cv::COLOR_BGR2HSV);

    cv::Scalar lower_green(lowH, lowS, lowV);
    cv::Scalar upper_green(highH, highS, highV);

    cv::Mat mask;
    cv::inRange(hsv, lower_green, upper_green, mask);

    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(5, 5));
    cv::morphologyEx(mask, mask, cv::MORPH_OPEN, kernel);
    cv::morphologyEx(mask, mask, cv::MORPH_CLOSE, kernel);

    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(mask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    std::vector<int> valid_indices;

    for (size_t i = 0; i < contours.size(); i++) {
        double area = cv::contourArea(contours[i]);
        double perimeter = cv::arcLength(contours[i], true);
        if (perimeter == 0)
            continue;

        double circularity = 4 * CV_PI * area / (perimeter * perimeter);

        cv::RotatedRect rRect = cv::minAreaRect(contours[i]);
        double width = rRect.size.width;
        double height = rRect.size.height;

        if (width <= 0 || height <= 0)
            continue;

        double rect_area = width * height;
        double fill_ratio = area / rect_area;
        double aspect_ratio = std::min(width, height) / std::max(width, height);

        if (area > min_area_ && area < max_area_ && fill_ratio > min_fill_ratio_
            && aspect_ratio > min_aspect_ratio)
        {
            cv::Point2f center;
            float radius;
            cv::minEnclosingCircle(contours[i], center, radius);
            GreenLight gl;
            gl.center_point = center;
            gl.box = cv::boundingRect(contours[i]);
            gl.score = circularity;
            gl.radius = radius;
            lights.push_back(gl);
        }
    }
    static auto last = std::chrono::steady_clock::now();
    auto now = std::chrono::steady_clock::now();
    double dt = std::chrono::duration<double, std::milli>(now - last).count();

    if (debug_ && dt > 33.3 && use_gui_) { // 30Hz 刷新
        cv::imshow("mask", mask);
        cv::waitKey(1); // 非阻塞
        last = now;
    }

    if (infer_callback_) {
        infer_callback_(lights, frame);
    }
}
void GuidanceDetectorOpenCV::pushInput(CommonFrame& frame) {
    frame.id = current_id_++;
    processCallback(frame);
}
} // namespace auto_guidance