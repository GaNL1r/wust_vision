
#include "wust_vl/video/uvc.hpp"
#include <iostream>
#include <opencv2/opencv.hpp>
int main() {
    wust_vl::video::UVC uvc;
    auto config = YAML::LoadFile("/home/hy/wust_vision/config/camera.yaml");
    uvc.loadConfig(config["uvc"]);
    uvc.start();
    while (true) {
        auto frame = uvc.readImage();
        if (!frame.src_img.empty()) {
            cv::imshow("frame", frame.src_img);
            cv::waitKey(1);
        }
    }
    uvc.stop();
    return 0;
}
