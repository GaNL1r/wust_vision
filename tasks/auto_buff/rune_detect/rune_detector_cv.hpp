#pragma once
#include "opencv2/opencv.hpp"
#include "tasks/auto_buff/type.hpp"
#include "tasks/utils.hpp"
class RuneDetectorCV {
public:
RuneDetectorCV() 
{
 template_mat = cv::imread("/home/hy/rune_dl/888.png",cv::IMREAD_GRAYSCALE);
}
    rune::RuneFan detect(const cv::Mat& input_roi, const cv::Point2f& r_tag,cv::Mat& debug_img,bool debug=false);
    cv::Mat template_mat;
};
