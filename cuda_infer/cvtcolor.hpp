#pragma once
#include <cuda_runtime.h>
#include <opencv2/core/mat.hpp>
namespace cuda_cvt {
class CudaBayer_EA {
public:
    CudaBayer_EA();
    ~CudaBayer_EA();

    // cv::Mat -> cv::Mat（回传 CPU）
    void process(const cv::Mat& bayer, cv::Mat& rgb);

private:
    void allocIfNeeded(int width, int height);
    void release();

private:
    uint8_t* d_bayer_ = nullptr;
    uint8_t* d_target_ = nullptr;

    uint8_t* h_bayer_pinned_ = nullptr;
    uint8_t* h_target_pinned_ = nullptr;

    int width_ = 0;
    int height_ = 0;

    cudaStream_t stream_ = nullptr;
};
} // namespace cuda_cvt