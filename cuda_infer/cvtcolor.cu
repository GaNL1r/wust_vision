#include "cvtcolor.hpp"

namespace cuda_cvt {
__device__ __forceinline__ uint8_t clamp_u8(int v) {
    return v < 0 ? 0 : (v > 255 ? 255 : v);
}

__global__ void bayerRG2BGR_EA_kernel(
    const uint8_t* __restrict__ bayer,
    uint8_t* __restrict__ bgr,
    int width,
    int height
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x <= 1 || y <= 1 || x >= width - 2 || y >= height - 2)
        return;

    int idx = y * width + x;
    bool er = (y & 1) == 0;
    bool ec = (x & 1) == 0;

    int c = bayer[idx];
    int l = bayer[idx - 1];
    int r = bayer[idx + 1];
    int u = bayer[idx - width];
    int d = bayer[idx + width];
    int ul = bayer[idx - width - 1];
    int ur = bayer[idx - width + 1];
    int dl = bayer[idx + width - 1];
    int dr = bayer[idx + width + 1];

    int R, G, B;

    if (er) {
        if (ec) {
            // R
            R = c;
            int dh = abs(l - r);
            int dv = abs(u - d);
            G = (dh < dv) ? ((l + r) >> 1) : (dv < dh) ? ((u + d) >> 1) : ((l + r + u + d) >> 2);
            B = (ul + ur + dl + dr) >> 2;
        } else {
            // G (R row)
            G = c;
            R = (l + r) >> 1;
            B = (u + d) >> 1;
        }
    } else {
        if (ec) {
            // G (B row)
            G = c;
            R = (u + d) >> 1;
            B = (l + r) >> 1;
        } else {
            // B
            B = c;
            int dh = abs(l - r);
            int dv = abs(u - d);
            G = (dh < dv) ? ((l + r) >> 1) : (dv < dh) ? ((u + d) >> 1) : ((l + r + u + d) >> 2);
            R = (ul + ur + dl + dr) >> 2;
        }
    }

    int o = (y * width + x) * 3;
    bgr[o + 0] = clamp_u8(R);
    bgr[o + 1] = clamp_u8(G);
    bgr[o + 2] = clamp_u8(B);
}
CudaBayer_EA::CudaBayer_EA() {
    cudaStreamCreate(&stream_);
}

CudaBayer_EA::~CudaBayer_EA() {
    release();
    if (stream_) {
        cudaStreamDestroy(stream_);
        stream_ = nullptr;
    }
}

void CudaBayer_EA::release() {
    if (d_bayer_)
        cudaFree(d_bayer_);
    if (d_target_)
        cudaFree(d_target_);
    if (h_bayer_pinned_)
        cudaFreeHost(h_bayer_pinned_);
    if (h_target_pinned_)
        cudaFreeHost(h_target_pinned_);

    d_bayer_ = d_target_ = nullptr;
    h_bayer_pinned_ = h_target_pinned_ = nullptr;
    width_ = height_ = 0;
}

void CudaBayer_EA::allocIfNeeded(int width, int height) {
    if (width == width_ && height == height_)
        return;

    release();

    size_t bayer_bytes = width * height;
    size_t rgb_bytes = width * height * 3;

    cudaMalloc(&d_bayer_, bayer_bytes);
    cudaMalloc(&d_target_, rgb_bytes);

    cudaMallocHost(&h_bayer_pinned_, bayer_bytes); // pinned
    cudaMallocHost(&h_target_pinned_, rgb_bytes);

    width_ = width;
    height_ = height;
}

void CudaBayer_EA::process(const cv::Mat& bayer, cv::Mat& target) {
    CV_Assert(bayer.type() == CV_8UC1);
    CV_Assert(bayer.isContinuous());

    int w = bayer.cols;
    int h = bayer.rows;

    allocIfNeeded(w, h);

    target.create(h, w, CV_8UC3);

    size_t bayer_bytes = w * h;
    size_t target_bytes = w * h * 3;

    // CPU -> pinned
    std::memcpy(h_bayer_pinned_, bayer.data, bayer_bytes);

    // async H2D
    cudaMemcpyAsync(d_bayer_, h_bayer_pinned_, bayer_bytes, cudaMemcpyHostToDevice, stream_);

    dim3 block(32, 32);
    dim3 grid((w + block.x - 1) / block.x, (h + block.y - 1) / block.y);

    bayerRG2BGR_EA_kernel<<<grid, block, 0, stream_>>>(d_bayer_, d_target_, w, h);

    // async D2H
    cudaMemcpyAsync(h_target_pinned_, d_target_, target_bytes, cudaMemcpyDeviceToHost, stream_);

    cudaStreamSynchronize(stream_);

    // pinned -> cv::Mat
    std::memcpy(target.data, h_target_pinned_, target_bytes);
}

} // namespace cuda_cvt