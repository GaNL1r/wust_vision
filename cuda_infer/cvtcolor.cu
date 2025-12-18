#include "cvtcolor.hpp"

namespace cuda_cvt {
constexpr int BLOCK_W = 16;
constexpr int BLOCK_H = 16;

template<BayerPattern P>
__device__ __forceinline__ constexpr int bayerColor(int x, int y) {
    int idx = ((y & 1) << 1) | (x & 1);

    if constexpr (P == BayerPattern::RGGB)
        return (idx == 0) ? 0 : (idx == 3 ? 2 : 1);
    else if constexpr (P == BayerPattern::BGGR)
        return (idx == 0) ? 2 : (idx == 3 ? 0 : 1);
    else if constexpr (P == BayerPattern::GBRG)
        return (idx == 1) ? 2 : (idx == 2 ? 0 : 1);
    else // GRBG
        return (idx == 1) ? 0 : (idx == 2 ? 2 : 1);
}
template<OutputOrder O>
__device__ __forceinline__ void storeRGB(uint8_t* out, int R, int G, int B) {
    if constexpr (O == OutputOrder::RGB) {
        out[0] = (uint8_t)R;
        out[1] = (uint8_t)G;
        out[2] = (uint8_t)B;
    } else {
        out[0] = (uint8_t)B;
        out[1] = (uint8_t)G;
        out[2] = (uint8_t)R;
    }
}
template<BayerPattern P, OutputOrder O, InterpMode M>
__global__ void
bayer_kernel(const uint8_t* __restrict__ bayer, uint8_t* __restrict__ out, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x <= 0 || y <= 0 || x >= width - 1 || y >= height - 1)
        return;

    int idx = y * width + x;
    int out_idx = idx * 3;

    int c = __ldg(&bayer[idx]);
    int l = __ldg(&bayer[idx - 1]);
    int r = __ldg(&bayer[idx + 1]);
    int u = __ldg(&bayer[idx - width]);
    int d = __ldg(&bayer[idx + width]);
    int ul = __ldg(&bayer[idx - width - 1]);
    int ur = __ldg(&bayer[idx - width + 1]);
    int dl = __ldg(&bayer[idx + width - 1]);
    int dr = __ldg(&bayer[idx + width + 1]);

    int R = 0, G = 0, B = 0;
    int color = bayerColor<P>(x, y);

    constexpr bool G_on_R_row_pattern = (P == BayerPattern::RGGB || P == BayerPattern::GRBG);
    bool g_on_r = G_on_R_row_pattern ? ((y & 1) == 0) : ((y & 1) == 1);

    if constexpr (M == InterpMode::EA) {
        if (color == 0) { // R
            R = c;
            int dh = l > r ? l - r : r - l;
            int dv = u > d ? u - d : d - u;
            G = (dh < dv) ? ((l + r) >> 1) : (dv < dh) ? ((u + d) >> 1) : ((l + r + u + d) >> 2);
            B = (ul + ur + dl + dr) >> 2;
        } else if (color == 2) { // B
            B = c;
            int dh = l > r ? l - r : r - l;
            int dv = u > d ? u - d : d - u;
            G = (dh < dv) ? ((l + r) >> 1) : (dv < dh) ? ((u + d) >> 1) : ((l + r + u + d) >> 2);
            R = (ul + ur + dl + dr) >> 2;
        } else { // G
            G = c;
            if (g_on_r) {
                R = (l + r) >> 1;
                B = (u + d) >> 1;
            } else {
                R = (u + d) >> 1;
                B = (l + r) >> 1;
            }
        }
    } else {
        if (color == 0) { // R
            R = c;
            G = (l + r + u + d) >> 2;
            B = (ul + ur + dl + dr) >> 2;
        } else if (color == 2) { // B
            B = c;
            G = (l + r + u + d) >> 2;
            R = (ul + ur + dl + dr) >> 2;
        } else { // G
            G = c;
            if (g_on_r) {
                R = (l + r) >> 1;
                B = (u + d) >> 1;
            } else {
                R = (u + d) >> 1;
                B = (l + r) >> 1;
            }
        }
    }

    storeRGB<O>(&out[out_idx], B, G, R);
}

inline void launchBayerKernel(
    int cv_code,
    dim3 grid,
    dim3 block,
    cudaStream_t stream,
    const uint8_t* d_bayer,
    uint8_t* d_out,
    int w,
    int h
) {
    if (cv_code == cv::COLOR_BayerRG2RGB) {
        bayer_kernel<BayerPattern::RGGB, OutputOrder::RGB, InterpMode::BILINEAR>
            <<<grid, block, 0, stream>>>(d_bayer, d_out, w, h);
    } else if (cv_code == cv::COLOR_BayerRG2BGR) {
        bayer_kernel<BayerPattern::RGGB, OutputOrder::BGR, InterpMode::BILINEAR>
            <<<grid, block, 0, stream>>>(d_bayer, d_out, w, h);
    } else if (cv_code == cv::COLOR_BayerRG2RGB_EA) {
        bayer_kernel<BayerPattern::RGGB, OutputOrder::RGB, InterpMode::EA>
            <<<grid, block, 0, stream>>>(d_bayer, d_out, w, h);
    } else if (cv_code == cv::COLOR_BayerRG2BGR_EA) {
        bayer_kernel<BayerPattern::RGGB, OutputOrder::BGR, InterpMode::EA>
            <<<grid, block, 0, stream>>>(d_bayer, d_out, w, h);
    }

    else if (cv_code == cv::COLOR_BayerBG2RGB)
    {
        bayer_kernel<BayerPattern::BGGR, OutputOrder::RGB, InterpMode::BILINEAR>
            <<<grid, block, 0, stream>>>(d_bayer, d_out, w, h);
    } else if (cv_code == cv::COLOR_BayerBG2BGR) {
        bayer_kernel<BayerPattern::BGGR, OutputOrder::BGR, InterpMode::BILINEAR>
            <<<grid, block, 0, stream>>>(d_bayer, d_out, w, h);
    } else if (cv_code == cv::COLOR_BayerBG2RGB_EA) {
        bayer_kernel<BayerPattern::BGGR, OutputOrder::RGB, InterpMode::EA>
            <<<grid, block, 0, stream>>>(d_bayer, d_out, w, h);
    } else if (cv_code == cv::COLOR_BayerBG2BGR_EA) {
        bayer_kernel<BayerPattern::BGGR, OutputOrder::BGR, InterpMode::EA>
            <<<grid, block, 0, stream>>>(d_bayer, d_out, w, h);
    }

    else if (cv_code == cv::COLOR_BayerGB2RGB)
    {
        bayer_kernel<BayerPattern::GBRG, OutputOrder::RGB, InterpMode::BILINEAR>
            <<<grid, block, 0, stream>>>(d_bayer, d_out, w, h);
    } else if (cv_code == cv::COLOR_BayerGB2BGR) {
        bayer_kernel<BayerPattern::GBRG, OutputOrder::BGR, InterpMode::BILINEAR>
            <<<grid, block, 0, stream>>>(d_bayer, d_out, w, h);
    } else if (cv_code == cv::COLOR_BayerGB2RGB_EA) {
        bayer_kernel<BayerPattern::GBRG, OutputOrder::RGB, InterpMode::EA>
            <<<grid, block, 0, stream>>>(d_bayer, d_out, w, h);
    } else if (cv_code == cv::COLOR_BayerGB2BGR_EA) {
        bayer_kernel<BayerPattern::GBRG, OutputOrder::BGR, InterpMode::EA>
            <<<grid, block, 0, stream>>>(d_bayer, d_out, w, h);
    }

    else if (cv_code == cv::COLOR_BayerGR2RGB)
    {
        bayer_kernel<BayerPattern::GRBG, OutputOrder::RGB, InterpMode::BILINEAR>
            <<<grid, block, 0, stream>>>(d_bayer, d_out, w, h);
    } else if (cv_code == cv::COLOR_BayerGR2BGR) {
        bayer_kernel<BayerPattern::GRBG, OutputOrder::BGR, InterpMode::BILINEAR>
            <<<grid, block, 0, stream>>>(d_bayer, d_out, w, h);
    } else if (cv_code == cv::COLOR_BayerGR2RGB_EA) {
        bayer_kernel<BayerPattern::GRBG, OutputOrder::RGB, InterpMode::EA>
            <<<grid, block, 0, stream>>>(d_bayer, d_out, w, h);
    } else if (cv_code == cv::COLOR_BayerGR2BGR_EA) {
        bayer_kernel<BayerPattern::GRBG, OutputOrder::BGR, InterpMode::EA>
            <<<grid, block, 0, stream>>>(d_bayer, d_out, w, h);
    }

    else
    {
        throw std::runtime_error("Unsupported cv::COLOR_Bayer* conversion");
    }
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
    if (d_bayer_ || d_target_) {
        cudaStreamSynchronize(stream_);
    }

    if (d_bayer_) {
        cudaFree(d_bayer_);
        d_bayer_ = nullptr;
    }

    if (d_target_) {
        cudaFree(d_target_);
        d_target_ = nullptr;
    }

    bayer_capacity_ = 0;
    rgb_capacity_ = 0;

    width_ = 0;
    height_ = 0;
}

void CudaBayer_EA::allocIfNeeded(int width, int height) {
    size_t need_bayer = static_cast<size_t>(width) * height;
    size_t need_rgb = need_bayer * 3;

    if (need_bayer <= bayer_capacity_ && need_rgb <= rgb_capacity_) {
        width_ = width;
        height_ = height;
        return;
    }

    // 容量不足才释放
    release();

    cudaMalloc(&d_bayer_, need_bayer);
    cudaMalloc(&d_target_, need_rgb);

    bayer_capacity_ = need_bayer;
    rgb_capacity_ = need_rgb;

    width_ = width;
    height_ = height;
}

void CudaBayer_EA::process(const cv::Mat& bayer, cv::Mat& target, int cv_bayer_code) {
    CV_Assert(bayer.type() == CV_8UC1);
    CV_Assert(bayer.isContinuous());

    int w = bayer.cols;
    int h = bayer.rows;

    allocIfNeeded(w, h);

    target.create(h, w, CV_8UC3);

    size_t bayer_bytes = static_cast<size_t>(w) * h;
    size_t target_bytes = bayer_bytes * 3;

    // H2D
    cudaMemcpyAsync(d_bayer_, bayer.data, bayer_bytes, cudaMemcpyHostToDevice, stream_);

    dim3 block(BLOCK_W, BLOCK_H);
    dim3 grid((w + block.x - 1) / block.x, (h + block.y - 1) / block.y);

    launchBayerKernel(cv_bayer_code, grid, block, stream_, d_bayer_, d_target_, w, h);

    // D2H
    cudaMemcpyAsync(target.data, d_target_, target_bytes, cudaMemcpyDeviceToHost, stream_);

    cudaStreamSynchronize(stream_);
}

} // namespace cuda_cvt