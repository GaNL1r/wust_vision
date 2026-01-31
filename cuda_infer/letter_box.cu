#include "letter_box.hpp"
__global__ void letterbox_kernel_shared(
    const uchar* __restrict__ input_bgr,
    int in_w,
    int in_h,
    float* __restrict__ output_nchw,
    int out_w,
    int out_h,
    float scale,
    int pad_t,
    int pad_l,
    float norm,
    bool swap_rb
) {
    int x = blockIdx.x * TILE_W + threadIdx.x;
    int y = blockIdx.y * TILE_H + threadIdx.y;
    if (x >= out_w || y >= out_h)
        return;

    // 共享内存大小
    __shared__ uchar4 smem[TILE_H + 1][TILE_W + 1];

    int tid = threadIdx.y * blockDim.x + threadIdx.x;
    int total_smem = (TILE_W + 1) * (TILE_H + 1);
    int iter = (total_smem + blockDim.x * blockDim.y - 1) / (blockDim.x * blockDim.y);

    float inv_scale = 1.0f / scale;
    float block_start_x = (blockIdx.x * TILE_W - pad_l) * inv_scale;
    float block_start_y = (blockIdx.y * TILE_H - pad_t) * inv_scale;

    for (int i = 0; i < iter; i++) {
        int idx = tid + i * (blockDim.x * blockDim.y);
        if (idx < total_smem) {
            int sx = idx % (TILE_W + 1);
            int sy = idx / (TILE_W + 1);

            float in_x = block_start_x + sx * inv_scale;
            float in_y = block_start_y + sy * inv_scale;
            int ix = (int)in_x; // 等价 floorf(in_x) if in_x>=0
            int iy = (int)in_y;

            uchar4 p = make_uchar4(114, 114, 114, 0); // padding color BGR + unused
            if (ix >= 0 && iy >= 0 && ix < in_w && iy < in_h) {
                int offset = (iy * in_w + ix) * 3;
                p.x = input_bgr[offset]; // b
                p.y = input_bgr[offset + 1]; // g
                p.z = input_bgr[offset + 2]; // r
            }
            smem[sy][sx] = p;
        }
    }
    __syncthreads();

    float in_x = (x - pad_l) * inv_scale;
    float in_y = (y - pad_t) * inv_scale;
    float dx = in_x - floorf(in_x);
    float dy = in_y - floorf(in_y);
    float dx1 = 1.0f - dx, dy1 = 1.0f - dy;

    uchar4 p00 = smem[threadIdx.y][threadIdx.x];
    uchar4 p01 = smem[threadIdx.y][threadIdx.x + 1];
    uchar4 p10 = smem[threadIdx.y + 1][threadIdx.x];
    uchar4 p11 = smem[threadIdx.y + 1][threadIdx.x + 1];

    float out_r = dx1 * dy1 * p00.z + dx * dy1 * p01.z + dx1 * dy * p10.z + dx * dy * p11.z;
    float out_g = dx1 * dy1 * p00.y + dx * dy1 * p01.y + dx1 * dy * p10.y + dx * dy * p11.y;
    float out_b = dx1 * dy1 * p00.x + dx * dy1 * p01.x + dx1 * dy * p10.x + dx * dy * p11.x;

    int out_idx = y * out_w + x;
    if (swap_rb) {
        output_nchw[out_idx + 0 * out_w * out_h] = out_r * norm;
        output_nchw[out_idx + 1 * out_w * out_h] = out_g * norm;
        output_nchw[out_idx + 2 * out_w * out_h] = out_b * norm;
    } else {
        output_nchw[out_idx + 0 * out_w * out_h] = out_b * norm;
        output_nchw[out_idx + 1 * out_w * out_h] = out_g * norm;
        output_nchw[out_idx + 2 * out_w * out_h] = out_r * norm;
    }
}

extern "C" __global__ void letterbox_kernel_pitched(
    const unsigned char* __restrict__ d_input_bgr,
    size_t pitch,
    int src_w,
    int src_h,
    float* __restrict__ d_nchw,
    int OUT_W,
    int OUT_H,
    float scale,
    int pad_t,
    int pad_l,
    float norm,
    bool swap_rb
) {
    int ox = blockIdx.x * blockDim.x + threadIdx.x;
    int oy = blockIdx.y * blockDim.y + threadIdx.y;
    if (ox >= OUT_W || oy >= OUT_H)
        return;

    // === inverse mapping ===
    float fx = (ox - pad_l) / scale;
    float fy = (oy - pad_t) / scale;

    int out_idx = oy * OUT_W + ox;
    int plane = OUT_W * OUT_H;

    // padding color
    float r = 114.f, g = 114.f, b = 114.f;

    if (fx >= 0.f && fy >= 0.f && fx < src_w - 1 && fy < src_h - 1) {
        int x0 = (int)fx;
        int y0 = (int)fy;
        int x1 = x0 + 1;
        int y1 = y0 + 1;

        float dx = fx - x0;
        float dy = fy - y0;
        float dx1 = 1.f - dx;
        float dy1 = 1.f - dy;

        // row pointers (pitch-safe)
        const unsigned char* row0 = (const unsigned char*)((const char*)d_input_bgr + y0 * pitch);
        const unsigned char* row1 = (const unsigned char*)((const char*)d_input_bgr + y1 * pitch);

        int i00 = x0 * 3;
        int i01 = x1 * 3;

        // load 4 pixels (BGR)
        float b00 = row0[i00 + 0], g00 = row0[i00 + 1], r00 = row0[i00 + 2];
        float b01 = row0[i01 + 0], g01 = row0[i01 + 1], r01 = row0[i01 + 2];
        float b10 = row1[i00 + 0], g10 = row1[i00 + 1], r10 = row1[i00 + 2];
        float b11 = row1[i01 + 0], g11 = row1[i01 + 1], r11 = row1[i01 + 2];

        // bilinear interpolation
        float w00 = dx1 * dy1;
        float w01 = dx * dy1;
        float w10 = dx1 * dy;
        float w11 = dx * dy;

        r = r00 * w00 + r01 * w01 + r10 * w10 + r11 * w11;
        g = g00 * w00 + g01 * w01 + g10 * w10 + g11 * w11;
        b = b00 * w00 + b01 * w01 + b10 * w10 + b11 * w11;
    }

    // NCHW (RGB)
    if (swap_rb) {
        d_nchw[out_idx + 0 * plane] = r * norm;
        d_nchw[out_idx + 1 * plane] = g * norm;
        d_nchw[out_idx + 2 * plane] = b * norm;
    } else {
        d_nchw[out_idx + 0 * plane] = b * norm;
        d_nchw[out_idx + 1 * plane] = g * norm;
        d_nchw[out_idx + 2 * plane] = r * norm;
    }
}
