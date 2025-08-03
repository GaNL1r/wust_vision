#pragma once
#include <Eigen/Dense>
#include <NvInferRuntime.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <cmath>
#include <cstdio>
#include <cuda_fp16.h>
#include <opencv2/core/hal/interface.h>
#include <thrust/device_ptr.h>
#include <thrust/sort.h>
static constexpr int TILE_W = 32;
static constexpr int TILE_H = 32;
__global__ void letterbox_kernel_shared(
    const uchar* __restrict__ input_bgr,
    int in_w,
    int in_h,
    float* __restrict__ output_nchw,
    int out_w,
    int out_h,
    float scale,
    int pad_t,
    int pad_l
) ;
cudaTextureObject_t createTextureObject(float4* d_img, int width, int height);
__global__ void letterbox_kernel_texture(
    float* __restrict__ output_nchw,
    int out_w,
    int out_h,
    float scale,
    int pad_t,
    int pad_l,
    cudaTextureObject_t texture
);
__global__ void convertBGRUcharToFloat4Kernel(
    const unsigned char* __restrict__ input_bgr,
    float4* __restrict__ output_float4,
    int width, int height
);
__global__ void letterbox_kernel_uchar_textureless(
    const uchar* __restrict__ input_bgr,
    float* __restrict__ output_nchw,
    int in_w, int in_h,
    int out_w, int out_h,
    float scale, int pad_t, int pad_l
);
