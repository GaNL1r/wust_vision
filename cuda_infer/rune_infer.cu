#include "rune_infer.hpp"
#include <cmath>
#include <cstdio>
#include <opencv2/core/hal/interface.h>
#include <thrust/device_ptr.h>
#include <thrust/sort.h>

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf( \
                stderr, \
                "CUDA error at %s:%d: %s\n", \
                __FILE__, \
                __LINE__, \
                cudaGetErrorString(err) \
            ); \
            exit(EXIT_FAILURE); \
        } \
    } while (0)
static constexpr int INPUT_W = 480; // Width of input
static constexpr int INPUT_H = 480; // Height of input
static constexpr int NUM_CLASSES = 2; // Number of classes
static constexpr int NUM_COLORS = 2; // Number of color
static constexpr int NUM_POINTS = 5;
static constexpr int NUM_POINTS_2 = 2 * NUM_POINTS;
static constexpr float MERGE_CONF_ERROR = 0.15;
static constexpr float MERGE_MIN_IOU = 0.9;
namespace rune_cuda_infer {
GPUGridAndStride* init_grid_strides_on_gpu(
    int in_w,
    int in_h,
    const std::vector<int>& strides,
    size_t& device_grid_count
) {
    std::vector<GPUGridAndStride> host_grid_strides;
    for (auto stride: strides) {
        int num_grid_w = in_w / stride;
        int num_grid_h = in_h / stride;
        for (int y = 0; y < num_grid_h; ++y) {
            for (int x = 0; x < num_grid_w; ++x) {
                host_grid_strides.emplace_back(GPUGridAndStride { x, y, stride });
            }
        }
    }
    device_grid_count = host_grid_strides.size();
    if (device_grid_count == 0)
        return nullptr;

    GPUGridAndStride* device_grid_strides = nullptr;
    CUDA_CHECK(cudaMalloc(
        reinterpret_cast<void**>(&device_grid_strides),
        device_grid_count * sizeof(GPUGridAndStride)
    ));
    CUDA_CHECK(cudaMemcpy(
        device_grid_strides,
        host_grid_strides.data(),
        device_grid_count * sizeof(GPUGridAndStride),
        cudaMemcpyHostToDevice
    ));

    return device_grid_strides;
}
__device__ float3 bilinear_interpolate_rgb_fast(const uchar* img, int w, int h, float x, float y) {
    x = fminf(fmaxf(x, 0.f), w - 1.001f);
    y = fminf(fmaxf(y, 0.f), h - 1.001f);

    int x0 = floorf(x), x1 = x0 + 1;
    int y0 = floorf(y), y1 = y0 + 1;

    float dx = x - x0, dy = y - y0;
    float dx1 = 1.f - dx, dy1 = 1.f - dy;

    const int s0 = (y0 * w + x0) * 3;
    const int s1 = (y0 * w + x1) * 3;
    const int s2 = (y1 * w + x0) * 3;
    const int s3 = (y1 * w + x1) * 3;

    float3 out;
    #pragma unroll
    for (int c = 0; c < 3; ++c) {
        float v00 = img[s0 + c];
        float v01 = img[s1 + c];
        float v10 = img[s2 + c];
        float v11 = img[s3 + c];
        float val = dx1 * dy1 * v00 + dx * dy1 * v01 + dx1 * dy * v10 + dx * dy * v11;
        (&out.x)[c] = val;
    }
    return out;
}


#define TILE_W 32
#define TILE_H 32

__global__ void letterbox_kernel_shared(
    const uchar* __restrict__ input_bgr, int in_w, int in_h,
    float* __restrict__ output_nchw, int out_w, int out_h,
    float scale, int pad_t, int pad_l
) {
    // 输出图像坐标
    int x = blockIdx.x * TILE_W + threadIdx.x;
    int y = blockIdx.y * TILE_H + threadIdx.y;
    int out_idx = y * out_w + x;

    // 输入图像坐标（浮点）
    float in_x = (x - pad_l) / scale;
    float in_y = (y - pad_t) / scale;

    // 线程块内共享内存（RGB 分开存）
    __shared__ uchar smem_r[TILE_H + 1][TILE_W + 1];
    __shared__ uchar smem_g[TILE_H + 1][TILE_W + 1];
    __shared__ uchar smem_b[TILE_H + 1][TILE_W + 1];

    // 加载共享内存中所需的 input patch
    float patch_x = (blockIdx.x * TILE_W + threadIdx.x - pad_l) / scale;
    float patch_y = (blockIdx.y * TILE_H + threadIdx.y - pad_t) / scale;

    int px = floorf(patch_x);
    int py = floorf(patch_y);

    uchar r = 114, g = 114, b = 114;
    if (px >= 0 && py >= 0 && px < in_w && py < in_h) {
        int offset = (py * in_w + px) * 3;
        b = input_bgr[offset + 0];
        g = input_bgr[offset + 1];
        r = input_bgr[offset + 2];
    }

    if (threadIdx.x <= TILE_W && threadIdx.y <= TILE_H) {
        smem_r[threadIdx.y][threadIdx.x] = r;
        smem_g[threadIdx.y][threadIdx.x] = g;
        smem_b[threadIdx.y][threadIdx.x] = b;
    }
    __syncthreads();

    if (x >= out_w || y >= out_h) return;

    float3 out_rgb = make_float3(114.f, 114.f, 114.f);
    if (in_x >= 0 && in_y >= 0 && in_x < in_w - 1 && in_y < in_h - 1) {
        float dx = in_x - floorf(in_x);
        float dy = in_y - floorf(in_y);
        float dx1 = 1.f - dx, dy1 = 1.f - dy;

        // 从共享内存中读取邻域像素
        uchar v00r = smem_r[threadIdx.y][threadIdx.x];
        uchar v01r = smem_r[threadIdx.y][threadIdx.x + 1];
        uchar v10r = smem_r[threadIdx.y + 1][threadIdx.x];
        uchar v11r = smem_r[threadIdx.y + 1][threadIdx.x + 1];

        uchar v00g = smem_g[threadIdx.y][threadIdx.x];
        uchar v01g = smem_g[threadIdx.y][threadIdx.x + 1];
        uchar v10g = smem_g[threadIdx.y + 1][threadIdx.x];
        uchar v11g = smem_g[threadIdx.y + 1][threadIdx.x + 1];

        uchar v00b = smem_b[threadIdx.y][threadIdx.x];
        uchar v01b = smem_b[threadIdx.y][threadIdx.x + 1];
        uchar v10b = smem_b[threadIdx.y + 1][threadIdx.x];
        uchar v11b = smem_b[threadIdx.y + 1][threadIdx.x + 1];

        out_rgb.x = dx1 * dy1 * v00r + dx * dy1 * v01r + dx1 * dy * v10r + dx * dy * v11r;
        out_rgb.y = dx1 * dy1 * v00g + dx * dy1 * v01g + dx1 * dy * v10g + dx * dy * v11g;
        out_rgb.z = dx1 * dy1 * v00b + dx * dy1 * v01b + dx1 * dy * v10b + dx * dy * v11b;
    }

    output_nchw[out_idx + 0 * out_w * out_h] = out_rgb.x;  // R
    output_nchw[out_idx + 1 * out_w * out_h] = out_rgb.y;  // G
    output_nchw[out_idx + 2 * out_w * out_h] = out_rgb.z;  // B
}

__global__ void init_objs_kernel(GPURuneObject* objs, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N)
        return;

    GPURuneObject& obj = objs[idx];
    for (int i = 0; i < NUM_POINTS; ++i) {
        obj.x[i] = 0.f;
        obj.y[i] = 0.f;
    }
    obj.confidence = 0.f;
    obj.color_id = 0;
    obj.type_id = 0;
    obj.valid = 0;
}

__global__ void decode_rune_kernel(
    const float* output,
    const GPUGridAndStride* grid_strides,
    int num_detections,
    const float* tf_matrix,
    GPURuneObject* output_objs,
    float conf_threshold
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_detections)
        return;

    const float* det = output + idx * 15;
    float conf = det[10];

    GPURuneObject& obj = output_objs[idx];
    // 先初始化
    for (int i = 0; i < NUM_POINTS; ++i) {
        obj.x[i] = 0.f;
        obj.y[i] = 0.f;
    }
    obj.confidence = 0.f;
    obj.color_id = 0;
    obj.type_id = 0;
    obj.valid = 0;

    if (!isfinite(conf) || conf < conf_threshold)
        return;

    const GPUGridAndStride grid = grid_strides[idx];

    float norm_x[5], norm_y[5];
    for (int i = 0; i < 5; ++i) {
        norm_x[i] = (det[i * 2 + 0] + grid.grid0) * grid.stride;
        norm_y[i] = (det[i * 2 + 1] + grid.grid1) * grid.stride;
    }

    float dst_x[5], dst_y[5];
    for (int i = 0; i < 5; ++i) {
        float x = norm_x[i], y = norm_y[i];
        float x_ = tf_matrix[0] * x + tf_matrix[1] * y + tf_matrix[2];
        float y_ = tf_matrix[3] * x + tf_matrix[4] * y + tf_matrix[5];
        float w = tf_matrix[6] * x + tf_matrix[7] * y + tf_matrix[8];
        dst_x[i] = x_ / (fabsf(w) > 1e-6f ? w : 1e-6f);
        dst_y[i] = y_ / (fabsf(w) > 1e-6f ? w : 1e-6f);
    }

    const float* color_scores = det + NUM_POINTS_2 + 1;
    int color_id = 0;
    float color_score = color_scores[0];
    for (int i = 1; i < NUM_COLORS; ++i) {
        float s = color_scores[i];
        if (s > color_score) {
            color_score = s;
            color_id = i;
        }
    }

    const float* type_scores = det + NUM_POINTS_2 + 1 + NUM_COLORS;
    int type_id = 0;
    float type_score = type_scores[0];
    for (int i = 1; i < NUM_CLASSES; ++i) {
        float s = type_scores[i];
        if (s > type_score) {
            type_score = s;
            type_id = i;
        }
    }

    for (int i = 0; i < NUM_POINTS; ++i) {
        obj.x[i] = dst_x[i];
        obj.y[i] = dst_y[i];
    }
    obj.confidence = conf;
    obj.color_id = color_id;
    obj.type_id = type_id;
    obj.valid = 1;
}

__global__ void clear_invalid_topk(GPURuneObject* objs, int N, int k) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N)
        return;
    if (i >= k)
        objs[i].valid = 0;
}

__global__ void nms_kernel(GPURuneObject* objs, int N, float thresh) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N || !objs[i].valid)
        return;

    GPURuneObject& a = objs[i];

    for (int j = 0; j < i; ++j) {
        GPURuneObject& b = objs[j];
        if (!b.valid)
            continue;

        if (a.color_id == b.color_id && a.type_id == b.type_id) {
            float ax_min = fminf(fminf(fminf(fminf(a.x[0], a.x[1]), a.x[2]), a.x[3]), a.x[4]);
            float ax_max = fmaxf(fmaxf(fmaxf(fmaxf(a.x[0], a.x[1]), a.x[2]), a.x[3]), a.x[4]);
            float ay_min = fminf(fminf(fminf(fminf(a.y[0], a.y[1]), a.y[2]), a.y[3]), a.y[4]);
            float ay_max = fmaxf(fmaxf(fmaxf(fmaxf(a.y[0], a.y[1]), a.y[2]), a.y[3]), a.y[4]);

            float bx_min = fminf(fminf(fminf(fminf(b.x[0], b.x[1]), b.x[2]), b.x[3]), b.x[4]);
            float bx_max = fmaxf(fmaxf(fmaxf(fmaxf(b.x[0], b.x[1]), b.x[2]), b.x[3]), b.x[4]);
            float by_min = fminf(fminf(fminf(fminf(b.y[0], b.y[1]), b.y[2]), b.y[3]), b.y[4]);
            float by_max = fmaxf(fmaxf(fmaxf(fmaxf(b.y[0], b.y[1]), b.y[2]), b.y[3]), b.y[4]);

            // IoU
            float x1 = fmaxf(ax_min, bx_min);
            float y1 = fmaxf(ay_min, by_min);
            float x2 = fminf(ax_max, bx_max);
            float y2 = fminf(ay_max, by_max);

            float inter = fmaxf(0.f, x2 - x1) * fmaxf(0.f, y2 - y1);
            float area_a = (ax_max - ax_min) * (ay_max - ay_min);
            float area_b = (bx_max - bx_min) * (by_max - by_min);
            float union_area = area_a + area_b - inter + 1e-6f;
            float iou = inter / union_area;

            if (iou > thresh) {
                a.valid = 0;
                break;
            }

            if (iou > MERGE_MIN_IOU && fabsf(a.confidence - b.confidence) < MERGE_CONF_ERROR) {
                int n = b.num_pts;
                if (n + 5 <= 10) {
                    for (int k = 0; k < 5; ++k) {
                        b.x[n + k] = a.x[k];
                        b.y[n + k] = a.y[k];
                    }
                    b.num_pts += 5;
                }
            }
        }
    }
}

CudaInfer::CudaInfer() = default;
CudaInfer::~CudaInfer() {
    release();
}

void CudaInfer::init(
    GPUGridAndStride* grid_strides,
    size_t img_bytes,
    int max_N,
    size_t grid_count
) {
    d_grid_strides_ = grid_strides;
    buf_image_bytes_ = img_bytes;
    buf_max_N_ = max_N;
    grid_count_ = grid_count;
    CUDA_CHECK(cudaMalloc(&d_input_bgr_, buf_image_bytes_));
    CUDA_CHECK(cudaMalloc(&d_nchw_, INPUT_W * INPUT_H * 3 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_objs_, buf_max_N_ * sizeof(GPURuneObject)));
    CUDA_CHECK(cudaMalloc(&d_tf_, 9 * sizeof(float)));
}

void CudaInfer::release() {
    if (d_grid_strides_)
        cudaFree(d_grid_strides_), d_grid_strides_ = nullptr;
    if (d_input_bgr_)
        cudaFree(d_input_bgr_), d_input_bgr_ = nullptr;
    if (d_nchw_)
        cudaFree(d_nchw_), d_nchw_ = nullptr;
    if (d_objs_)
        cudaFree(d_objs_), d_objs_ = nullptr;
    if (d_tf_)
        cudaFree(d_tf_), d_tf_ = nullptr;
}

float* CudaInfer::preprocess(
    const unsigned char* input_bgr_host,
    int img_w,
    int img_h,
    Eigen::Matrix3f& tf_matrix,
    cudaStream_t stream
) {
    float scale = fminf(INPUT_W / (float)img_w, INPUT_H / (float)img_h);
    int rw = round(img_w * scale), rh = round(img_h * scale);
    int pad_l = (INPUT_W - rw) / 2, pad_t = (INPUT_H - rh) / 2;

    tf_matrix << 1.f / scale, 0, -pad_l / scale, 0, 1.f / scale, -pad_t / scale, 0, 0, 1;

    size_t img_size = img_w * img_h * 3;
    CUDA_CHECK(
        cudaMemcpyAsync(d_input_bgr_, input_bgr_host, img_size, cudaMemcpyHostToDevice, stream)
    );

    dim3 threads(TILE_W, TILE_H);
    dim3 blocks((INPUT_W + TILE_W - 1) / TILE_W, (INPUT_H + TILE_H - 1) / TILE_H);
    letterbox_kernel_shared<<<blocks, threads, 0, stream>>>(
        d_input_bgr_,
        img_w, img_h,
        d_nchw_,
        INPUT_W, INPUT_H,
        scale,
        pad_t, pad_l
    );
    CUDA_CHECK(cudaGetLastError());
    return d_nchw_;
}
std::vector<GPURuneObject> CudaInfer::postprocess(
    const float* output,
    int N,
    const Eigen::Matrix3f& tf_matrix_eigen,
    float conf_th,
    float nms_th,
    int top_k
) {
    if (!output || !d_grid_strides_ || !d_tf_ || !d_objs_) {
        fprintf(stderr, "[Error] Null pointer in postprocess input\n");
        return {};
    }

    const int threads = 256;
    const int blocks = (N + threads - 1) / threads;

    float h_tf[9];
    Eigen::Map<Eigen::Matrix<float, 3, 3, Eigen::RowMajor>>(h_tf, 3, 3) = tf_matrix_eigen;
    CUDA_CHECK(cudaMemcpy(d_tf_, h_tf, sizeof(h_tf), cudaMemcpyHostToDevice));

    // 用kernel初始化结构体数组更安全
    init_objs_kernel<<<blocks, threads>>>(d_objs_, N);
    CUDA_CHECK(cudaGetLastError());

    decode_rune_kernel<<<blocks, threads>>>(output, d_grid_strides_, N, d_tf_, d_objs_, conf_th);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    int sort_N = std::min(N, buf_max_N_);
    if (sort_N > 0) {
        thrust::device_ptr<GPURuneObject> dev_ptr(d_objs_);
        thrust::sort(
            thrust::device,
            dev_ptr,
            dev_ptr + sort_N,
            ConfidenceComparator<GPURuneObject>()
        );
    }

    dim3 nmsBlocks((sort_N + threads - 1) / threads);
    clear_invalid_topk<<<nmsBlocks, threads>>>(d_objs_, sort_N, top_k);
    CUDA_CHECK(cudaGetLastError());

    nms_kernel<<<nmsBlocks, threads>>>(d_objs_, top_k, nms_th);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<GPURuneObject> h_objs(top_k);
    CUDA_CHECK(
        cudaMemcpy(h_objs.data(), d_objs_, sizeof(GPURuneObject) * top_k, cudaMemcpyDeviceToHost)
    );

    // 移除无效对象
    h_objs.erase(
        std::remove_if(
            h_objs.begin(),
            h_objs.end(),
            [](const GPURuneObject& o) { return o.valid == 0; }
        ),
        h_objs.end()
    );

    return h_objs;
}

std::vector<GPURuneObject> CudaInfer::process_trt(
    nvinfer1::IExecutionContext* context,
    void* device_buffers[2],
    int input_idx_,
    int output_idx_,
    const unsigned char* input_bgr_host,
    int img_w,
    int img_h,
    Eigen::Matrix3f& tf_matrix,
    cudaStream_t stream,
    int N,
    float conf_th,
    float nms_th,
    int top_k
) {
    void* input_tensor_ptr = preprocess(input_bgr_host, img_w, img_h, tf_matrix, stream);

    context->setTensorAddress("images", input_tensor_ptr);
    context->setTensorAddress("output", device_buffers[output_idx_]);

    if (!context->enqueueV3(stream)) {
        std::cerr << "enqueueV3 failed!";
        return {};
    }

    std::vector<GPURuneObject> results =
        postprocess((float*)device_buffers[output_idx_], N, tf_matrix, conf_th, nms_th, top_k);
    return results;
}

} // namespace rune_cuda_infer