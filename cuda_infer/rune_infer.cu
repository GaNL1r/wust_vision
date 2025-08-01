#include "rune_infer.hpp"
#include <thrust/device_ptr.h>
#include <thrust/sort.h>
#include <cstdio>
#include <cmath>

#define CUDA_CHECK(call) do {                                \
    cudaError_t err = call;                                  \
    if (err != cudaSuccess) {                                \
        fprintf(stderr, "CUDA error at %s:%d: %s\n",         \
                __FILE__, __LINE__, cudaGetErrorString(err));\
        exit(EXIT_FAILURE);                                  \
    }                                                        \
} while (0)
static constexpr int INPUT_W = 480; // Width of input
static constexpr int INPUT_H = 480; // Height of input
static constexpr int NUM_CLASSES = 2; // Number of classes
static constexpr int NUM_COLORS = 2; // Number of color
static constexpr int NUM_POINTS = 5;
static constexpr int NUM_POINTS_2 = 2 * NUM_POINTS;
static constexpr float MERGE_CONF_ERROR = 0.15;
static constexpr float MERGE_MIN_IOU = 0.9;
namespace  rune_cuda_infer
{
    GPUGridAndStride* init_grid_strides_on_gpu(
    int in_w,
    int in_h,
    const std::vector<int>& strides,
    size_t& device_grid_count)
{
    std::vector<GPUGridAndStride> host_grid_strides;
    for (auto stride : strides) {
        int num_grid_w = in_w / stride;
        int num_grid_h = in_h / stride;
        for (int y = 0; y < num_grid_h; ++y) {
            for (int x = 0; x < num_grid_w; ++x) {
                host_grid_strides.emplace_back(GPUGridAndStride{
                    x,
                    y,
                    stride});
            }
        }
    }
    device_grid_count = host_grid_strides.size();
    if (device_grid_count == 0) return nullptr;

    GPUGridAndStride* device_grid_strides = nullptr;
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&device_grid_strides),
        device_grid_count * sizeof(GPUGridAndStride)));
    CUDA_CHECK(cudaMemcpy(device_grid_strides, host_grid_strides.data(),
        device_grid_count * sizeof(GPUGridAndStride),
        cudaMemcpyHostToDevice));

    return device_grid_strides;
}

__device__ float bilinear_interpolate(const unsigned char* img, int w, int h, float x, float y, int c) {
    x = fminf(fmaxf(x, 0.f), w - 1.f);
    y = fminf(fmaxf(y, 0.f), h - 1.f);
    int x0 = floorf(x), x1 = min(x0 + 1, w - 1);
    int y0 = floorf(y), y1 = min(y0 + 1, h - 1);
    float dx = x - x0, dy = y - y0;
    float v00 = img[(y0 * w + x0) * 3 + c];
    float v01 = img[(y0 * w + x1) * 3 + c];
    float v10 = img[(y1 * w + x0) * 3 + c];
    float v11 = img[(y1 * w + x1) * 3 + c];
    return (1 - dx) * (1 - dy) * v00 + dx * (1 - dy) * v01 + (1 - dx) * dy * v10 + dx * dy * v11;
}

__global__ void letterbox_kernel(const unsigned char* input_bgr, int img_w, int img_h, float* output_nchw,
    int out_w, int out_h, float scale, int pad_top, int pad_left) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= out_w || y >= out_h) return;
    float in_x = (x - pad_left) / scale;
    float in_y = (y - pad_top) / scale;
    bool pad = (in_x < 0 || in_y < 0 || in_x >= img_w || in_y >= img_h);
    for (int c = 0; c < 3; ++c) {
        int oc = 2 - c;
        float v = pad ? 114.f : bilinear_interpolate(input_bgr, img_w, img_h, in_x, in_y, c);
        output_nchw[oc * out_h * out_w + y * out_w + x] = v;
    }
}

__global__ void decode_rune_kernel(
    const float* output, const GPUGridAndStride* grid_strides,
    int num_detections, const float* tf_matrix, GPURuneObject* output_objs,
    float conf_threshold
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_detections) return;

    const float* det = output + idx * 15;  // 5 pts + conf + 3+8 classes = 15
    float conf = det[10];
    if (conf < conf_threshold) return;

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
        float w  = tf_matrix[6] * x + tf_matrix[7] * y + tf_matrix[8];
        dst_x[i] = x_ / w;
        dst_y[i] = y_ / w;
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

    GPURuneObject& obj = output_objs[idx];
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
    if (i >= N) return;
    if (i >= k) objs[i].valid = 0;
}

__global__ void nms_kernel(GPURuneObject* objs, int N, float thresh) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N || !objs[i].valid) return;

    GPURuneObject& a = objs[i];

    for (int j = 0; j < i; ++j) {
        GPURuneObject& b = objs[j];
        if (!b.valid) continue;

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
CudaInfer::~CudaInfer() { release(); }

void CudaInfer::init(GPUGridAndStride* grid_strides, size_t img_bytes, int max_N) {
    d_grid_strides_ = grid_strides;
    buf_image_bytes_ = img_bytes;
    buf_max_N_ = max_N;
    CUDA_CHECK(cudaMalloc(&d_input_bgr_, buf_image_bytes_));
    CUDA_CHECK(cudaMalloc(&d_nchw_, INPUT_W * INPUT_H * 3 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_objs_, buf_max_N_ * sizeof(GPURuneObject)));
    CUDA_CHECK(cudaMalloc(&d_tf_, 9 * sizeof(float)));
}

void CudaInfer::release() {
    if (d_grid_strides_) cudaFree(d_grid_strides_), d_grid_strides_ = nullptr;
    if (d_input_bgr_) cudaFree(d_input_bgr_), d_input_bgr_ = nullptr;
    if (d_nchw_) cudaFree(d_nchw_), d_nchw_ = nullptr;
    if (d_objs_) cudaFree(d_objs_), d_objs_ = nullptr;
    if (d_tf_) cudaFree(d_tf_), d_tf_ = nullptr;
}

float* CudaInfer::preprocess(const unsigned char* input_bgr_host,
    int img_w, int img_h, Eigen::Matrix3f& tf_matrix,
    cudaStream_t stream)
{
    float scale = fminf(INPUT_W / (float)img_w, INPUT_H / (float)img_h);
    int rw = round(img_w * scale), rh = round(img_h * scale);
    int pad_l = (INPUT_W - rw) / 2, pad_t = (INPUT_H - rh) / 2;

    tf_matrix << 1.f / scale, 0, -pad_l / scale,
                 0, 1.f / scale, -pad_t / scale,
                 0, 0, 1;

    size_t img_size = img_w * img_h * 3;
    CUDA_CHECK(cudaMemcpyAsync(d_input_bgr_, input_bgr_host, img_size,
        cudaMemcpyHostToDevice, stream));

    dim3 threads(32, 32);
    dim3 blocks((INPUT_W + 31) / 32, (INPUT_H + 31) / 32);
    letterbox_kernel<<<blocks, threads, 0, stream>>>(d_input_bgr_, img_w, img_h,
        d_nchw_, INPUT_W, INPUT_H, scale, pad_t, pad_l);
    CUDA_CHECK(cudaGetLastError());
    return d_nchw_;
}
std::vector<GPURuneObject> CudaInfer::postprocess(
    const float* output, int N,
    const Eigen::Matrix3f& tf_matrix_eigen,
    float conf_th, float nms_th, int top_k
) {
    if (!output || !d_grid_strides_ || !d_tf_ || !d_objs_) {
        fprintf(stderr, "[Error] Null pointer in postprocess input\n");
        return {};
    }

    const int threads = 256;
    const int blocks = (N + threads - 1) / threads;

    // 1. 准备变换矩阵（Eigen矩阵转换成行优先float数组）
    float h_tf[9];
    Eigen::Map<Eigen::Matrix<float, 3, 3, Eigen::RowMajor>>(h_tf, 3, 3) = tf_matrix_eigen;
    CUDA_CHECK(cudaMemcpy(d_tf_, h_tf, sizeof(h_tf), cudaMemcpyHostToDevice));

    // 2. 初始化GPU对象数组
    CUDA_CHECK(cudaMemset(d_objs_, 0, sizeof(GPURuneObject) * N));

    // 3. GPU解码核函数，需要你自己实现
    decode_rune_kernel<<<blocks, threads>>>(
        output, d_grid_strides_, N,
        d_tf_, d_objs_, conf_th
    );
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // 4. 用Thrust在GPU端排序，按置信度降序
    thrust::device_ptr<GPURuneObject> dev_ptr(d_objs_);
    int sort_N = std::min(N, buf_max_N_);
    thrust::sort(thrust::device, dev_ptr, dev_ptr + sort_N, ConfidenceComparator<GPURuneObject>());

    // 5. GPU 清理非 top_k 的元素 (valid = 0)
    dim3 nmsBlocks((sort_N + threads - 1) / threads);
    clear_invalid_topk<<<nmsBlocks, threads>>>(d_objs_, sort_N, top_k);
    CUDA_CHECK(cudaGetLastError());

    // 6. GPU执行NMS
    nms_kernel<<<nmsBlocks, threads>>>(d_objs_, top_k, nms_th);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // 7. 拷贝NMS后的 top_k 结果回Host
    std::vector<GPURuneObject> h_objs(top_k);
    CUDA_CHECK(cudaMemcpy(h_objs.data(), d_objs_, sizeof(GPURuneObject) * top_k, cudaMemcpyDeviceToHost));

    // 8. 过滤valid==0的结果
    h_objs.erase(std::remove_if(h_objs.begin(), h_objs.end(),
        [](const GPURuneObject& o) { return o.valid == 0; }),
        h_objs.end());

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
    )
{

    void* input_tensor_ptr = preprocess(
        input_bgr_host,
        img_w,
        img_h,
        tf_matrix,
        stream
    );

    context->setTensorAddress("images", input_tensor_ptr);
    context->setTensorAddress("output", device_buffers[output_idx_]);


    if (!context->enqueueV3(stream)) {
        std::cerr<<"enqueueV3 failed!";
        return {};
    }

    std::vector<GPURuneObject> results = postprocess(
        (float*)device_buffers[output_idx_],
        N,
        tf_matrix,
        conf_th,
        nms_th,
        top_k
    );
    return results;
}


}