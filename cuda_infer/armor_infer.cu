// armor_cuda_infer.cu
// Optimized for robustness, RAII resource management, and proper error handling.

#include "armor_infer.hpp"
#include <thrust/device_ptr.h>
#include <thrust/sort.h>
#include <cstdio>
#include <cmath>
#include <stdexcept>
#include <vector>

// Input dimensions
static const int INPUT_W = 416;
static const int INPUT_H = 416;
// Model settings
static constexpr int NUM_CLASSES     = 8;
static constexpr int NUM_COLORS      = 4;
static constexpr float MERGE_CONF_ERROR = 0.15f;
static constexpr float MERGE_MIN_IOU    = 0.9f;

// Exception for CUDA errors
struct CudaException : public std::runtime_error {
    CudaException(const char* func, cudaError_t err)
      : std::runtime_error(std::string(func) + " failed: " + cudaGetErrorString(err)) {}
};

// Check CUDA calls and throw on error
inline void checkCuda(cudaError_t err, const char* func) {
    if (err != cudaSuccess) throw CudaException(func, err);
}

// Simple RAII wrapper for a device buffer
template<typename T>
class DeviceBuffer {
public:
    DeviceBuffer() : ptr_(nullptr), size_(0) {}
    DeviceBuffer(size_t count) : ptr_(nullptr), size_(0) {
        allocate(count);
    }
    ~DeviceBuffer() { free(); }

    void allocate(size_t count) {
        free();
        size_ = count;
        checkCuda(cudaMalloc(&ptr_, size_ * sizeof(T)), "cudaMalloc");
    }

    void free() {
        if (ptr_) {
            cudaFree(ptr_);
            ptr_ = nullptr;
            size_ = 0;
        }
    }

    T* ptr() const { return ptr_; }
    size_t size() const { return size_; }

    // Disallow copy
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;
    // Allow move
    DeviceBuffer(DeviceBuffer&& o) noexcept : ptr_(o.ptr_), size_(o.size_) { o.ptr_ = nullptr; o.size_ = 0; }
    DeviceBuffer& operator=(DeviceBuffer&& o) noexcept {
        free();
        ptr_ = o.ptr_; size_ = o.size_;
        o.ptr_ = nullptr; o.size_ = 0;
        return *this;
    }

private:
    T*    ptr_;
    size_t size_;
};

namespace armor_cuda_infer {

// Initialize grid+stride table on GPU
GPUGridAndStride* init_grid_strides_on_gpu(
    int in_w,
    int in_h,
    const std::vector<int>& strides,
    size_t& device_grid_count)
{
    std::vector<GPUGridAndStride> host_grid_strides;
    for (auto s : strides) {
        int gw = in_w / s, gh = in_h / s;
        for (int y = 0; y < gh; ++y) {
            for (int x = 0; x < gw; ++x) {
                host_grid_strides.push_back({ x, y, s });
            }
        }
    }
    device_grid_count = host_grid_strides.size();
    if (device_grid_count == 0) return nullptr;

    GPUGridAndStride* d_ptr = nullptr;
    checkCuda(cudaMalloc(&d_ptr, device_grid_count * sizeof(GPUGridAndStride)), "cudaMalloc(grid_strides)");
    checkCuda(cudaMemcpy(d_ptr, host_grid_strides.data(),
                        device_grid_count * sizeof(GPUGridAndStride),
                        cudaMemcpyHostToDevice),
              "cudaMemcpy(grid_strides)");
    return d_ptr;
}

// Bilinear interpolation helper
__device__ float bilinear_interpolate(
    const unsigned char* img, int w, int h, float x, float y, int c)
{
    x = fminf(fmaxf(x, 0.f), w - 1.f);
    y = fminf(fmaxf(y, 0.f), h - 1.f);
    int x0 = (int)floorf(x), x1 = min(x0 + 1, w - 1);
    int y0 = (int)floorf(y), y1 = min(y0 + 1, h - 1);
    float dx = x - x0, dy = y - y0;
    float v00 = img[(y0 * w + x0) * 3 + c];
    float v01 = img[(y0 * w + x1) * 3 + c];
    float v10 = img[(y1 * w + x0) * 3 + c];
    float v11 = img[(y1 * w + x1) * 3 + c];
    return (1 - dx)*(1 - dy)*v00 + dx*(1 - dy)*v01 + (1 - dx)*dy*v10 + dx*dy*v11;
}

// Letterbox kernel: convert BGR to NCHW with padding
__global__ void letterbox_kernel(
    const unsigned char* __restrict__ input_bgr,
    int img_w, int img_h,
    float* __restrict__ output_nchw,
    int out_w, int out_h,
    float scale, int pad_top, int pad_left)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= out_w || y >= out_h) return;

    float in_x = (x - pad_left) / scale;
    float in_y = (y - pad_top) / scale;
    bool pad = (in_x < 0 || in_y < 0 || in_x >= img_w || in_y >= img_h);

    for (int c = 0; c < 3; ++c) {
        int oc = 2 - c;  // BGR->RGB reorder
        float v = pad ? 114.f : bilinear_interpolate(input_bgr, img_w, img_h, in_x, in_y, c);
        output_nchw[oc * out_h * out_w + y * out_w + x] = v;
    }
}

// Argmax helper
__device__ int argmax(const float* ptr, int len) {
    float m = ptr[0]; int idx = 0;
    for (int i = 1; i < len; ++i) {
        if (ptr[i] > m) { m = ptr[i]; idx = i; }
    }
    return idx;
}

// Decode raw detections to GPUArmorObject
__global__ void decode_kernel(
    const float* __restrict__ output,
    const GPUGridAndStride* __restrict__ grid_strides,
    int N,
    const float* __restrict__ t,
    GPUArmorObject* __restrict__ results,
    float conf_thresh)
{
    extern __shared__ float tf_shared[];
    if (threadIdx.x == 0) {
        for (int i = 0; i < 9; ++i) tf_shared[i] = t[i];
    }
    __syncthreads();

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;

    const float* det = output + idx * 21;
    GPUArmorObject obj = {};
    if (det[8] < conf_thresh) {
        obj.valid = 0;
        results[idx] = obj;
        return;
    }

    auto gs = grid_strides[idx];
    float gx = gs.grid0, gy = gs.grid1, s = gs.stride;

    for (int j = 0; j < 4; ++j) {
        float px = (det[2*j] + gx) * s;
        float py = (det[2*j+1] + gy) * s;
        float w = tf_shared[6]*px + tf_shared[7]*py + tf_shared[8];
        if (fabsf(w) < 1e-6f) w = 1e-6f;
        obj.x[j] = (tf_shared[0]*px + tf_shared[1]*py + tf_shared[2]) / w;
        obj.y[j] = (tf_shared[3]*px + tf_shared[4]*py + tf_shared[5]) / w;
    }

    obj.confidence = det[8];
    obj.color_id   = argmax(det + 9, NUM_COLORS);
    obj.number_id  = argmax(det + 9 + NUM_COLORS, NUM_CLASSES);
    obj.valid      = 1;
    obj.num_pts    = 4;
    results[idx]   = obj;
}

// Clear any detections beyond top_k
__global__ void clear_invalid_topk(GPUArmorObject* objs, int N, int k) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;
    if (idx >= k) objs[idx].valid = 0;
}

// NMS + merge
__global__ void nms_kernel(GPUArmorObject* objs, int N, float thresh) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N || !objs[i].valid) return;

    auto& a = objs[i];
    for (int j = 0; j < i; ++j) {
        auto& b = objs[j];
        if (!b.valid) continue;
        if (a.color_id != b.color_id || a.number_id != b.number_id) continue;

        // IoU
        float x1 = fmaxf(fminf(a.x[0], a.x[2]), fminf(b.x[0], b.x[2]));
        float y1 = fmaxf(fminf(a.y[0], a.y[2]), fminf(b.y[0], b.y[2]));
        float x2 = fminf(fmaxf(a.x[0], a.x[2]), fmaxf(b.x[0], b.x[2]));
        float y2 = fminf(fmaxf(a.y[0], a.y[2]), fmaxf(b.y[0], b.y[2]));
        float inter = fmaxf(0.f, x2 - x1) * fmaxf(0.f, y2 - y1);
        float areaA = fabsf((a.x[2] - a.x[0])*(a.y[2] - a.y[0]));
        float areaB = fabsf((b.x[2] - b.x[0])*(b.y[2] - b.y[0]));
        float u = areaA + areaB - inter + 1e-6f;
        float iou = inter / u;

        if (iou > thresh) {
            a.valid = 0;
            break;
        }
        // Merge if very close and confidences similar
        if (iou > MERGE_MIN_IOU && fabsf(a.confidence - b.confidence) < MERGE_CONF_ERROR) {
            if (b.num_pts + 4 <= 4) {
                for (int k = 0; k < 4; ++k) {
                    b.x[b.num_pts + k] = a.x[k];
                    b.y[b.num_pts + k] = a.y[k];
                }
                b.num_pts += 4;
            }
        }
    }
}

// CudaInfer implementation
class CudaInfer::Impl {
public:
    GPUGridAndStride*     d_grid_strides_ = nullptr;
    DeviceBuffer<unsigned char> d_input_bgr_;
    DeviceBuffer<float>        d_nchw_;
    DeviceBuffer<GPUArmorObject> d_objs_;
    DeviceBuffer<float>        d_tf_;
    size_t buf_image_bytes_ = 0;
    int    buf_max_N_       = 0;

    ~Impl() { release(); }

    void init(GPUGridAndStride* grid_strides, size_t img_bytes, int max_N) {
        d_grid_strides_  = grid_strides;
        buf_image_bytes_ = img_bytes;
        buf_max_N_       = max_N;
        d_nchw_.allocate(INPUT_W * INPUT_H * 3);
        d_objs_.allocate(buf_max_N_);
        d_tf_.allocate(9);
    }

    void release() {
        if (d_grid_strides_) {
            cudaFree(d_grid_strides_);
            d_grid_strides_ = nullptr;
        }
        d_input_bgr_.free();
        d_nchw_.free();
        d_objs_.free();
        d_tf_.free();
    }

    float* preprocess(const unsigned char* input_bgr_host,
                      int img_w, int img_h,
                      Eigen::Matrix3f& tf_matrix,
                      cudaStream_t stream)
    {
        if (img_w <= 0 || img_h <= 0) {
            throw std::invalid_argument("Invalid image dimensions");
        }
        buf_image_bytes_ = static_cast<size_t>(img_w) * img_h * 3;
        d_input_bgr_.allocate(buf_image_bytes_);

        float scale = fminf(INPUT_W / float(img_w), INPUT_H / float(img_h));
        int rw = int(round(img_w * scale)), rh = int(round(img_h * scale));
        int pad_l = (INPUT_W - rw) / 2, pad_t = (INPUT_H - rh) / 2;

        tf_matrix << 1.f/scale, 0, -pad_l/scale,
                     0, 1.f/scale, -pad_t/scale,
                     0, 0, 1;

        // copy input
        checkCuda(cudaMemcpyAsync(
            d_input_bgr_.ptr(), input_bgr_host, buf_image_bytes_,
            cudaMemcpyHostToDevice, stream),
            "cudaMemcpyAsync(input_bgr)");

        dim3 threads(32,32);
        dim3 blocks((INPUT_W+31)/32, (INPUT_H+31)/32);
        letterbox_kernel<<<blocks,threads,0,stream>>>(
            d_input_bgr_.ptr(), img_w, img_h,
            d_nchw_.ptr(),
            INPUT_W, INPUT_H,
            scale, pad_t, pad_l);
        checkCuda(cudaGetLastError(), "letterbox_kernel");

        return d_nchw_.ptr();
    }

    std::vector<GPUArmorObject> postprocess(
        const float* output, int N,
        const Eigen::Matrix3f& tfm,
        float conf_th, float nms_th, int top_k)
    {
        if (!output || !d_grid_strides_) {
            throw std::runtime_error("Null pointer in postprocess inputs");
        }
        if (N <= 0 || top_k <= 0 || top_k > buf_max_N_ || N > buf_max_N_) {
            throw std::invalid_argument("Invalid N/top_k in postprocess");
        }

        // copy transform matrix
        float h_tf[9] = {
            tfm(0,0), tfm(0,1), tfm(0,2),
            tfm(1,0), tfm(1,1), tfm(1,2),
            tfm(2,0), tfm(2,1), tfm(2,2)
        };
        checkCuda(cudaMemcpy(
            d_tf_.ptr(), h_tf, sizeof(h_tf), cudaMemcpyHostToDevice),
            "cudaMemcpy(tf)");

        // decode
        int threads = 256;
        int blocks  = (N + threads - 1) / threads;
        size_t shared_bytes = 9 * sizeof(float);
        decode_kernel<<<blocks,threads,shared_bytes>>>(
            output, d_grid_strides_, N, d_tf_.ptr(), d_objs_.ptr(), conf_th);
        checkCuda(cudaGetLastError(), "decode_kernel");

        // sort by confidence
        thrust::device_ptr<GPUArmorObject> dev_ptr(d_objs_.ptr());
        thrust::sort(thrust::device, dev_ptr, dev_ptr + N,
                     ConfidenceComparator());

        // keep top_k
        clear_invalid_topk<<<blocks,threads>>>(d_objs_.ptr(), N, top_k);
        checkCuda(cudaGetLastError(), "clear_invalid_topk");

        // NMS + merge
        nms_kernel<<<blocks,threads>>>(d_objs_.ptr(), top_k, nms_th);
        checkCuda(cudaGetLastError(), "nms_kernel");

        // copy back
        std::vector<GPUArmorObject> results(top_k);
        checkCuda(cudaMemcpy(
            results.data(), d_objs_.ptr(),
            top_k * sizeof(GPUArmorObject),
            cudaMemcpyDeviceToHost),
            "cudaMemcpy(results)");

        return results;
    }
};

CudaInfer::CudaInfer() : impl_(new Impl()) {}
CudaInfer::~CudaInfer() {  }

void CudaInfer::init(GPUGridAndStride* grid_strides, size_t img_bytes, int max_N) {
    impl_->init(grid_strides, img_bytes, max_N);
}

void CudaInfer::release() {
    impl_->release();
}

float* CudaInfer::preprocess(
    const unsigned char* input_bgr_host,
    int img_w, int img_h,
    Eigen::Matrix3f& tf_matrix,
    cudaStream_t stream)
{
    return impl_->preprocess(
        input_bgr_host, img_w, img_h, tf_matrix, stream);
}

std::vector<GPUArmorObject> CudaInfer::postprocess(
    const float* output, int N,
    const Eigen::Matrix3f& tfm,
    float conf_th, float nms_th, int top_k)
{
    return impl_->postprocess(output, N, tfm, conf_th, nms_th, top_k);
}

std::vector<GPUArmorObject> CudaInfer::process_trt(
    nvinfer1::IExecutionContext* context,
    void* device_buffers[2],
    int input_idx,
    int output_idx,
    const unsigned char* input_bgr_host,
    int img_w, int img_h,
    Eigen::Matrix3f& tf_matrix,
    cudaStream_t stream,
    int N, float conf_th, float nms_th, int top_k)
{
    // preprocess
    float* input_dev = preprocess(input_bgr_host, img_w, img_h, tf_matrix, stream);
    cudaStreamSynchronize(stream);
    context->setTensorAddress("images", input_dev);
    context->setTensorAddress("output", device_buffers[output_idx]);

    if (!context->enqueueV3(stream)) {
        throw std::runtime_error("TensorRT enqueueV3 failed");
    }

    return postprocess(
        reinterpret_cast<const float*>(device_buffers[output_idx]),
        N, tf_matrix, conf_th, nms_th, top_k);
}

} // namespace armor_cuda_infer
