#ifdef GOOGLE_CUDA

#define EIGEN_USE_GPU
#define EIGEN_USE_THREADS

// Disabling all warnings from Tensorflow
#pragma warning(push, 0)
#include "tf_kernel.h"
#include "tensorflow/core/util/cuda_kernel_helper.h"
#pragma warning(pop)

using namespace tensorflow;

#define EIGEN_USE_GPU

// Define the CUDA kernel.
template <typename T>
__global__ void InteractiveCudaKernel(const int size, const T* in, T* out) {
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < size;
       i += blockDim.x * gridDim.x) {
    out[i] = 2 * in[i];
  }
}

// Define the GPU implementation that launches the CUDA kernel.
template <typename T>
struct InteractiveFunctor<GPUDevice, T> {
  void operator()(const GPUDevice& d, int size, const T* in, T* out) {
    // Launch the cuda kernel.
    //
    // See core/util/cuda_kernel_helper.h for example of computing
    // block count and thread_per_block count.
    int block_count = 1024;
    int thread_per_block = 20;
    InteractiveCudaKernel<T>
        <<<block_count, thread_per_block, 0, d.stream()>>>(size, in, out);
  }
};

// Instantiate functors for the types of OpKernels registered.
typedef Eigen::GpuDevice GPUDevice;
template struct InteractiveFunctor<GPUDevice, float>;

#endif  // GOOGLE_CUDA