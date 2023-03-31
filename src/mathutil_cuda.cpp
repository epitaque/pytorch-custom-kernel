// #include <THC/THC.h>
// #include "mathutil_cuda_kernel.h"

// extern THCState *state;

// int broadcast_sum(THCudaTensor *a_tensor, THCudaTensor *b_tensor, int x, int y)
// {
//     float *a = THCudaTensor_data(state, a_tensor);
//     float *b = THCudaTensor_data(state, b_tensor);
//     cudaStream_t stream = THCState_getCurrentStream(state);

//     broadcast_sum_cuda(a, b, x, y, stream);

//     return 1;
// }

#include <torch/all.h>
#include <torch/python.h>
#include <c10/cuda/CUDAGuard.h>

void vecadd_cuda(
  torch::Tensor x, torch::Tensor y, torch::Tensor z
);

void vecadd(
  torch::Tensor x, torch::Tensor y, torch::Tensor z
) {
  const at::cuda::OptionalCUDAGuard device_guard(device_of(x));
  vecadd_cuda(x, y, z);
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("vecadd", &vecadd, "vecadd (CUDA)");
}