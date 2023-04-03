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
#include <cuda_runtime.h>

void linear_cuda(
  torch::Tensor input, torch::Tensor weight, torch::Tensor bias, torch::Tensor output
);

void linear(
  torch::Tensor input, torch::Tensor weight, torch::Tensor bias, torch::Tensor output
) {
  const at::cuda::OptionalCUDAGuard device_guard(device_of(input));
  linear_cuda(input, weight, bias, output);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("linear", &linear, "linear (CUDA)");
}