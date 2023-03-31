#include <torch/all.h>
#include <torch/python.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

template <typename scalar_t>
__global__ void VecAddKernel(
    const  scalar_t* __restrict__ x,
           scalar_t* __restrict__ y,
    const  scalar_t* __restrict__ z,
    int height,
    int width
);

const int BLOCKWIDTH  = 256;
const int BLOCKHEIGHT =  24;

void vecadd_cuda(
  torch::Tensor x,
  torch::Tensor y,
  torch::Tensor z
) {
  int height = x.size(0);
  int width = x.size(1);

  dim3 blocks(
    (height + BLOCKHEIGHT - 1) / BLOCKHEIGHT,
    (width + BLOCKWIDTH - 1) / BLOCKWIDTH
  );
  dim3 threads(BLOCKWIDTH);

  AT_DISPATCH_FLOATING_TYPES(
    x.type(), "vecadd_cuda", ([&] {
      VecAddKernel<<<blocks, threads>>>(
        x.data<scalar_t>(),
        y.data<scalar_t>(),
        z.data<scalar_t>(),
        height,
        width
      );
    })
  );
}


template <typename scalar_t>
__global__ void VecAddKernel(
    const  scalar_t* __restrict__ x,
           scalar_t* __restrict__ y,
    const  scalar_t* __restrict__ z,
    int height,
    int width
) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;

  if (i < height && j < width) {
    y[i * width + j] = x[i * width + j] + z[i * width + j];
  }
}