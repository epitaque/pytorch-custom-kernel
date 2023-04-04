#include <torch/all.h>
#include <torch/python.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>

#include <ATen/cuda/CUDAContext.h>

template <typename scalar_t>
__global__ void LinearKernel(
    const scalar_t* __restrict__ input,
    const scalar_t* __restrict__ weight,
    const scalar_t* __restrict__ bias,
          scalar_t* __restrict__ output,
    int batch_size,
    int input_size,
    int output_size
);

const int BLOCKWIDTH  = 16;
const int BLOCKHEIGHT = 16;

void linear_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    torch::Tensor output
  ) {
  int batch_size = input.size(0);
  int input_size = input.size(1);
  int output_size = weight.size(0);
  printf("Hello output size  %d\n", output_size);
  dim3 blocks((batch_size + BLOCKWIDTH - 1) / BLOCKWIDTH, (output_size + BLOCKHEIGHT - 1) / BLOCKHEIGHT);
  dim3 threads(BLOCKWIDTH, BLOCKHEIGHT);

  AT_DISPATCH_FLOATING_TYPES(
    input.type(), "linear_cuda", ([&] {
      LinearKernel<<<blocks, threads>>>(
        input.data<scalar_t>(),
        weight.data<scalar_t>(),
        bias.data<scalar_t>(),
        output.data<scalar_t>(),
        batch_size,
        input_size,
        output_size
      );
    })
  );

  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  cudaStreamSynchronize(stream);
}

template <typename scalar_t>
__global__ void LinearKernel(
    const scalar_t* __restrict__ input,
    const scalar_t* __restrict__ weight,
    const scalar_t* __restrict__ bias,
          scalar_t* __restrict__ output,
    int batch_size,
    int input_size,
    int output_size
) {
  // index of the batch, i.e. if we have 2 batches, this will be 0 or 1
  int batch_idx = blockIdx.x * blockDim.x + threadIdx.x;
  // index of the output, i.e. if we have 4 outputs, this will be 0,1,2,3
  int output_idx = blockIdx.y * blockDim.y + threadIdx.y;

  if (batch_idx < batch_size && output_idx < output_size) {
    scalar_t value = bias[output_idx];

    for (int input_idx = 0; input_idx < input_size; ++input_idx) {
      value += input[batch_idx * input_size + input_idx] * weight[output_idx * input_size + input_idx];
    }
    output[batch_idx * output_size + output_idx] = value;
    // output[batch_idx * output_size + output_idx] = scalar_t(output_size);
  }
}

// note to future self: I think this kernel is expecting batches. that's why it's all transposed.