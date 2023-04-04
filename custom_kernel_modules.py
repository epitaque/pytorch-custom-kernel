import torch
from torch.utils.cpp_extension import load

test_kernel = load(name="test_kernel", sources=["src/mathutil_cuda.cpp", "src/mathutil_cuda_kernel.cu"])

class CudaLinear(torch.nn.Linear):
    def forward(self, x):
        # y will be (*, out_features)
        x_flat = x.view(-1, x.shape[-1])
        y_shape = (*x.shape[:-1], self.out_features)
        y = torch.zeros(y_shape, device='cuda')
        test_kernel.linear(x_flat, self.weight, self.bias, y)
        return y