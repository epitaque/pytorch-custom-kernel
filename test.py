from regex import W
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load

import custom_kernel_modules as ckm
#test_kernel = load(name="test_kernel", sources=["src/mathutil_cuda.cpp", "src/mathutil_cuda_kernel.cu"])
# (m x n) * (n x p) = (m x p)

# 3 inputs, 4 outputs (because W is transposed)
W = torch.randn(4, 3, device='cuda')
# 3x2 samples, 3 input features
x = torch.randn(3, 2, 3, device='cuda')
b = torch.tensor([[1., 0., 1.0, 1.0]], device='cuda')
y = torch.zeros(3, 2, 4, device='cuda')

print(f"Before: W = {W}\nx = {x}\nb = {b}\ny = {y}")
pytorch_linear = nn.Linear(3, 4, bias=True)
pytorch_linear.weight = nn.Parameter(W)
pytorch_linear.bias = nn.Parameter(b)

kernel_linear = ckm.CudaLinear(3, 4, bias=True)
kernel_linear.weight = nn.Parameter(W)
kernel_linear.bias = nn.Parameter(b)

print(f"After: kernel_linear(x): {kernel_linear(x)}") 
print(f"torch.matmul output: {torch.matmul(x,W.t())+b}")
print(f"pytorch_linear output: {pytorch_linear(x)}")