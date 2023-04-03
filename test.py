from regex import W
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load

test_kernel = load(name="test_kernel", sources=["src/mathutil_cuda.cpp", "src/mathutil_cuda_kernel.cu"])

W = torch.zeros(2, 2, device='cuda')
x = torch.randn(2, 2, device='cuda')
b = torch.tensor([1., 0.], device='cuda')
# stack bias twice
b_stacked = torch.hstack([b, b])
print(f"b_stacked = {b_stacked}")
y = torch.zeros(2, 2, device='cuda')

print(f"Before: W = {W}\nx = {x}\nb = {b}\ny = {y}")
pytorch_linear = nn.Linear(2, 2, bias=True)
pytorch_linear.weight = nn.Parameter(W)
pytorch_linear.bias = nn.Parameter(b)


test_kernel.linear(W, x, b, y)

print(f"After: y = {y}") 
print(f"output should be {torch.matmul(x,W.t())+b}")
print(f"Pytorch output: {pytorch_linear(x)}")