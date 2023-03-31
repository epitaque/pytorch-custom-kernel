import torch
import test_kernel

x = torch.randn(1, 1, device='cuda')
y = torch.randn(1, 1, device='cuda')
z = torch.randn(1, 1, device='cuda')

print(f"Before: x = {x}, y = {y}, z = {z}")

test_kernel.vecadd(x, y, z)

print(f"After: x = {x}, y = {y}, z = {z}")