from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='test_kernel',
    ext_modules=[
        CUDAExtension('test_kernel', ['src/mathutil_cuda.cpp', 'src/mathutil_cuda_kernel.cu']),    
    ],
    cmdclass={'build_ext': BuildExtension}
)