import os

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

COMPUTE_CAP = os.environ.get('COMPUTE_CAP', '90')

setup(
    ext_modules=[
        CUDAExtension(
            name="minisgl.shadowkv_kernels",
            sources=[
                "python/minisgl/kernel/shadowkv/csrc/lib.cpp",
                "python/minisgl/kernel/higgs/csrc/higgs_quantizer.cpp",
                "python/minisgl/kernel/higgs/csrc/higgs_quantizer_cuda.cu",
                "python/minisgl/kernel/higgs/csrc/higgs_quantization_cuda.cu",
                "python/minisgl/kernel/higgs/csrc/higgs_dequantization_full.cu",
                "python/minisgl/kernel/shadowkv/csrc/gather_kv_cache.cu",
                "python/minisgl/kernel/shadowkv/csrc/metadata_kernels.cu",
                "python/minisgl/kernel/shadowkv/csrc/shadowkv.cpp",
            ],
            # include_dirs=[os.path.abspath("python/minisgl/kernels/shadowkv/csrc/include")],
            extra_compile_args={
            "cxx": ["-O3", "-DTORCH_USE_CUDA_DSA", "-w", '-lineinfo'],
            "nvcc": ["-O3",
                     "-DTORCH_USE_CUDA_DSA",
                     "-w",
                     "-U__CUDA_NO_BFLOAT16_OPERATORS__",
                     "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
                     "-U__CUDA_NO_BFLOAT162_OPERATORS__",
                     "-U__CUDA_NO_BFLOAT162_CONVERSIONS__",
                     "--expt-relaxed-constexpr",
                     "--expt-extended-lambda",
                     f"-gencode=arch=compute_{COMPUTE_CAP},code=\"sm_{COMPUTE_CAP}\"",
                     "-Xptxas=-v"]
        },
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)

