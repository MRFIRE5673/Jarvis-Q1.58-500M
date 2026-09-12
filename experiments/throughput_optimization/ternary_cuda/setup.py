# experiments/throughput_optimization/ternary_cuda/setup.py
import os
import sys
import glob
import shutil
import subprocess
import torch
import torch.utils.cpp_extension as cpp_ext
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

# 1. Automatically configure MSVC environment if cl.exe is not in PATH
if not shutil.which("cl"):
    vcvars_candidates = [
        r"C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat",
        r"C:\Program Files\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat",
        r"C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat",
        r"C:\Program Files (x86)\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat",
    ]
    for vcvars in vcvars_candidates:
        if os.path.exists(vcvars):
            print(f"[setup.py] Initializing MSVC environment via {vcvars}")
            cmd = f'"{vcvars}" && set'
            out = subprocess.check_output(cmd, shell=True, text=True)
            for line in out.splitlines():
                if "=" in line:
                    k, _, v = line.partition("=")
                    os.environ[k] = v
            os.environ["DISTUTILS_USE_SDK"] = "1"
            break

# 2. Detect or configure CUDA_HOME
cuda_home = os.environ.get("CUDA_HOME") or os.environ.get("CUDA_PATH")
if not cuda_home:
    cuda_candidates = sorted(
        glob.glob(r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v*"),
        reverse=True
    )
    if cuda_candidates:
        cuda_home = cuda_candidates[0]
        os.environ["CUDA_HOME"] = cuda_home
        os.environ["CUDA_PATH"] = cuda_home

if cuda_home:
    cuda_bin = os.path.join(cuda_home, "bin")
    if cuda_bin not in os.environ.get("PATH", ""):
        os.environ["PATH"] = cuda_bin + os.pathsep + os.environ.get("PATH", "")

# 3. Relax PyTorch major-version check between PyTorch and installed CUDA Toolkit
orig_check = cpp_ext._check_cuda_version
def relaxed_check(compiler_name, compiler_version):
    try:
        orig_check(compiler_name, compiler_version)
    except RuntimeError as e:
        print(f"[setup.py] Note: Relaxed CUDA version check ({e}).")
cpp_ext._check_cuda_version = relaxed_check

# 4. Architecture and preprocessor flags for RTX 5070 (Blackwell sm_120)
extra_cuda_cflags = [
    "-O3",
    "--use_fast_math",
    "-gencode=arch=compute_120,code=sm_120",
    "-Xcompiler", "/Zc:preprocessor",
    "-DCCCL_IGNORE_MSVC_TRADITIONAL_PREPROCESSOR_WARNING",
    "--ptxas-options=-v",
]

extra_compile_args = {
    "cxx": ["/O2", "/std:c++17", "/Zc:preprocessor", "-DCCCL_IGNORE_MSVC_TRADITIONAL_PREPROCESSOR_WARNING"],
    "nvcc": extra_cuda_cflags,
}

cur_dir = os.path.dirname(os.path.abspath(__file__))

setup(
    name="ternary_gemm_cuda",
    ext_modules=[
        CUDAExtension(
            name="ternary_gemm_cuda",
            sources=[
                os.path.join(cur_dir, "ternary_gemm_cpp.cpp"),
                os.path.join(cur_dir, "ternary_gemm_cuda.cu"),
            ],
            extra_compile_args=extra_compile_args,
        ),
    ],
    cmdclass={"build_ext": BuildExtension}
)
