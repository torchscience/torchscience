# Copyright (c) 2024
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import glob
import os
import sys

import torch
from setuptools import find_packages, setup
from torch.utils.cpp_extension import (
    CUDA_HOME,
    BuildExtension,
    CppExtension,
    CUDAExtension,
)

library_name = "torchscience"

py_limited_api = torch.__version__ >= "2.6.0"


def get_extensions():
    debug_mode = os.getenv("DEBUG", "0") == "1"
    use_cuda = os.getenv("USE_CUDA", "1") == "1"
    use_rocm = os.getenv("USE_ROCM", "0") == "1"

    if debug_mode:
        print("Compiling in debug mode")

    # Check for CUDA availability
    use_cuda = use_cuda and torch.cuda.is_available() and CUDA_HOME is not None

    # Check for ROCm availability (AMD GPUs)
    rocm_home = os.getenv("ROCM_HOME")
    use_rocm = use_rocm and torch.version.hip is not None and rocm_home is not None

    if use_cuda:
        print("Building with CUDA support")
    if use_rocm:
        print("Building with ROCm/HIP support")

    extension = CUDAExtension if use_cuda else CppExtension

    # Detect platform
    is_macos = sys.platform == "darwin"

    # Configure compiler flags based on platform
    if is_macos:
        # macOS-specific configuration (Homebrew LLVM, SDK paths)
        sdk_path = "/Applications/Xcode.app/Contents/Developer/Platforms/MacOSX.platform/Developer/SDKs/MacOSX.sdk"
        extra_link_args = [
            "-isysroot",
            sdk_path,
            "-L/opt/homebrew/opt/llvm/lib/c++",
            "-Wl,-rpath,/opt/homebrew/opt/llvm/lib/c++",
        ]
        extra_compile_args = {
            "cxx": [
                "-O3" if not debug_mode else "-O0",
                "-fdiagnostics-color=always",
                "-DPy_LIMITED_API=0x030A0000",  # min CPython version 3.10
                "-isysroot",
                sdk_path,
            ],
            "nvcc": [
                "-O3" if not debug_mode else "-O0",
            ],
        }
        if debug_mode:
            extra_compile_args["cxx"].append("-g")
            extra_compile_args["nvcc"].append("-g")
            extra_link_args.extend(["-O0", "-g"])
    else:
        # Linux/Windows - use default compiler detection
        extra_link_args = []
        extra_compile_args = {
            "cxx": [
                "-O3" if not debug_mode else "-O0",
                "-fdiagnostics-color=always",
                "-DPy_LIMITED_API=0x030A0000",  # min CPython version 3.10
            ],
            "nvcc": [
                "-O3" if not debug_mode else "-O0",
            ],
        }
        if debug_mode:
            extra_compile_args["cxx"].append("-g")
            extra_compile_args["nvcc"].append("-g")
            extra_link_args.extend(["-O0", "-g"])

    extensions_dir = os.path.join("src", library_name, "csrc")

    # Collect all source files
    sources = list(glob.glob(os.path.join(extensions_dir, "*.cpp")))

    # Add operator implementation sources
    sources += list(glob.glob(os.path.join(extensions_dir, "ops", "*.cpp")))
    sources += list(glob.glob(os.path.join(extensions_dir, "ops", "cpu", "*.cpp")))
    sources += list(glob.glob(os.path.join(extensions_dir, "ops", "autograd", "*.cpp")))
    sources += list(glob.glob(os.path.join(extensions_dir, "ops", "autocast", "*.cpp")))
    sources += list(glob.glob(os.path.join(extensions_dir, "ops", "meta", "*.cpp")))
    sources += list(glob.glob(os.path.join(extensions_dir, "ops", "quantized", "cpu", "*.cpp")))

    # Sparse CPU backend (always included)
    sources += list(glob.glob(os.path.join(extensions_dir, "ops", "sparse", "cpu", "*.cpp")))

    # MPS backend (Apple Silicon) - only on macOS
    # Note: MPS implementation requires macOS 12.0+ at runtime
    if is_macos:
        mps_sources = list(glob.glob(os.path.join(extensions_dir, "ops", "mps", "*.mm")))
        sources += mps_sources

    # CUDA sources (including sparse CUDA)
    extensions_cuda_dir = os.path.join(extensions_dir, "cuda")
    cuda_sources = list(glob.glob(os.path.join(extensions_cuda_dir, "*.cu")))
    cuda_sources += list(glob.glob(os.path.join(extensions_dir, "ops", "cuda", "*.cu")))
    cuda_sources += list(glob.glob(os.path.join(extensions_dir, "ops", "sparse", "cuda", "*.cu")))

    if use_cuda:
        sources += cuda_sources

    # HIP/ROCm sources (AMD GPUs)
    if use_rocm:
        hip_sources = list(glob.glob(os.path.join(extensions_dir, "ops", "hip", "*.cpp")))
        sources += hip_sources
        # Add ROCm-specific compile flags
        extra_compile_args["cxx"].append("-DUSE_ROCM")

    # Build extension kwargs
    ext_kwargs = {
        "sources": sources,
        "extra_compile_args": extra_compile_args,
        "extra_link_args": extra_link_args,
        "py_limited_api": py_limited_api,
    }

    ext_modules = [extension(f"{library_name}._C", **ext_kwargs)]

    return ext_modules


def read_readme():
    with open("README.md") as f:
        return f.read()


setup(
    name=library_name,
    version="0.0.1",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    ext_modules=get_extensions(),
    install_requires=["torch"],
    description="Example of PyTorch C++ and CUDA extensions",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/0x00b1/torch-science",
    cmdclass={"build_ext": BuildExtension},
    options={"bdist_wheel": {"py_limited_api": "cp310"}} if py_limited_api else {},
    python_requires=">=3.10",
)
