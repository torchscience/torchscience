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

    if debug_mode:
        print("Compiling in debug mode")

    # Check for CUDA availability
    use_cuda = use_cuda and torch.cuda.is_available() and CUDA_HOME is not None

    extension = CUDAExtension if use_cuda else CppExtension

    # Detect platform
    is_macos = sys.platform == "darwin"

    # Configure compiler flags
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

    # Define WITH_CUDA when building with CUDA support
    if use_cuda:
        extra_compile_args["cxx"].append("-DWITH_CUDA")
        extra_compile_args["nvcc"].append("-DWITH_CUDA")

    # macOS-specific configuration
    extra_link_args = []
    if is_macos:
        sdk_path = "/Applications/Xcode.app/Contents/Developer/Platforms/MacOSX.platform/Developer/SDKs/MacOSX.sdk"
        homebrew_prefix = "/opt/homebrew"  # Apple Silicon default

        # Add SDK and libc++ paths
        extra_compile_args["cxx"].extend(
            [
                "-isysroot",
                sdk_path,
            ]
        )
        extra_link_args = [
            "-isysroot",
            sdk_path,
            f"-L{homebrew_prefix}/opt/llvm/lib/c++",
            f"-Wl,-rpath,{homebrew_prefix}/opt/llvm/lib/c++",
        ]

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

    # Build extension kwargs
    ext_kwargs = {
        "sources": sources,
        "extra_compile_args": extra_compile_args,
        "py_limited_api": py_limited_api,
    }

    # Add macOS-specific link args
    if extra_link_args:
        ext_kwargs["extra_link_args"] = extra_link_args

    return [
        extension(
            f"{library_name}._C",
            **ext_kwargs,
        ),
    ]


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
    url="https://github.com/torchscience/torchscience",
    cmdclass={"build_ext": BuildExtension},
    options={"bdist_wheel": {"py_limited_api": "cp310"}} if py_limited_api else {},
    python_requires=">=3.10",
)
