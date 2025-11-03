import distutils.command.clean
import distutils.spawn
import glob
import os
import shlex
import shutil
import subprocess
import sys
import sysconfig
from pathlib import Path

import torch
from pkg_resources import DistributionNotFound, get_distribution, parse_version
from setuptools import find_packages, setup
from torch.utils.cpp_extension import BuildExtension, CppExtension, CUDA_HOME, CUDAExtension, ROCM_HOME

FORCE_CUDA = os.getenv("FORCE_CUDA", "0") == "1"
FORCE_MPS = os.getenv("FORCE_MPS", "0") == "1"
DEBUG = os.getenv("DEBUG", "0") == "1"

NVCC_FLAGS = os.getenv("NVCC_FLAGS", None)

TORCHSCIENCE_INCLUDE = os.environ.get("TORCHSCIENCE_INCLUDE", "")
TORCHSCIENCE_LIBRARY = os.environ.get("TORCHSCIENCE_LIBRARY", "")
TORCHSCIENCE_INCLUDE = TORCHSCIENCE_INCLUDE.split(os.pathsep) if TORCHSCIENCE_INCLUDE else []
TORCHSCIENCE_LIBRARY = TORCHSCIENCE_LIBRARY.split(os.pathsep) if TORCHSCIENCE_LIBRARY else []

ROOT_DIR = Path(__file__).absolute().parent
CSRS_DIR = ROOT_DIR / "torchscience/csrc"
IS_ROCM = (torch.version.hip is not None) and (ROCM_HOME is not None)
BUILD_CUDA_SOURCES = (torch.cuda.is_available() and ((CUDA_HOME is not None) or IS_ROCM)) or FORCE_CUDA

package_name = os.getenv("TORCHSCIENCE_PACKAGE_NAME", "torchscience")

print("torch-science build configuration:")
print(f"{FORCE_CUDA = }")
print(f"{FORCE_MPS = }")
print(f"{DEBUG = }")
print(f"{NVCC_FLAGS = }")
print(f"{TORCHSCIENCE_INCLUDE = }")
print(f"{TORCHSCIENCE_LIBRARY = }")
print(f"{IS_ROCM = }")
print(f"{BUILD_CUDA_SOURCES = }")


def get_version():
    with open(ROOT_DIR / "VERSION") as f:
        version = f.readline().strip()
    sha = "Unknown"

    try:
        sha = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=str(ROOT_DIR)).decode("ascii").strip()
    except Exception:
        pass

    if os.getenv("BUILD_VERSION"):
        version = os.getenv("BUILD_VERSION")
    elif sha != "Unknown":
        version += "+" + sha[:7]

    return version, sha


def write_version_file(version, sha):
    # Exists for BC, probably completely useless.
    with open(ROOT_DIR / "src" / "torchscience" / "version.py", "w") as f:
        f.write(f"__version__ = '{version}'\n")
        f.write(f"git_version = {repr(sha)}\n")
        f.write("from torchscience.extension import _check_cuda_version\n")
        f.write("if _check_cuda_version() > 0:\n")
        f.write("    cuda = _check_cuda_version()\n")


def get_requirements():
    def get_dist(pkgname):
        try:
            return get_distribution(pkgname)
        except DistributionNotFound:
            return None

    pytorch_dep = os.getenv("TORCH_PACKAGE_NAME", "torch")
    if version_pin := os.getenv("PYTORCH_VERSION"):
        pytorch_dep += "==" + version_pin
    elif (version_pin_ge := os.getenv("PYTORCH_VERSION_GE")) and (version_pin_lt := os.getenv("PYTORCH_VERSION_LT")):
        # This branch and the associated env vars exist to help third-party
        # builds like in https://github.com/pytorch/vision/pull/8936. This is
        # supported on a best-effort basis, we don't guarantee that this won't
        # eventually break (and we don't test it.)
        pytorch_dep += f">={version_pin_ge},<{version_pin_lt}"

    requirements = [
        "numpy",
        pytorch_dep,
    ]

    # Excluding 8.3.* because of https://github.com/pytorch/vision/issues/4934
    pillow_ver = " >= 5.3.0, !=8.3.*"
    pillow_req = "pillow-simd" if get_dist("pillow-simd") is not None else "pillow"
    requirements.append(pillow_req + pillow_ver)

    return requirements


def get_macros_and_flags():
    define_macros = []
    extra_compile_args = {"cxx": []}
    if BUILD_CUDA_SOURCES:
        if IS_ROCM:
            define_macros += [("WITH_HIP", None)]
            nvcc_flags = []
        else:
            define_macros += [("WITH_CUDA", None)]
            if NVCC_FLAGS is None:
                nvcc_flags = []
            else:
                nvcc_flags = shlex.split(NVCC_FLAGS)
        extra_compile_args["nvcc"] = nvcc_flags

    if sys.platform == "win32":
        define_macros += [("torchscience_EXPORTS", None)]
        extra_compile_args["cxx"].append("/MP")
        if sysconfig.get_config_var("Py_GIL_DISABLED"):
            extra_compile_args["cxx"].append("-DPy_GIL_DISABLED")

    if DEBUG:
        extra_compile_args["cxx"].append("-g")
        extra_compile_args["cxx"].append("-O0")
        if "nvcc" in extra_compile_args:
            # we have to remove "-OX" and "-g" flag if exists and append
            nvcc_flags = extra_compile_args["nvcc"]
            extra_compile_args["nvcc"] = [f for f in nvcc_flags if not ("-O" in f or "-g" in f)]
            extra_compile_args["nvcc"].append("-O0")
            extra_compile_args["nvcc"].append("-g")
    else:
        extra_compile_args["cxx"].append("-g0")

    return define_macros, extra_compile_args


def make_extension():
    print("Building _C extension")

    sources = (
        list(CSRS_DIR.glob("*.cpp"))
        + list(CSRS_DIR.glob("ops/*.cpp"))
        + list(CSRS_DIR.glob("ops/autocast/*.cpp"))
        + list(CSRS_DIR.glob("ops/autograd/*.cpp"))
        + list(CSRS_DIR.glob("ops/cpu/*.cpp"))
        + list(CSRS_DIR.glob("ops/quantized/cpu/*.cpp"))
    )
    mps_sources = list(CSRS_DIR.glob("ops/mps/*.mm"))

    if IS_ROCM:
        from torch.utils.hipify import hipify_python

        hipify_python.hipify(
            project_directory=str(ROOT_DIR),
            output_directory=str(ROOT_DIR),
            includes="torchscience/csrc/ops/cuda/*",
            show_detailed=True,
            is_pytorch_extension=True,
        )
        cuda_sources = list(CSRS_DIR.glob("ops/hip/*.hip"))
        for header in CSRS_DIR.glob("ops/cuda/*.h"):
            shutil.copy(str(header), str(CSRS_DIR / "ops/hip"))
    else:
        cuda_sources = list(CSRS_DIR.glob("ops/cuda/*.cu"))

    if BUILD_CUDA_SOURCES:
        extension = CUDAExtension
        sources += cuda_sources
    else:
        extension = CppExtension
        if torch.backends.mps.is_available() or FORCE_MPS:
            sources += mps_sources

    define_macros, extra_compile_args = get_macros_and_flags()
    return extension(
        name="torchscience._C",
        sources=sorted(str(s) for s in sources),
        include_dirs=[CSRS_DIR],
        define_macros=define_macros,
        extra_compile_args=extra_compile_args,
    )



def find_library(header):
    # returns (found, include dir, library dir)
    # if include dir or library dir is None, it means that the library is in
    # standard paths and don't need to be added to compiler / linker search
    # paths

    searching_for = f"Searching for {header}"

    for folder in TORCHSCIENCE_INCLUDE:
        if (Path(folder) / header).exists():
            print(f"{searching_for} in {Path(folder) / header}. Found in TORCHSCIENCE_INCLUDE.")
            return True, None, None
    print(f"{searching_for}. Didn't find in TORCHSCIENCE_INCLUDE.")

    # Try conda-related prefixes. If BUILD_PREFIX is set it means conda-build is
    # being run. If CONDA_PREFIX is set then we're in a conda environment.
    for prefix_env_var in ("BUILD_PREFIX", "CONDA_PREFIX"):
        if (prefix := os.environ.get(prefix_env_var)) is not None:
            prefix = Path(prefix)
            if sys.platform == "win32":
                prefix = prefix / "Library"
            include_dir = prefix / "include"
            library_dir = prefix / "lib"
            if (include_dir / header).exists():
                print(f"{searching_for}. Found in {prefix_env_var}.")
                return True, str(include_dir), str(library_dir)
        print(f"{searching_for}. Didn't find in {prefix_env_var}.")

    if sys.platform == "linux":
        for prefix in (Path("/usr/include"), Path("/usr/local/include")):
            if (prefix / header).exists():
                print(f"{searching_for}. Found in {prefix}.")
                return True, None, None
            print(f"{searching_for}. Didn't find in {prefix}")

    if sys.platform == "darwin":
        include_dir = Path("/opt/homebrew") / "include"
        library_dir = Path("/opt/homebrew") / "lib"
        if (include_dir / header).exists():
            print(f"{searching_for}. Found in {include_dir}.")
            return True, str(include_dir), str(library_dir)

    return False, None, None



class clean(distutils.command.clean.clean):
    def run(self):
        with open(".gitignore") as f:
            ignores = f.read()
            for wildcard in filter(None, ignores.split("\n")):
                for filename in glob.glob(wildcard):
                    try:
                        os.remove(filename)
                    except OSError:
                        shutil.rmtree(filename, ignore_errors=True)

        # It's an old-style class in Python 2.7...
        distutils.command.clean.clean.run(self)


if __name__ == "__main__":
    version, sha = get_version()
    write_version_file(version, sha)

    print(f"Building wheel {package_name}-{version}")

    with open("README.md") as f:
        readme = f.read()

    extensions = [
        make_extension(),
    ]

    setup(
        name=package_name,
        version=version,
        packages=find_packages(exclude=("test",)),
        package_data={package_name: ["*.dll", "*.dylib", "*.so", "prototype/datasets/_builtin/*.categories"]},
        zip_safe=False,
        install_requires=get_requirements(),
        extras_require={
            "gdown": ["gdown>=4.7.3"],
            "scipy": ["scipy"],
        },
        ext_modules=extensions,
        python_requires=">=3.10",
        cmdclass={
            "build_ext": BuildExtension.with_options(no_python_abi_suffix=True),
            "clean": clean,
        },
    )
