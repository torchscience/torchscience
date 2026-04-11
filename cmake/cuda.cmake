# cmake/cuda.cmake
# CUDA backend configuration - only included when TORCHSCIENCE_ENABLE_CUDA=ON

# Set CUDA architectures before enable_language(CUDA).
# "native" compiles for the GPU on the build machine.
# Fall back to common architectures for cross-compilation.
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES "native")
endif()

enable_language(CUDA)

# Match CUDA C++ standard to the host standard
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Find CUDA toolkit
find_package(CUDAToolkit 12.0 REQUIRED)
message(STATUS "Found CUDA Toolkit: ${CUDAToolkit_VERSION}")
message(STATUS "CUDA architectures: ${CMAKE_CUDA_ARCHITECTURES}")

# CUDA sources
target_sources(_csrc PRIVATE
  src/torchscience/csrc/cuda/special_functions.cu
)

# CUDA-specific includes
target_include_directories(_csrc PRIVATE ${CUDAToolkit_INCLUDE_DIRS})

# CUDA-specific linking
target_link_libraries(_csrc PRIVATE CUDA::cudart)

# Compile definition to enable CUDA code paths
target_compile_definitions(_csrc PRIVATE TORCHSCIENCE_CUDA)

message(STATUS "CUDA backend enabled with ${CUDAToolkit_VERSION}")
