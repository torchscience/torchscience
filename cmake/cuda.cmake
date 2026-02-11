# cmake/cuda.cmake
# CUDA backend configuration - only included when TORCHSCIENCE_ENABLE_CUDA=ON

enable_language(CUDA)

# Find CUDA toolkit
find_package(CUDAToolkit 12.0 REQUIRED)
message(STATUS "Found CUDA Toolkit: ${CUDAToolkit_VERSION}")

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
