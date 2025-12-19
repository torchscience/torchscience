cmake_minimum_required(
  VERSION 3.15
)

project(
  ${SKBUILD_PROJECT_NAME}
  LANGUAGES CXX
)

option(TORCHSCIENCE_BUILD_BENCHMARKS "Build benchmarks" OFF)

if(TORCHSCIENCE_BUILD_BENCHMARKS AND NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

find_package(
  Python
  REQUIRED COMPONENTS
  Interpreter
  Development.Module
)

find_package(
  Torch
  REQUIRED
)

set(TORCHSCIENCE_SOURCES
  src/torchscience/csrc/torchscience.cpp
)

if(TORCH_CUDA_LIBRARIES)
  enable_language(CUDA)
  list(APPEND TORCHSCIENCE_SOURCES
    src/torchscience/csrc/cuda/special_functions/chebyshev_polynomial_t.cu
    src/torchscience/csrc/cuda/special_functions/gamma.cu
  )
  add_definitions(-DTORCHSCIENCE_CUDA)
endif()

add_library(
  _csrc
  MODULE
  ${TORCHSCIENCE_SOURCES}
)

target_include_directories(
  _csrc
  PRIVATE
  ${Python_INCLUDE_DIRS}
)

target_link_libraries(
  _csrc
  PRIVATE
  ${TORCH_LIBRARIES}
  Python::Module
)

set_target_properties(
  _csrc
  PROPERTIES
  BUILD_WITH_INSTALL_RPATH TRUE
  CXX_STANDARD 17
  INSTALL_RPATH_USE_LINK_PATH TRUE
  PREFIX ""
)

if(APPLE)
  set_target_properties(
    _csrc
    PROPERTIES
    INSTALL_RPATH "@loader_path;@loader_path/../torch/lib"
    SUFFIX ".so"
  )
endif()

install(
  TARGETS _csrc
  DESTINATION ${SKBUILD_PROJECT_NAME}
)

if(TORCHSCIENCE_BUILD_BENCHMARKS)
  include(FetchContent)

  FetchContent_Declare(
    benchmark
    GIT_REPOSITORY https://github.com/google/benchmark.git
    GIT_TAG v1.9.1
  )

  set(BENCHMARK_ENABLE_TESTING OFF CACHE BOOL "" FORCE)
  set(BENCHMARK_ENABLE_INSTALL OFF CACHE BOOL "" FORCE)
  set(BENCHMARK_ENABLE_GTEST_TESTS OFF CACHE BOOL "" FORCE)

  FetchContent_MakeAvailable(benchmark)

  add_subdirectory(benchmark)
endif()
