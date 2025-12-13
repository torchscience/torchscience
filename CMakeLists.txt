cmake_minimum_required(VERSION 3.15)

if(NOT DEFINED SKBUILD_PROJECT_NAME)
    set(SKBUILD_PROJECT_NAME "torchscience")
endif()

# Check for CUDA availability
include(CheckLanguage)
check_language(CUDA)

if(CMAKE_CUDA_COMPILER)
    project(${SKBUILD_PROJECT_NAME} LANGUAGES CXX CUDA)
    set(TORCHSCIENCE_CUDA_AVAILABLE ON)
    message(STATUS "CUDA compiler found: ${CMAKE_CUDA_COMPILER}")
else()
    project(${SKBUILD_PROJECT_NAME} LANGUAGES CXX)
    set(TORCHSCIENCE_CUDA_AVAILABLE OFF)
    message(STATUS "CUDA compiler not found, building CPU-only version")
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(TORCHSCIENCE_CUDA_AVAILABLE)
    set(CMAKE_CUDA_STANDARD 17)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
endif()

find_package(Python REQUIRED COMPONENTS Interpreter Development)

file(
  GLOB
  TORCH_CMAKE_PATH
  "${CMAKE_SOURCE_DIR}/.venv/lib/python*/site-packages/torch/share/cmake"
)

list(APPEND CMAKE_PREFIX_PATH ${TORCH_CMAKE_PATH})

find_package(Torch REQUIRED)

# nlohmann/json is required by PyTorch's functorch headers for vmap support
include(FetchContent)
FetchContent_Declare(
  nlohmann_json
  URL https://github.com/nlohmann/json/releases/download/v3.11.3/json.tar.xz
  DOWNLOAD_EXTRACT_TIMESTAMP TRUE
)

FetchContent_MakeAvailable(nlohmann_json)

FetchContent_Declare(
  Boost
  URL https://github.com/boostorg/boost/releases/download/boost-1.87.0/boost-1.87.0-cmake.tar.xz
  DOWNLOAD_EXTRACT_TIMESTAMP TRUE
)

set(BOOST_INCLUDE_LIBRARIES math)
set(BUILD_SHARED_LIBS OFF)

FetchContent_MakeAvailable(Boost)

file(GLOB_RECURSE CPP_SOURCES "src/torchscience/csrc/*.cpp")

if(TORCHSCIENCE_CUDA_AVAILABLE)
    file(GLOB_RECURSE CUDA_SOURCES "src/torchscience/csrc/*.cu")
    set(SOURCES ${CPP_SOURCES} ${CUDA_SOURCES})
    message(STATUS "Found CUDA sources: ${CUDA_SOURCES}")
else()
    set(SOURCES ${CPP_SOURCES})
endif()

add_library(_csrc MODULE ${SOURCES})

set_target_properties(_csrc PROPERTIES PREFIX "")

if(TORCHSCIENCE_CUDA_AVAILABLE)
    set_target_properties(_csrc PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_RESOLVE_DEVICE_SYMBOLS ON
    )
    target_compile_definitions(_csrc PRIVATE TORCHSCIENCE_CUDA_AVAILABLE)
endif()

target_include_directories(_csrc PRIVATE
  ${CMAKE_SOURCE_DIR}/include
  ${CMAKE_SOURCE_DIR}/src
)

target_link_libraries(_csrc PRIVATE Python::Python ${TORCH_LIBRARIES} nlohmann_json::nlohmann_json Boost::math)

if(TORCHSCIENCE_CUDA_AVAILABLE)
    target_link_libraries(_csrc PRIVATE ${TORCH_CUDA_LIBRARIES})
endif()

if(SKBUILD)
    install(TARGETS _csrc DESTINATION ${SKBUILD_PROJECT_NAME})
endif()

FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/archive/refs/tags/v1.15.2.tar.gz
  DOWNLOAD_EXTRACT_TIMESTAMP TRUE
)

FetchContent_Declare(
  googlebenchmark
  URL https://github.com/google/benchmark/archive/refs/tags/v1.9.1.tar.gz
  DOWNLOAD_EXTRACT_TIMESTAMP TRUE
)

set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)

set(BENCHMARK_ENABLE_TESTING OFF CACHE BOOL "" FORCE)
set(BENCHMARK_ENABLE_GTEST_TESTS OFF CACHE BOOL "" FORCE)

FetchContent_MakeAvailable(googletest googlebenchmark)

enable_testing()

file(GLOB_RECURSE TEST_SOURCES "tests/csrc/**/*.cpp")

add_executable(torchscience_tests ${TEST_SOURCES})

target_include_directories(torchscience_tests PRIVATE
  ${CMAKE_SOURCE_DIR}/include
  ${CMAKE_SOURCE_DIR}/src
)

target_link_libraries(torchscience_tests PRIVATE
  GTest::gtest_main
  ${TORCH_LIBRARIES}
  Boost::math
)

include(GoogleTest)

gtest_discover_tests(torchscience_tests)

file(GLOB_RECURSE BENCHMARK_SOURCES "benchmarks/csrc/**/*.cpp")

if(BENCHMARK_SOURCES)
  add_executable(torchscience_benchmarks ${BENCHMARK_SOURCES})

  target_include_directories(torchscience_benchmarks PRIVATE
    ${CMAKE_SOURCE_DIR}
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/src
  )

  target_link_libraries(torchscience_benchmarks PRIVATE
    benchmark::benchmark_main
    ${TORCH_LIBRARIES}
    Boost::math
  )
endif()
