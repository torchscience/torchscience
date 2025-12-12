cmake_minimum_required(VERSION 3.15)

if(NOT DEFINED SKBUILD_PROJECT_NAME)
    set(SKBUILD_PROJECT_NAME "torchscience")
endif()

project(${SKBUILD_PROJECT_NAME} LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(Python REQUIRED COMPONENTS Interpreter Development)

file(GLOB TORCH_CMAKE_PATH "${CMAKE_SOURCE_DIR}/.venv/lib/python*/site-packages/torch/share/cmake")
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

file(GLOB_RECURSE SOURCES "src/torchscience/csrc/*.cpp")

add_library(_csrc MODULE ${SOURCES})

set_target_properties(_csrc PROPERTIES PREFIX "")

target_include_directories(_csrc PRIVATE
  ${CMAKE_SOURCE_DIR}/include
  ${CMAKE_SOURCE_DIR}/src
)

target_link_libraries(_csrc PRIVATE Python::Python ${TORCH_LIBRARIES} nlohmann_json::nlohmann_json)

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
  )
endif()
