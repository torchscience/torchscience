cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(torchscience_tests CXX)

# Auto-detect PyTorch if CMAKE_PREFIX_PATH not set
if(NOT DEFINED CMAKE_PREFIX_PATH OR CMAKE_PREFIX_PATH STREQUAL "")
    message(STATUS "CMAKE_PREFIX_PATH not set, attempting to auto-detect PyTorch...")

    # Try multiple Python commands in order of preference
    set(PYTHON_COMMANDS "uv run python" "python3" "python")

    foreach(PYTHON_CMD ${PYTHON_COMMANDS})
        # Split command into list for execute_process
        string(REPLACE " " ";" PYTHON_CMD_LIST ${PYTHON_CMD})

        execute_process(
            COMMAND ${PYTHON_CMD_LIST} -c "import torch; print(torch.utils.cmake_prefix_path, end='')"
            OUTPUT_VARIABLE TORCH_CMAKE_PATH
            ERROR_QUIET
            RESULT_VARIABLE PYTHON_RESULT
        )

        if(PYTHON_RESULT EQUAL 0 AND TORCH_CMAKE_PATH)
            message(STATUS "Found PyTorch using '${PYTHON_CMD}': ${TORCH_CMAKE_PATH}")
            set(CMAKE_PREFIX_PATH ${TORCH_CMAKE_PATH})
            break()
        endif()
    endforeach()

    if(NOT CMAKE_PREFIX_PATH)
        message(WARNING "Could not auto-detect PyTorch with any Python command.")
        message(WARNING "Tried: ${PYTHON_COMMANDS}")
        message(WARNING "Please set CMAKE_PREFIX_PATH manually:")
        message(WARNING "  cmake . -DCMAKE_PREFIX_PATH=$(python3 -c 'import torch; print(torch.utils.cmake_prefix_path)')")
        message(WARNING "Or if using uv:")
        message(WARNING "  cmake . -DCMAKE_PREFIX_PATH=$(uv run python -c 'import torch; print(torch.utils.cmake_prefix_path)' | head -1)")
    endif()
endif()

# Find required packages
find_package(Torch REQUIRED)
find_package(GTest REQUIRED)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Include source headers
include_directories(
    ${CMAKE_SOURCE_DIR}/src/torchscience/csrc
    ${TORCH_INCLUDE_DIRS}
)

# Source files for tests - mirrors src/torchscience/csrc structure
set(TEST_SOURCES
        tests/torchscience/csrc/ops/cpu/test_example_kernel.cpp
    tests/torchscience/csrc/ops/meta/test_example_kernel.cpp
    tests/torchscience/csrc/ops/autograd/test_example_kernel.cpp
    tests/torchscience/csrc/ops/sparse/cpu/test_example_kernel.cpp
    tests/torchscience/csrc/ops/quantized/cpu/test_example_kernel.cpp
)

# CUDA support
if(WITH_CUDA OR CUDA_FOUND OR TORCH_CUDA_AVAILABLE)
    enable_language(CUDA)
    list(APPEND TEST_SOURCES
        tests/torchscience/csrc/ops/cuda/test_example_kernel.cu
        tests/torchscience/csrc/ops/autocast/test_example_kernel.cpp
        tests/torchscience/csrc/ops/sparse/cuda/test_example_kernel.cu
    )
    add_definitions(-DWITH_CUDA)
endif()

# HIP/ROCm support
if(WITH_HIP)
    list(APPEND TEST_SOURCES tests/torchscience/csrc/ops/hip/test_example_kernel.cpp)
    add_definitions(-DWITH_HIP)
endif()

# MPS support (macOS/Apple Silicon)
if(APPLE)
    list(APPEND TEST_SOURCES tests/torchscience/csrc/ops/mps/test_example_kernel.mm)
endif()

# Add source files for the kernels being tested
list(APPEND TEST_SOURCES
    src/torchscience/csrc/ops/cpu/example_kernel.cpp
    src/torchscience/csrc/ops/meta/example_kernel.cpp
)

if(WITH_CUDA OR CUDA_FOUND OR TORCH_CUDA_AVAILABLE)
    list(APPEND TEST_SOURCES
        src/torchscience/csrc/ops/cuda/example_kernel.cu
    )
endif()

# Test executable
add_executable(torchscience_tests ${TEST_SOURCES})

# Link libraries
target_link_libraries(torchscience_tests
    GTest::gtest
    GTest::gtest_main
    ${TORCH_LIBRARIES}
)

# Set RPATH for shared libraries
set_target_properties(torchscience_tests PROPERTIES
    INSTALL_RPATH_USE_LINK_PATH TRUE
)

# Enable testing
enable_testing()
add_test(NAME torchscience_cpp_tests COMMAND torchscience_tests)

# Display configuration info
message(STATUS "Torch libraries: ${TORCH_LIBRARIES}")
message(STATUS "Torch include dirs: ${TORCH_INCLUDE_DIRS}")
