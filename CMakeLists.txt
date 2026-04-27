cmake_minimum_required(VERSION 3.18)

if(NOT DEFINED SKBUILD_PROJECT_NAME)
  set(SKBUILD_PROJECT_NAME "torchscience")
endif()

project(${SKBUILD_PROJECT_NAME} LANGUAGES CXX)

# Build options
# Core dependencies
find_package(Python REQUIRED COMPONENTS Interpreter Development.Module)

# Get Torch installation path from the Python environment to avoid finding system Torch
execute_process(
  COMMAND ${Python_EXECUTABLE} -c "import torch; import os; print(os.path.dirname(torch.__file__))"
  OUTPUT_VARIABLE TORCH_INSTALL_PREFIX
  OUTPUT_STRIP_TRAILING_WHITESPACE
  RESULT_VARIABLE _torch_result
)
if(_torch_result EQUAL 0 AND TORCH_INSTALL_PREFIX)
  # Set TORCH_INSTALL_PREFIX so TorchConfig.cmake uses the correct torch
  set(ENV{TORCH_INSTALL_PREFIX} ${TORCH_INSTALL_PREFIX})
  # Prepend to CMAKE_LIBRARY_PATH so find_library finds venv's torch libs first
  list(PREPEND CMAKE_LIBRARY_PATH "${TORCH_INSTALL_PREFIX}/lib")
  list(PREPEND CMAKE_PREFIX_PATH "${TORCH_INSTALL_PREFIX}/share/cmake")
endif()

find_package(Torch REQUIRED)

# Backend detection (sets TORCHSCIENCE_ENABLE_CUDA, TORCHSCIENCE_ENABLE_MPS)
include(cmake/backend_detection.cmake)

# Base sources
set(TORCHSCIENCE_SOURCES
  src/torchscience/csrc/torchscience.cpp
  src/torchscience/csrc/special_functions.cpp
  src/torchscience/csrc/noise.cpp
)

# Create library target
add_library(_csrc MODULE ${TORCHSCIENCE_SOURCES})

# Include backend configurations
include(cmake/cpu.cmake)

if(TORCHSCIENCE_ENABLE_CUDA)
  include(cmake/cuda.cmake)
endif()

if(TORCHSCIENCE_ENABLE_MPS)
  include(cmake/mps.cmake)
endif()

# Common configuration
# IMPORTANT: Add Torch includes BEFORE other system includes to avoid conflicts
# with system-installed Torch (e.g., from Homebrew)
target_include_directories(_csrc BEFORE PRIVATE ${TORCH_INCLUDE_DIRS})
target_include_directories(_csrc PRIVATE ${Python_INCLUDE_DIRS})
target_link_libraries(_csrc PRIVATE ${TORCH_LIBRARIES} Python::Module)

set_target_properties(_csrc PROPERTIES
  BUILD_WITH_INSTALL_RPATH TRUE
  CXX_STANDARD 17
  INSTALL_RPATH_USE_LINK_PATH FALSE
  PREFIX ""
)

if(APPLE)
  set_target_properties(_csrc PROPERTIES
    INSTALL_RPATH "@loader_path;@loader_path/../torch/lib"
    SUFFIX ".so"
  )
elseif(UNIX)
  set_target_properties(_csrc PROPERTIES
    INSTALL_RPATH "$ORIGIN;$ORIGIN/../torch/lib"
  )
elseif(WIN32)
  set_target_properties(_csrc PROPERTIES
    SUFFIX ".pyd"
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
  )
  # On Windows, DLLs are found via PATH, not rpath. Copy torch DLLs
  # to the build directory so tests can find them.
  add_custom_command(TARGET _csrc POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E echo "Windows build: torch DLLs resolved via PATH"
  )
endif()

# Install
if(SKBUILD)
  install(TARGETS _csrc DESTINATION ${SKBUILD_PROJECT_NAME})
endif()
