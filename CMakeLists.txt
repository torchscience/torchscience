cmake_minimum_required(
  VERSION 3.15
)

project(
  ${SKBUILD_PROJECT_NAME}
  LANGUAGES CXX
)

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

add_library(
  _csrc
  MODULE
  src/torchscience/csrc/torchscience.cpp
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
