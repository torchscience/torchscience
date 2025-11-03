cmake_minimum_required(VERSION 3.18)
project(torchscience)
set(CMAKE_CXX_STANDARD 17)
file(STRINGS VERSION TORCHSCIENCE_VERSION)

option(WITH_CUDA "Enable CUDA support" OFF)
option(WITH_MPS "Enable MPS support" OFF)

if(WITH_CUDA)
  enable_language(CUDA)
  add_definitions(-D__CUDA_NO_HALF_OPERATORS__)
  add_definitions(-DWITH_CUDA)
  set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-relaxed-constexpr")
endif()

if(WITH_MPS)
  enable_language(OBJC OBJCXX)
  add_definitions(-DWITH_MPS)
endif()

find_package(Torch REQUIRED)

function(CUDA_CONVERT_FLAGS EXISTING_TARGET)
    get_property(old_flags TARGET ${EXISTING_TARGET} PROPERTY INTERFACE_COMPILE_OPTIONS)
    if(NOT "${old_flags}" STREQUAL "")
        string(REPLACE ";" "," CUDA_flags "${old_flags}")
        set_property(TARGET ${EXISTING_TARGET} PROPERTY INTERFACE_COMPILE_OPTIONS
            "$<$<BUILD_INTERFACE:$<COMPILE_LANGUAGE:CXX>>:${old_flags}>$<$<BUILD_INTERFACE:$<COMPILE_LANGUAGE:CUDA>>:-Xcompiler=${CUDA_flags}>"
            )
    endif()
endfunction()

if(MSVC)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /wd4819")
  if(WITH_CUDA)
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler=/wd4819")
    foreach(diag cc_clobber_ignored integer_sign_change useless_using_declaration
      set_but_not_used field_without_dll_interface
      base_class_has_different_dll_interface
      dll_interface_conflict_none_assumed
      dll_interface_conflict_dllexport_assumed
      implicit_return_from_non_void_function
      unsigned_compare_with_zero
      declared_but_not_referenced
      bad_friend_decl)
      string(APPEND CMAKE_CUDA_FLAGS " -Xcudafe --diag_suppress=${diag}")
    endforeach()
    CUDA_CONVERT_FLAGS(torch_cpu)
    if(TARGET torch_cuda)
      CUDA_CONVERT_FLAGS(torch_cuda)
    endif()
    if(TARGET torch_cuda_cu)
      CUDA_CONVERT_FLAGS(torch_cuda_cu)
    endif()
    if(TARGET torch_cuda_cpp)
      CUDA_CONVERT_FLAGS(torch_cuda_cpp)
    endif()
  endif()
endif()

include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

set(TSCPP torchscience/csrc)

list(
    APPEND
    ALLOW_LISTED
    ${TSCPP}
    ${TSCPP}/ops
    ${TSCPP}/ops/autograd
    ${TSCPP}/ops/cpu
)

if(WITH_CUDA)
    list(APPEND ALLOW_LISTED ${TSCPP}/ops/autocast ${TSCPP}/ops/cuda)
endif()

if(WITH_MPS)
    list(APPEND ALLOW_LISTED ${TSCPP}/ops/mps)
endif()

FOREACH(DIR ${ALLOW_LISTED})
    file(GLOB ALL_SOURCES ${ALL_SOURCES} ${DIR}/*.*)
ENDFOREACH()

add_library(${PROJECT_NAME} SHARED ${ALL_SOURCES})

target_link_libraries(${PROJECT_NAME} PRIVATE ${TORCH_LIBRARIES})

if(WITH_MPS)
    find_library(metal NAMES Metal)

    find_library(foundation NAMES Foundation)

    target_link_libraries(${PROJECT_NAME} PRIVATE ${metal} ${foundation})
endif()

set_target_properties(
    ${PROJECT_NAME} PROPERTIES
    EXPORT_NAME TorchScience
    INSTALL_RPATH ${TORCH_INSTALL_PREFIX}/lib
)

include_directories(torchscience/csrc)

set(
    TORCHSCIENCE_CMAKECONFIG_INSTALL_DIR "share/cmake/TorchScience"
    CACHE STRING "install path for TorchScienceConfig.cmake"
)

configure_package_config_file(
    cmake/TorchScienceConfig.cmake.in
    "${CMAKE_CURRENT_BINARY_DIR}/TorchScienceConfig.cmake"
    INSTALL_DESTINATION ${TORCHSCIENCE_CMAKECONFIG_INSTALL_DIR}
)

write_basic_package_version_file(
    ${CMAKE_CURRENT_BINARY_DIR}/TorchScienceConfigVersion.cmake
    VERSION ${TORCHSCIENCE_VERSION}
    COMPATIBILITY AnyNewerVersion
)

install(
    FILES
    ${CMAKE_CURRENT_BINARY_DIR}/TorchScienceConfig.cmake
    ${CMAKE_CURRENT_BINARY_DIR}/TorchScienceConfigVersion.cmake
    DESTINATION ${TORCHSCIENCE_CMAKECONFIG_INSTALL_DIR}
)

install(
    TARGETS ${PROJECT_NAME}
    EXPORT TorchVisionTargets
    LIBRARY
    DESTINATION ${CMAKE_INSTALL_LIBDIR}
)

install(
    EXPORT TorchVisionTargets
    NAMESPACE TorchScience::
    DESTINATION ${TORCHSCIENCE_CMAKECONFIG_INSTALL_DIR}
)

FOREACH(INPUT_DIR ${ALLOW_LISTED})
    string(REPLACE "${TSCPP}" "${CMAKE_INSTALL_INCLUDEDIR}/${PROJECT_NAME}" OUTPUT_DIR ${INPUT_DIR})
    file(GLOB INPUT_FILES ${INPUT_DIR}/*.*)
    install(FILES ${INPUT_FILES} DESTINATION ${OUTPUT_DIR})
ENDFOREACH()
