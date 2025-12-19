function(add_benchmark NAME SOURCE)
  add_executable(${NAME} ${SOURCE})

  target_include_directories(
    ${NAME}
    PRIVATE
    ${CMAKE_SOURCE_DIR}/src/torchscience/csrc
    ${TORCH_INCLUDE_DIRS}
  )

  target_link_libraries(
    ${NAME}
    PRIVATE
    benchmark::benchmark
    ${TORCH_LIBRARIES}
  )

  set_target_properties(
    ${NAME}
    PROPERTIES
    CXX_STANDARD 17
  )
endfunction()

add_benchmark(
  cos_pi_benchmark
  torchscience/csrc/impl/special_functions/cos_pi_benchmark.cpp
)
