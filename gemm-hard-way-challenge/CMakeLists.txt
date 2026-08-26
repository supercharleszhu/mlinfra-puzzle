cmake_minimum_required(VERSION 3.18)

project(gemm_hard_way_navigation LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

if(NOT DEFINED CUTLASS_DIR)
  set(CUTLASS_DIR "${CMAKE_CURRENT_SOURCE_DIR}/.cache/cutlass-v4.6.1")
endif()

if(NOT EXISTS "${CUTLASS_DIR}/include/cute/tensor.hpp")
  message(FATAL_ERROR
    "CUTLASS v4.6.1 was not found at ${CUTLASS_DIR}. "
    "Run scripts/setup_advanced_cutlass.sh first or pass -DCUTLASS_DIR=<path>.")
endif()

function(add_cute_navigation_target day source_name architecture)
  set(target "cute_navigation_day_${day}")
  add_library(${target} OBJECT
    "cuda/${source_name}"
    "solutions/cuda/${source_name}")

  # These targets exist to populate compile_commands.json. Each day receives
  # the same architecture and include flags as the direct NVCC runner.
  set_property(TARGET ${target} PROPERTY EXCLUDE_FROM_ALL TRUE)
  set_property(TARGET ${target} PROPERTY CUDA_ARCHITECTURES OFF)
  target_compile_options(${target} PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>
    $<$<COMPILE_LANGUAGE:CUDA>:-arch=sm_${architecture}>)
  target_include_directories(${target} PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/cuda"
    "${CUTLASS_DIR}/include"
    "${CUTLASS_DIR}/tools/util/include"
    "${CUTLASS_DIR}/examples/common"
    "${CUTLASS_DIR}/examples/cute/tutorial"
    "${CUTLASS_DIR}/examples/cute/tutorial/hopper"
    "${CUTLASS_DIR}/examples/cute/tutorial/blackwell")
endfunction()

add_cute_navigation_target(20 20_cute_sgemm_1.cu 80)
add_cute_navigation_target(21 21_cute_sgemm_2.cu 80)
add_cute_navigation_target(22 22_cute_sgemm_sm80.cu 90)
add_cute_navigation_target(23 23_cute_hopper_wgmma.cu 90a)
add_cute_navigation_target(24 24_cute_hopper_wgmma_tma.cu 90a)
add_cute_navigation_target(25 25_cute_blackwell_mma.cu 100a)
add_cute_navigation_target(26 26_cute_blackwell_mma_tma.cu 100a)
add_cute_navigation_target(27 27_cute_blackwell_mma_tma_2sm.cu 100a)
add_cute_navigation_target(28 28_cute_blackwell_mma_tma_epilogue.cu 100a)
