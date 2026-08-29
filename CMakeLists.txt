cmake_minimum_required(VERSION 3.18)

# -----------------------------------------------------------------------------
# cutlass-puzzle: learn CUTLASS and CuTe by filling in TODOs.
# -----------------------------------------------------------------------------

project(cutlass_puzzle LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# -----------------------------------------------------------------------------
# CUDA architectures
# -----------------------------------------------------------------------------
#
# Set CUTLASS_PUZZLE_CUDA_ARCHS to the archs you want to target, e.g.:
#   cmake -B build -DCUTLASS_PUZZLE_CUDA_ARCHS="80;90a;100a"
#
# Defaults to 80 (Ampere). Days requiring SM90 or SM100 will still compile
# with appropriate archs present; they will report "waived" at runtime if
# the running GPU is too old.
if (NOT DEFINED CUTLASS_PUZZLE_CUDA_ARCHS)
  if (DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CUTLASS_PUZZLE_CUDA_ARCHS "${CMAKE_CUDA_ARCHITECTURES}")
  else()
    set(CUTLASS_PUZZLE_CUDA_ARCHS "80")
  endif()
endif()
message(STATUS "cutlass-puzzle: targeting CUDA archs = ${CUTLASS_PUZZLE_CUDA_ARCHS}")

# -----------------------------------------------------------------------------
# CUTLASS discovery
# -----------------------------------------------------------------------------
#
# Priority:
#   1. -DCUTLASS_DIR=<path>
#   2. Environment variable CUTLASS_DIR
#   3. Git submodule at third_party/cutlass
#
if (NOT DEFINED CUTLASS_DIR)
  if (DEFINED ENV{CUTLASS_DIR})
    set(CUTLASS_DIR "$ENV{CUTLASS_DIR}")
  elseif (EXISTS "${CMAKE_SOURCE_DIR}/third_party/cutlass/include/cutlass/cutlass.h")
    set(CUTLASS_DIR "${CMAKE_SOURCE_DIR}/third_party/cutlass")
  endif()
endif()

if (NOT CUTLASS_DIR OR NOT EXISTS "${CUTLASS_DIR}/include/cutlass/cutlass.h")
  message(FATAL_ERROR
    "CUTLASS headers not found.\n"
    "  Option A: add the submodule — git submodule add https://github.com/NVIDIA/cutlass third_party/cutlass\n"
    "  Option B: point to an existing clone — cmake -B build -DCUTLASS_DIR=/path/to/cutlass\n"
    "  Option C: export CUTLASS_DIR=/path/to/cutlass before invoking cmake.\n"
    "  Currently resolved CUTLASS_DIR='${CUTLASS_DIR}'")
endif()
message(STATUS "cutlass-puzzle: using CUTLASS at ${CUTLASS_DIR}")

# -----------------------------------------------------------------------------
# Common interface target
# -----------------------------------------------------------------------------
#
# All blog and puzzle executables link this. It captures the CUTLASS include
# paths and the nvcc flags required to make CuTe-heavy code compile cleanly.

add_library(cutlass_puzzle_common INTERFACE)
target_include_directories(cutlass_puzzle_common INTERFACE
  ${CUTLASS_DIR}/include
  ${CUTLASS_DIR}/tools/util/include)

target_compile_options(cutlass_puzzle_common INTERFACE
  $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>
  $<$<COMPILE_LANGUAGE:CUDA>:--expt-extended-lambda>
  $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-Wno-unknown-pragmas>)

# CUTLASS/CuTe uses heavy templates; these reduce register spills.
if (CMAKE_BUILD_TYPE MATCHES "Release")
  target_compile_options(cutlass_puzzle_common INTERFACE
    $<$<COMPILE_LANGUAGE:CUDA>:-O3>)
endif()

# -----------------------------------------------------------------------------
# Subdirectories
# -----------------------------------------------------------------------------

add_subdirectory(blog)

message(STATUS "cutlass-puzzle: configured. Build with cmake --build build --target blog_all.")
