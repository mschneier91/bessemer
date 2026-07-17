cmake_minimum_required(VERSION 3.24)

project(incns LANGUAGES CXX)

# --- Global build settings (CLAUDE.md: RelWithDebInfo default, C++17, -Werror) ---
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE RelWithDebInfo CACHE STRING "" FORCE)
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Warnings are errors. MFEM's own headers are consumed as SYSTEM includes
# (see cmake/FindMFEM.cmake) so this only polices *our* code.
add_compile_options(-Wall -Wextra -Werror)

# CUDA is optional and off by default (CLAUDE.md). The flag exists in Sprint 1
# only as a build switch; no CUDA sources yet.
option(USE_CUDA "Enable the optional CUDA build" OFF)
if(USE_CUDA)
  message(STATUS "incns: USE_CUDA=ON requested")
  enable_language(CUDA)
endif()

# INCNS_PROFILE scoped timing in libincns' own hot paths (lightweight; on by
# default). Consumers opt in per target via the INCNS_ENABLE_PROFILING define.
option(INCNS_PROFILING "Enable INCNS_PROFILE scoped timing in libincns" ON)

# Python bindings (off by default; the numerics build never depends on Python).
option(INCNS_PYTHON "Build the incns pybind11 Python module" OFF)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake")

# Dependencies come exclusively from the activated spack env (see scripts/env.sh).
find_package(MFEM REQUIRED)   # provides the `incns_mfem` interface target

enable_testing()

add_subdirectory(src)    # libincns
add_subdirectory(apps)
add_subdirectory(bench)
add_subdirectory(test)
if(INCNS_PYTHON)
  add_subdirectory(python)
endif()
