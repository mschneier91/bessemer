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

# The presets hardcode a ccache compiler launcher, but not every environment
# has ccache: without it, every single compile died with code=127 until the launcher
# was overridden by hand on the command line. Keeping that override alive only
# in the cache is fragile -- anything that wipes the cache (e.g. CMake deciding
# the compiler changed, which happens if the spack env is not active) silently
# restores the broken launcher. Drop it here instead, where it cannot be lost.
if(CMAKE_CXX_COMPILER_LAUNCHER)
  find_program(INCNS_LAUNCHER_EXE NAMES ${CMAKE_CXX_COMPILER_LAUNCHER})
  if(NOT INCNS_LAUNCHER_EXE)
    message(STATUS "incns: compiler launcher '${CMAKE_CXX_COMPILER_LAUNCHER}' "
                   "not found on PATH -- building without it")
    unset(CMAKE_CXX_COMPILER_LAUNCHER CACHE)
    set(CMAKE_CXX_COMPILER_LAUNCHER "")
  endif()
endif()

# Warnings are errors. MFEM's own headers are consumed as SYSTEM includes
# (see cmake/FindMFEM.cmake) so this only polices *our* code.
#
# Scoped to CXX: nvcc rejects a bare -Werror (it wants a kind, e.g.
# "-Werror all-warnings"), so the device TUs below hand the same three flags to
# their host compiler behind -Xcompiler instead.
add_compile_options($<$<COMPILE_LANGUAGE:CXX>:-Wall>
                    $<$<COMPILE_LANGUAGE:CXX>:-Wextra>
                    $<$<COMPILE_LANGUAGE:CXX>:-Werror>)

# CUDA is optional and off by default. ON compiles the TUs that
# define their own device kernels with nvcc -- see src/CMakeLists.txt for the
# list, which is deliberately as short as possible.
option(USE_CUDA "Enable the optional CUDA build" OFF)
if(USE_CUDA)
  message(STATUS "incns: USE_CUDA=ON requested")

  # The compute capability must be the one the environment's MFEM was built
  # for (-arch=sm_NN in its config.mk, cuda_arch=NN in the environment's
  # spack.yaml); a mismatch produces kernels the card will not run.
  # scripts/env.sh exports it as INCNS_CUDA_ARCH; -DINCNS_CUDA_ARCH=NN overrides.
  if(NOT INCNS_CUDA_ARCH AND DEFINED ENV{INCNS_CUDA_ARCH})
    set(INCNS_CUDA_ARCH "$ENV{INCNS_CUDA_ARCH}")
  endif()
  if(NOT INCNS_CUDA_ARCH)
    message(FATAL_ERROR "incns: USE_CUDA=ON needs the CUDA compute capability. "
                        "Build through scripts/build.sh (it takes cuda_arch= from the "
                        "environment's spack.yaml) or pass -DINCNS_CUDA_ARCH=<NN>, "
                        "e.g. 90 for an H100.")
  endif()
  set(INCNS_CUDA_ARCH "${INCNS_CUDA_ARCH}" CACHE STRING
      "CUDA compute capability for device kernels (e.g. 90 = H100)")
  # FORCE because enable_language(CUDA) seeds the cache with CMake's own guess
  # (52) on first configure, and that stale value would otherwise keep winning
  # on every later configure of an existing build dir.
  set(CMAKE_CUDA_ARCHITECTURES "${INCNS_CUDA_ARCH}" CACHE STRING "" FORCE)

  set(CMAKE_CUDA_STANDARD 17)
  set(CMAKE_CUDA_STANDARD_REQUIRED ON)
  set(CMAKE_CUDA_EXTENSIONS OFF)

  enable_language(CUDA)

  # --expt-extended-lambda is MANDATORY: every kernel here is an
  # [=] MFEM_HOST_DEVICE lambda passed to mfem::forall. --expt-relaxed-constexpr
  # mirrors how MFEM itself was compiled (both appear in its config.mk).
  add_compile_options($<$<COMPILE_LANGUAGE:CUDA>:--expt-extended-lambda>
                      $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>)
  add_compile_options($<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-Wall,-Wextra,-Werror>)

  # This project has no find_package(MPI): the MPI include paths arrive
  # implicitly from CMAKE_CXX_COMPILER=mpicxx. nvcc inherits none of that, and
  # these TUs include mfem.hpp -> mpi.h, so aim its host compiler at that same
  # wrapper. Doing it this way (a flag) rather than via CMAKE_CUDA_HOST_COMPILER
  # is deliberate: changing that variable on an existing cache is a hard CMake
  # error demanding the cache be deleted, and deleting build/cuda would force a
  # --preset reconfigure, which re-stamps the ccache launcher this env lacks.
  # It also keeps the device TUs on the spack toolchain rather than whatever
  # bare g++ nvcc defaults to (AGENTS.md rule 1: no system-toolchain leak). MFEM
  # resolves it identically (-ccbin in its config.mk).
  add_compile_options($<$<COMPILE_LANGUAGE:CUDA>:-ccbin=${CMAKE_CXX_COMPILER}>)

  # The link is driven by mpicxx, not nvcc, so nothing supplies the CUDA runtime
  # implicitly -- and an nvcc-compiled TU needs libcudart for the fatbin
  # registration nvcc emits into it. MFEM's config.mk also asks for
  # -lcusparse/-lcublas/-lcurand/-lcusolver with no -L at all: Bridges-2's CUDA
  # module resolved them through LIBRARY_PATH, Delta's leaves it empty, so the
  # toolkit's library directory goes on every link line.
  find_package(CUDAToolkit REQUIRED)
  link_directories(${CUDAToolkit_LIBRARY_DIR})

  # Cray MPICH moves device buffers only through its GPU Transport Layer
  # (libmpi_gtl_cuda in <mpich>/gtl/lib), which Cray's own compiler wrappers
  # link and its plain mpicxx does not. With Cray MPICH as the compiler wrapper
  # (<prefix>/pe/mpich/<version>/ofi/<vendor>/<version>/bin/mpicxx), link it
  # into every target, kept even though nothing calls it directly, and tell
  # ConfigureMpiTransport (INCNS_MPI_GTL) to switch MPICH's GPU support on.
  get_filename_component(_incns_mpicxx "${CMAKE_CXX_COMPILER}" REALPATH)
  if(_incns_mpicxx MATCHES "^(.*/pe/mpich/[^/]+)/ofi/")
    find_library(INCNS_MPI_GTL_CUDA NAMES mpi_gtl_cuda
                 PATHS "${CMAKE_MATCH_1}/gtl/lib" NO_DEFAULT_PATH REQUIRED)
    message(STATUS "incns: Cray MPICH, GPU transport layer ${INCNS_MPI_GTL_CUDA}")
    add_compile_definitions(INCNS_MPI_GTL)
    link_libraries("-Wl,--push-state,--no-as-needed" "${INCNS_MPI_GTL_CUDA}"
                   "-Wl,--pop-state")
  endif()
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

# The MPI launcher the tests run under: $INCNS_MPIEXEC from scripts/env.sh,
# which is mpirun unless the machine's environment names another in its
# `mpiexec` file (Cray MPICH under Slurm launches only through srun). Every
# launcher here takes `-n <ranks>`.
if(DEFINED ENV{INCNS_MPIEXEC} AND NOT "$ENV{INCNS_MPIEXEC}" STREQUAL "")
  separate_arguments(INCNS_MPIEXEC UNIX_COMMAND "$ENV{INCNS_MPIEXEC}")
else()
  set(INCNS_MPIEXEC mpirun)
endif()
list(GET INCNS_MPIEXEC 0 _incns_launcher)
find_program(INCNS_MPIEXEC_EXECUTABLE NAMES ${_incns_launcher} REQUIRED)
list(REMOVE_AT INCNS_MPIEXEC 0)
list(PREPEND INCNS_MPIEXEC ${INCNS_MPIEXEC_EXECUTABLE})

add_subdirectory(src)    # libincns
add_subdirectory(apps)
add_subdirectory(bench)
add_subdirectory(test)
if(INCNS_PYTHON)
  add_subdirectory(python)
endif()
