# FindMFEM.cmake -- locate the spack-installed MFEM and expose it as a target.
#
# This spack build of MFEM installs a Makefile config.mk but no MFEMConfig.cmake,
# so we read config.mk (via cmake/mfem_query.mk) to obtain the fully-expanded
# include and link flags. Result: an interface target `incns_mfem` that any
# executable/library can link against.
#
# MFEM's own headers are attached as SYSTEM includes so that the project-wide
# -Werror does not fire on warnings originating inside MFEM.

if(TARGET incns_mfem)
  return()
endif()

# 1. Locate the MFEM install prefix. scripts/env.sh exports MFEM_DIR after
#    activating the spack env; fall back to querying spack directly.
if(DEFINED ENV{MFEM_DIR})
  set(_mfem_dir "$ENV{MFEM_DIR}")
else()
  find_program(_spack_exe spack)
  if(NOT _spack_exe)
    message(FATAL_ERROR
      "FindMFEM: MFEM_DIR is unset and spack is not on PATH. "
      "Activate the env first (scripts/env.sh) or build via scripts/build.sh.")
  endif()
  execute_process(
    COMMAND ${_spack_exe} location -i mfem
    OUTPUT_VARIABLE _mfem_dir
    OUTPUT_STRIP_TRAILING_WHITESPACE
    RESULT_VARIABLE _rc)
  if(NOT _rc EQUAL 0)
    message(FATAL_ERROR "FindMFEM: `spack location -i mfem` failed.")
  endif()
endif()

set(_config_mk "${_mfem_dir}/share/mfem/config.mk")
if(NOT EXISTS "${_config_mk}")
  message(FATAL_ERROR "FindMFEM: config.mk not found at ${_config_mk}")
endif()

# 2. Query make for the expanded flags.
find_program(_make_exe NAMES make gmake REQUIRED)
set(_qmk "${CMAKE_CURRENT_LIST_DIR}/mfem_query.mk")

function(_mfem_query out_var target)
  execute_process(
    COMMAND ${_make_exe} -s -f "${_qmk}" "MFEM_CONFIG_MK=${_config_mk}" "${target}"
    OUTPUT_VARIABLE _out
    OUTPUT_STRIP_TRAILING_WHITESPACE
    RESULT_VARIABLE _r)
  if(NOT _r EQUAL 0)
    message(FATAL_ERROR "FindMFEM: query '${target}' failed for ${_config_mk}")
  endif()
  set(${out_var} "${_out}" PARENT_SCOPE)
endfunction()

_mfem_query(_incflags print-incflags)
_mfem_query(_libflags  print-libs)

# 3. Split include flags into directories (strip the -I prefix).
separate_arguments(_inc_tokens UNIX_COMMAND "${_incflags}")
set(_inc_dirs "")
foreach(_tok IN LISTS _inc_tokens)
  string(REGEX REPLACE "^-I" "" _dir "${_tok}")
  if(_dir AND IS_DIRECTORY "${_dir}")
    list(APPEND _inc_dirs "${_dir}")
  endif()
endforeach()

# 4. Link flags (rpath / -L / -l ...) pass through verbatim, order preserved.
separate_arguments(_lib_tokens UNIX_COMMAND "${_libflags}")

add_library(incns_mfem INTERFACE)
target_include_directories(incns_mfem SYSTEM INTERFACE ${_inc_dirs})
target_link_libraries(incns_mfem INTERFACE ${_lib_tokens})

set(MFEM_DIR "${_mfem_dir}" CACHE PATH "MFEM install prefix" FORCE)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(MFEM REQUIRED_VARS MFEM_DIR _inc_dirs _lib_tokens)

message(STATUS "FindMFEM: using ${_mfem_dir}")
