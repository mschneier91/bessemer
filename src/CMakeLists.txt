# libincns -- the solver library. Grows one module per sub-sprint; a driver or
# test never edits it to run a case. Sprint 1.1 seeds it with mesh + spaces.
add_library(incns STATIC
  mesh/periodic_box.cpp
  spaces/mixed_spaces.cpp
  quadrature/rule_book.cpp
  util/profiler.cpp
  util/device.cpp
  post/pressure_mean.cpp
  bc/boundary_conditions.cpp
  operators/stokes_operator.cpp
  operators/convection.cpp
  operators/grad_div_integrator.cpp
  operators/rotational_convection.cpp
  operators/pressure_schur.cpp
  operators/block_preconditioner.cpp
  solver/stokes_solver.cpp
  time/multistep_coeffs.cpp
  time/time_integrator.cpp
  time/adaptive_controller.cpp
  config/parameters.cpp
  config/nondimensionalization.cpp
  config/initial_conditions.cpp
  post/output.cpp
  post/checkpoint.cpp
  post/diagnostics.cpp
  precond/cahouet_chabard.cpp
  precond/point_block_jacobi.cpp
  solver/case.cpp)

# Public: headers are included as "mesh/...", "spaces/..." from this root, and
# every consumer needs MFEM.
target_include_directories(incns PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(incns PUBLIC incns_mfem)

# --- The nvcc TUs -------------------------------------------------------------
# Keep this list minimal: a file belongs here only if it *defines* device
# kernels that cannot be expressed as device-aware mfem::Vector operations.
# grad_div_integrator.cpp -- sum-factorized grad-div built on forall_2D with
# MFEM_SHARED tiles; rotational_convection.cpp -- the lagged-vorticity
# rotational term's setup/apply/nodal-skew kernels (same tile machinery);
# point_block_jacobi.cpp -- essential-dof rule, block fill and byNODES
# gather/scatter foralls for the velocity point-block Jacobi.
#
# Why it must be nvcc: MFEM_HOST_DEVICE and forall's GPU dispatch are both
# guarded on __CUDACC__ (mfem config/config.hpp, general/forall.hpp). Compiled
# by a host compiler the kernels do not fail to build -- they quietly become
# host loops iterating over device pointers, which segfaults with "Invalid
# permissions" at run time. grad_div_integrator.cpp carries NO #error guard
# for this (it is written to go upstream into MFEM, which would not carry one),
# so this LANGUAGE CUDA property is the only thing preventing that -- keep it.
# The bessemer-specific entries open with the #error guard (CLAUDE.md).
if(USE_CUDA)
  set_source_files_properties(operators/grad_div_integrator.cpp
                              operators/rotational_convection.cpp
                              precond/point_block_jacobi.cpp
                              PROPERTIES LANGUAGE CUDA)
  target_link_libraries(incns PUBLIC CUDA::cudart)
endif()

# Position-independent so the static library links into the pybind11 shared
# module; harmless for the executables.
set_target_properties(incns PROPERTIES POSITION_INDEPENDENT_CODE ON)

# The YAML deck loader; PRIVATE -- no yaml-cpp types leak into public headers.
find_package(yaml-cpp REQUIRED)
target_link_libraries(incns PRIVATE yaml-cpp::yaml-cpp)

# PRIVATE: libincns' own hot paths get INCNS_PROFILE; the define does not
# propagate to consumers, so they choose their own profiling toggle.
if(INCNS_PROFILING)
  target_compile_definitions(incns PRIVATE INCNS_ENABLE_PROFILING)
endif()
