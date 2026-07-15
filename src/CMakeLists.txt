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
  solver/case.cpp)

# Public: headers are included as "mesh/...", "spaces/..." from this root, and
# every consumer needs MFEM.
target_include_directories(incns PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(incns PUBLIC incns_mfem)

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
