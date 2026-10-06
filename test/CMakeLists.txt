# CTest registration. Per CLAUDE.md, EVERY test runs under MPI at 1, 2, and 4
# ranks via the configured launcher; np=1 exists for debugging, green requires
# np in {2, 4}.
#
# Labels:
#   smoke -- harness-only (hello_mpi), all np
#   fast  -- the real gate: every test at np in {1, 2, 4}. Green means THIS.
#   dev   -- INNER-LOOP PRE-FILTER ONLY: the np=1 registration of each `fast`
#            test, additionally labelled. Same binaries, same assertions, just
#            the serial rank count. It does NOT redefine green (np=1 cannot see
#            partition-boundary or reduction bugs) -- it exists so a small edit
#            can be checked in seconds instead of ~25 min. See INCNS_DEV_NP.
#   slow  -- the full CC robustness grid, np=4, on demand only.

find_program(MPIEXEC_EXECUTABLE NAMES mpirun REQUIRED)
find_package(GTest REQUIRED)
find_package(yaml-cpp REQUIRED)

# Rank counts every test is registered at (guardrail: never above 4).
set(INCNS_TEST_NPS 1 2 4)

# Which rank count also carries the `dev` label (see the header comment).
#
# WHY np=1: on a 1-GPU allocation the np2/np4 registrations of a GPU-touching
# test buy little parallel coverage relative to what they cost -- measured
# 2026-07-24 on one H100, stokes_solver_test took 4.8 s at np1 and 47.3 s at
# np2, ~10x for identical assertions, purely from two ranks serializing on one
# card. Dropping to np=1 removes that contention tax without dropping a test.
set(INCNS_DEV_NP 1)

# --- harness smoke (from 1.0b): pure MPI/MFEM hello, label `smoke` ----------
foreach(np IN LISTS INCNS_TEST_NPS)
  add_test(
    NAME hello_mpi_np${np}
    COMMAND ${MPIEXEC_EXECUTABLE} -np ${np} $<TARGET_FILE:hello_mpi>)
  set_tests_properties(hello_mpi_np${np} PROPERTIES
    LABELS "smoke"
    PROCESSORS ${np}
    PASS_REGULAR_EXPRESSION "incns hello_mpi: ranks=${np} global_dofs=9")
endforeach()

# --- module unit tests (googletest), label `fast` --------------------------
# Shared MPI-aware entry point linked into every gtest binary.
add_library(incns_test_main STATIC mpi_gtest_main.cpp)
# incns (not just incns_mfem) so the shared main can call incns::ConfigureDevice
# and see the src/ headers.
target_link_libraries(incns_test_main PUBLIC GTest::gtest incns)

# Register a gtest module test at np in {1, 2, 4}. Any extra arguments are added
# as PRIVATE compile definitions (used to toggle INCNS_ENABLE_PROFILING).
function(incns_add_mpi_test name src)
  add_executable(${name} ${src})
  target_link_libraries(${name} PRIVATE incns incns_test_main)
  if(ARGN)
    target_compile_definitions(${name} PRIVATE ${ARGN})
  endif()
  foreach(np IN LISTS INCNS_TEST_NPS)
    add_test(
      NAME ${name}_np${np}
      COMMAND ${MPIEXEC_EXECUTABLE} -np ${np} $<TARGET_FILE:${name}>)
    # The np=1 run carries `dev` IN ADDITION to `fast`, so `-L dev` selects the
    # serial pre-filter while `-L fast` still selects everything it always did.
    set(_labels "fast")
    if(np EQUAL INCNS_DEV_NP)
      list(APPEND _labels "dev")
    endif()
    set_tests_properties(${name}_np${np} PROPERTIES
      LABELS "${_labels}"
      PROCESSORS ${np})
  endforeach()
endfunction()

incns_add_mpi_test(periodic_box_test periodic_box_test.cpp)
incns_add_mpi_test(mixed_spaces_test mixed_spaces_test.cpp)
incns_add_mpi_test(rule_book_test rule_book_test.cpp)

# Profiler: one binary with the toggle ON (times/reductions), one with it OFF
# (proves INCNS_PROFILE compiles and is a no-op) -- the "compiles both ways" gate.
incns_add_mpi_test(profiler_test profiler_test.cpp INCNS_ENABLE_PROFILING)
incns_add_mpi_test(profiler_off_test profiler_off_test.cpp)
incns_add_mpi_test(pressure_mean_test pressure_mean_test.cpp)
incns_add_mpi_test(stokes_operator_test stokes_operator_test.cpp)
incns_add_mpi_test(convection_test convection_test.cpp)
incns_add_mpi_test(nse_mms_test nse_mms_test.cpp)
incns_add_mpi_test(boundary_conditions_test boundary_conditions_test.cpp)
incns_add_mpi_test(stokes_solver_test stokes_solver_test.cpp)

# Solver-health baselines come from test/baselines.yaml -- the single source of
# truth for stored numbers (never magic numbers in test source).
incns_add_mpi_test(schur_quality_test schur_quality_test.cpp
  INCNS_BASELINES_FILE="${CMAKE_CURRENT_SOURCE_DIR}/baselines.yaml")
target_link_libraries(schur_quality_test PRIVATE yaml-cpp::yaml-cpp)

incns_add_mpi_test(multistep_coeffs_test multistep_coeffs_test.cpp)
incns_add_mpi_test(unsteady_mms_test unsteady_mms_test.cpp)
incns_add_mpi_test(tgv_stokes_temporal_test tgv_stokes_temporal_test.cpp)
# The NSE counterpart: unforced TGV with convection ON (exact NSE solution,
# nonzero analytic pressure) + the analytic energy-decay harness that the 2.2b
# convective-form comparison will read.
incns_add_mpi_test(tgv_nse_test tgv_nse_test.cpp)
incns_add_mpi_test(adaptive_controller_test adaptive_controller_test.cpp)
incns_add_mpi_test(adaptive_tgv_test adaptive_tgv_test.cpp)
incns_add_mpi_test(grad_div_test grad_div_test.cpp)
incns_add_mpi_test(grad_div_integrator_test grad_div_integrator_test.cpp)
# Self-contained: MFEM + the integrator header only.
incns_add_mpi_test(grad_div_integrator_elasticity_test grad_div_integrator_elasticity_test.cpp)
incns_add_mpi_test(rotational_convection_test rotational_convection_test.cpp)
incns_add_mpi_test(point_block_jacobi_test point_block_jacobi_test.cpp)
incns_add_mpi_test(kinetic_head_test kinetic_head_test.cpp)
incns_add_mpi_test(diagnostics_test diagnostics_test.cpp)
incns_add_mpi_test(mass_inverse_test mass_inverse_test.cpp)
incns_add_mpi_test(nullspace_test nullspace_test.cpp)
incns_add_mpi_test(mixed_poisson_test mixed_poisson_test.cpp)
incns_add_mpi_test(lp_surrogate_test lp_surrogate_test.cpp)
incns_add_mpi_test(cahouet_chabard_test cahouet_chabard_test.cpp)
# AMR (amr_spec.md): the solver on nonconforming meshes, then the AMR modules.
incns_add_mpi_test(nc_stokes_test nc_stokes_test.cpp)
incns_add_mpi_test(amr_indicator_test amr_indicator_test.cpp)
incns_add_mpi_test(amr_marker_test amr_marker_test.cpp)
incns_add_mpi_test(amr_transfer_test amr_transfer_test.cpp)
incns_add_mpi_test(cc_robustness_test cc_robustness_test.cpp
  INCNS_BASELINES_FILE="${CMAKE_CURRENT_SOURCE_DIR}/baselines.yaml")
target_link_libraries(cc_robustness_test PRIVATE yaml-cpp::yaml-cpp)

# The FULL CC robustness grid: `slow` label, np = 4 only, run on demand via
# `scripts/test.sh cpu -L slow` -- never in the agentic loop (~10-20 min).
add_executable(cc_sweep_slow_test cc_sweep_slow_test.cpp)
target_link_libraries(cc_sweep_slow_test PRIVATE incns incns_test_main
  yaml-cpp::yaml-cpp)
target_compile_definitions(cc_sweep_slow_test PRIVATE
  INCNS_BASELINES_FILE="${CMAKE_CURRENT_SOURCE_DIR}/baselines.yaml")
add_test(NAME cc_sweep_slow_np4
  COMMAND ${MPIEXEC_EXECUTABLE} -np 4 $<TARGET_FILE:cc_sweep_slow_test>)
set_tests_properties(cc_sweep_slow_np4 PROPERTIES
  LABELS "slow" PROCESSORS 4 TIMEOUT 3600)
incns_add_mpi_test(nondim_test nondim_test.cpp)
incns_add_mpi_test(checkpoint_test checkpoint_test.cpp)
incns_add_mpi_test(bc_integration_test bc_integration_test.cpp)
incns_add_mpi_test(deck_test deck_test.cpp
  INCNS_TGV_DECK="${CMAKE_SOURCE_DIR}/cases/tgv2d_stokes.yaml")

# The actual driver executables, run on the committed deck (label fast).
foreach(np IN LISTS INCNS_TEST_NPS)
  set(_labels "fast")
  if(np EQUAL INCNS_DEV_NP)
    list(APPEND _labels "dev")
  endif()
  add_test(NAME run_case_np${np}
    COMMAND ${MPIEXEC_EXECUTABLE} -np ${np} $<TARGET_FILE:run_case>
            ${CMAKE_SOURCE_DIR}/cases/tgv2d_stokes.yaml)
  set_tests_properties(run_case_np${np} PROPERTIES
    LABELS "${_labels}" PROCESSORS ${np}
    PASS_REGULAR_EXPRESSION "run_case: done")
  add_test(NAME taylor_green_np${np}
    COMMAND ${MPIEXEC_EXECUTABLE} -np ${np} $<TARGET_FILE:taylor_green>)
  set_tests_properties(taylor_green_np${np} PROPERTIES
    LABELS "${_labels}" PROCESSORS ${np}
    PASS_REGULAR_EXPRESSION "taylor_green: done")
endforeach()
