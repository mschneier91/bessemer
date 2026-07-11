# CTest registration. Per CLAUDE.md, EVERY test runs under MPI at 1, 2, and 4
# ranks via the configured launcher; np=1 exists for debugging, green requires
# np in {2, 4}.

find_program(MPIEXEC_EXECUTABLE NAMES mpirun REQUIRED)
find_package(GTest REQUIRED)

# Rank counts every test is registered at (guardrail: never above 4).
set(INCNS_TEST_NPS 1 2 4)

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
target_link_libraries(incns_test_main PUBLIC GTest::gtest incns_mfem)

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
    set_tests_properties(${name}_np${np} PROPERTIES
      LABELS "fast"
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
incns_add_mpi_test(boundary_conditions_test boundary_conditions_test.cpp)
