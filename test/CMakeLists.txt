# CTest registration. Per CLAUDE.md, EVERY test runs under MPI at 1, 2, and 4
# ranks via the configured launcher; np=1 exists for debugging, green requires
# np in {2, 4}. Real unit tests + convergence oracles (labelled `fast`) arrive
# from sub-sprint 1.1 onward. For 1.0b the only test is the harness smoke.

find_program(MPIEXEC_EXECUTABLE NAMES mpirun REQUIRED)

# Rank counts every test is registered at (guardrail: never above 4).
set(INCNS_TEST_NPS 1 2 4)

foreach(np IN LISTS INCNS_TEST_NPS)
  add_test(
    NAME hello_mpi_np${np}
    COMMAND ${MPIEXEC_EXECUTABLE} -np ${np} $<TARGET_FILE:hello_mpi>)
  set_tests_properties(hello_mpi_np${np} PROPERTIES
    LABELS "smoke"
    PROCESSORS ${np}
    # Each rank count verifies its own launch: the printed rank total must match.
    PASS_REGULAR_EXPRESSION "incns hello_mpi: ranks=${np} global_dofs=9")
endforeach()
