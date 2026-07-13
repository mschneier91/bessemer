# The pybind11 module (built only when INCNS_PYTHON=ON; the numerics build never
# depends on Python). Produces build/python/incns/_core*.so next to a staged copy
# of the pure-Python package, so `PYTHONPATH=build/<preset>/python` -> import incns.
find_package(pybind11 REQUIRED)

pybind11_add_module(incns_pymodule bindings.cpp)
set_target_properties(incns_pymodule PROPERTIES
  OUTPUT_NAME _core
  LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/python/incns)
target_link_libraries(incns_pymodule PRIVATE incns)

# Stage the Python package (re-copied whenever __init__.py changes).
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/incns/__init__.py
               ${CMAKE_BINARY_DIR}/python/incns/__init__.py COPYONLY)

# The Stokes transient MMS example, run through the Python interface at
# np in {1,2,4}: defines the exact solution + forcing as compiled fields,
# marches, and checks the reproduced-to-tolerance error (label `fast`).
find_program(MPIEXEC_EXECUTABLE NAMES mpirun REQUIRED)
find_program(INCNS_PYTHON_EXE NAMES python3 REQUIRED)
set(INCNS_TEST_NPS 1 2 4)
foreach(np IN LISTS INCNS_TEST_NPS)
  add_test(
    NAME py_stokes_ex_np${np}
    COMMAND ${MPIEXEC_EXECUTABLE} -np ${np} ${INCNS_PYTHON_EXE}
            ${CMAKE_SOURCE_DIR}/examples/python/stokes_ex/run.py)
  set_tests_properties(py_stokes_ex_np${np} PROPERTIES
    LABELS "fast"
    PROCESSORS ${np}
    ENVIRONMENT "PYTHONPATH=${CMAKE_BINARY_DIR}/python"
    PASS_REGULAR_EXPRESSION "stokes_ex: PASSED"
    DEPENDS incns_pymodule)

  # The same MMS driven from a YAML deck (config + BC topology in YAML).
  add_test(
    NAME py_stokes_ex_yaml_np${np}
    COMMAND ${MPIEXEC_EXECUTABLE} -np ${np} ${INCNS_PYTHON_EXE}
            ${CMAKE_SOURCE_DIR}/examples/python/stokes_ex/run_yaml.py)
  set_tests_properties(py_stokes_ex_yaml_np${np} PROPERTIES
    LABELS "fast"
    PROCESSORS ${np}
    ENVIRONMENT "PYTHONPATH=${CMAKE_BINARY_DIR}/python"
    PASS_REGULAR_EXPRESSION "stokes_ex_yaml: PASSED"
    DEPENDS incns_pymodule)

  # No-slip channel from a deck (the `no_slip` boundary type, no field bound).
  add_test(
    NAME py_channel_noslip_np${np}
    COMMAND ${MPIEXEC_EXECUTABLE} -np ${np} ${INCNS_PYTHON_EXE}
            ${CMAKE_SOURCE_DIR}/examples/python/stokes_ex/run_channel.py)
  set_tests_properties(py_channel_noslip_np${np} PROPERTIES
    LABELS "fast"
    PROCESSORS ${np}
    ENVIRONMENT "PYTHONPATH=${CMAKE_BINARY_DIR}/python"
    PASS_REGULAR_EXPRESSION "channel_noslip: PASSED"
    DEPENDS incns_pymodule)
endforeach()
