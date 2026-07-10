# libincns -- the solver library. Grows one module per sub-sprint; a driver or
# test never edits it to run a case. Sprint 1.1 seeds it with mesh + spaces.
add_library(incns STATIC
  mesh/periodic_box.cpp
  spaces/mixed_spaces.cpp
  quadrature/rule_book.cpp
  util/profiler.cpp
  post/pressure_mean.cpp)

# Public: headers are included as "mesh/...", "spaces/..." from this root, and
# every consumer needs MFEM.
target_include_directories(incns PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(incns PUBLIC incns_mfem)

# PRIVATE: libincns' own hot paths get INCNS_PROFILE; the define does not
# propagate to consumers, so they choose their own profiling toggle.
if(INCNS_PROFILING)
  target_compile_definitions(incns PRIVATE INCNS_ENABLE_PROFILING)
endif()
