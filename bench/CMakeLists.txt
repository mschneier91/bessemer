# Micro-benchmarks -- built with the library, never registered in ctest.
add_executable(bench_graddiv bench_graddiv.cpp)
target_link_libraries(bench_graddiv PRIVATE incns)
