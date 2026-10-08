# Micro-benchmarks -- built with the library, never registered in ctest.
add_executable(bench_graddiv bench_graddiv.cpp)
target_link_libraries(bench_graddiv PRIVATE incns)
# Velocity-block PCs with the semi-implicit rotation term (iteration table).
add_executable(bench_rotation_pc bench_rotation_pc.cpp)
target_link_libraries(bench_rotation_pc PRIVATE incns)
# Velocity-block PC of the CC path across the viscous ratio (default choice).
add_executable(bench_velocity_pc bench_velocity_pc.cpp)
target_link_libraries(bench_velocity_pc PRIVATE incns)
