# Micro-benchmarks -- built with the library, never registered in ctest.
add_executable(bench_graddiv bench_graddiv.cpp)
target_link_libraries(bench_graddiv PRIVATE incns)
# Velocity-block PCs with the semi-implicit rotation term (iteration table).
add_executable(bench_rotation_pc bench_rotation_pc.cpp)
target_link_libraries(bench_rotation_pc PRIVATE incns)
# Velocity-block PC of the CC path across the viscous ratio (default choice).
add_executable(bench_velocity_pc bench_velocity_pc.cpp)
target_link_libraries(bench_velocity_pc PRIVATE incns)
# Hyperelastic operators: memory per dof and apply throughput by order and tangent storage.
add_executable(bench_hyperelastic bench_hyperelastic.cpp)
target_link_libraries(bench_hyperelastic PRIVATE incns)
