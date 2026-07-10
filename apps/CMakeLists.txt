# Harness smoke executable. The compiler is the spack `mpicxx` (set by the cpu
# preset), so MPI is linked by the wrapper; incns_mfem brings MFEM + its TPLs.
add_executable(hello_mpi hello_mpi.cpp)
target_link_libraries(hello_mpi PRIVATE incns_mfem)
