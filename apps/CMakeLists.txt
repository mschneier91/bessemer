# Harness smoke executable. The compiler is the spack `mpicxx` (set by the cpu
# preset), so MPI is linked by the wrapper; incns_mfem brings MFEM + its TPLs.
add_executable(hello_mpi hello_mpi.cpp)
target_link_libraries(hello_mpi PRIVATE incns_mfem)

# Case drivers: the generic YAML deck runner and the in-code TGV example.
add_executable(run_case run_case.cpp)
target_link_libraries(run_case PRIVATE incns)

add_executable(taylor_green taylor_green.cpp)
target_link_libraries(taylor_green PRIVATE incns)

# 3D unsteady-Stokes MMS driver, for CPU-vs-GPU timing (self-validating).
add_executable(unsteady_mms_3d unsteady_mms_3d.cpp)
target_link_libraries(unsteady_mms_3d PRIVATE incns)

# DFG flow-around-a-cylinder benchmark: lift/drag (John's volume formulation),
# the curved cylinder-channel mesh, optional AMR.
add_executable(dfg_cylinder dfg_cylinder.cpp)
target_link_libraries(dfg_cylinder PRIVATE incns)

# Square cylinder in a large domain (Joly, Etienne & Pelletier 2012): IMEX,
# CFL-controlled steps, AMR in simulated time, live dt/force progress.
add_executable(square_cylinder square_cylinder.cpp)
target_link_libraries(square_cylinder PRIVATE incns)

# Where a checkpointed AMR run put its elements, pass by pass (checks an
# amr.end_time freeze).
add_executable(mesh_census mesh_census.cpp)
target_link_libraries(mesh_census PRIVATE incns)

# dFEM probe (MFEM 4.10's fem/dfem): the vector mass as MFEM partial assembly
# and as a dFEM DifferentiableOperator -- agreement, time per apply, memory.
# Its quadrature-point function is a device lambda: nvcc in a CUDA build.
add_executable(dfem_probe dfem_probe.cpp)
target_link_libraries(dfem_probe PRIVATE incns)
if(USE_CUDA)
  set_source_files_properties(dfem_probe.cpp PROPERTIES LANGUAGE CUDA)
endif()

# DFG 2D-3 with the semi-implicit / implicit BDF2 Oseen schemes (src/oseen/,
# docs/design/oseen_bdf2.md).
add_executable(oseen_cylinder oseen_cylinder.cpp)
target_link_libraries(oseen_cylinder PRIVATE incns)
