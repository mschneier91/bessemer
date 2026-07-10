// Sprint 1.0b harness smoke target: prove the full spack toolchain + MFEM + MPI
// chain builds and runs at np in {1, 2, 4}. Not a solver -- it exercises just
// enough of MFEM (a partitioned ParMesh + H1 space) that a broken build or MPI
// layer fails loudly. Deterministic output across rank counts is asserted by the
// ctest PASS_REGULAR_EXPRESSION (see test/CMakeLists.txt).

#include "mfem.hpp"

#include <iostream>

using namespace mfem;

int main(int argc, char* argv[])
{
   Mpi::Init(argc, argv);
   Hypre::Init();

   const int size = Mpi::WorldSize();

   // Tiny 2x2 quad mesh, partitioned across ranks; Q1 => 3x3 = 9 global dofs,
   // independent of the partition, so the check is rank-count robust.
   Mesh serial = Mesh::MakeCartesian2D(2, 2, Element::QUADRILATERAL);
   ParMesh mesh(MPI_COMM_WORLD, serial);
   H1_FECollection fec(1, mesh.Dimension());
   ParFiniteElementSpace fes(&mesh, &fec);

   const HYPRE_BigInt global_dofs = fes.GlobalTrueVSize();

   if (Mpi::Root())
   {
      std::cout << "incns hello_mpi: ranks=" << size
                << " global_dofs=" << global_dofs << std::endl;
   }

   MFEM_VERIFY(global_dofs == 9, "unexpected global dof count");

   return 0;
}
