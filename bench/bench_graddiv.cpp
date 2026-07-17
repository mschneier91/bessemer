// Micro-benchmark (vecdivdiv_spec.md par.8): AddMult throughput of
//   (a) incns::VectorDivDivIntegrator PA   (fused sum-factorized, this work)
//   (b) mfem::ElasticityIntegrator(Q, mu=0) PA (the status quo it replaces)
//   (c) mfem::VectorDiffusionIntegrator PA (structural upper bound)
// on 3D hexes. CPU numbers now; GPU acceptance (>= 5x over (b) at p >= 4)
// re-measured at PSC. Not a ctest -- run manually:
//   mpirun -np 1 build/cpu/bench/bench_graddiv

#include "operators/vecdivdiv_integrator.hpp"
#include "mfem.hpp"

#include <chrono>
#include <cmath>
#include <cstdio>
#include <functional>

using namespace mfem;

int main(int argc, char** argv)
{
   Mpi::Init(argc, argv);
   Hypre::Init();

   for (int p : {2, 4, 6})
   {
      // Size the mesh for ~1M vector dofs at each order.
      const int n = (p == 2) ? 40 : (p == 4) ? 20 : 14;
      Mesh serial = Mesh::MakeCartesian3D(n, n, n, Element::HEXAHEDRON,
                                          1.0, 1.0, 1.0);
      ParMesh mesh(MPI_COMM_WORLD, serial);
      H1_FECollection fec(p, 3);
      ParFiniteElementSpace fes(&mesh, &fec, 3, Ordering::byNODES);
      const HYPRE_BigInt ndof = fes.GlobalTrueVSize();

      IntegrationRules gl(0, Quadrature1D::GaussLegendre);
      const IntegrationRule& ir = gl.Get(Geometry::CUBE, 2 * p + 2);
      ConstantCoefficient one(1.0), zero(0.0);

      auto bench = [&](const char* name,
                       std::function<BilinearFormIntegrator*()> make)
      {
         ParBilinearForm a(&fes);
         a.AddDomainIntegrator(make());
         a.SetAssemblyLevel(AssemblyLevel::PARTIAL);
         a.Assemble();
         Array<int> empty;
         OperatorPtr A;
         a.FormSystemMatrix(empty, A);
         Vector x(fes.GetTrueVSize()), y(fes.GetTrueVSize());
         x.Randomize(1);
         A->Mult(x, y); // warm-up (lazy setups)
         const int reps = 50;
         MPI_Barrier(MPI_COMM_WORLD);
         const auto t0 = std::chrono::steady_clock::now();
         for (int r = 0; r < reps; ++r) { A->Mult(x, y); }
         MPI_Barrier(MPI_COMM_WORLD);
         const auto t1 = std::chrono::steady_clock::now();
         const double s =
            std::chrono::duration<double>(t1 - t0).count() / reps;
         if (Mpi::Root())
         {
            std::printf("  %-28s %8.3f ms/apply   %7.3f GDOF/s\n", name,
                        1e3 * s, static_cast<double>(ndof) / s / 1e9);
         }
      };

      if (Mpi::Root())
      {
         std::printf("p = %d, %lld dofs (%d^3 hexes):\n", p,
                     static_cast<long long>(ndof), n);
      }
      bench("VectorDivDiv (this work)", [&]()
      {
         auto* i = new incns::VectorDivDivIntegrator(one);
         i->SetIntRule(&ir);
         return i;
      });
      bench("Elasticity(mu=0) status quo", [&]()
      {
         auto* i = new ElasticityIntegrator(one, zero);
         i->SetIntRule(&ir);
         return i;
      });
      bench("VectorDiffusion (bound)", [&]()
      {
         auto* i = new VectorDiffusionIntegrator(one);
         i->SetIntRule(&ir);
         return i;
      });
   }
   return 0;
}
