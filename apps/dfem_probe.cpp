// dFEM probe (human 2026-10-15): is MFEM's dFEM (fem/dfem, MFEM 4.10) a path
// for bessemer's operators? The simplest one only: the vector mass, as MFEM's
// partial assembly (VectorMassIntegrator, the kernel the solver applies, at
// bessemer's compile-time sizes) and as a dFEM DifferentiableOperator. The
// same result to roundoff? How long per apply, and how much device memory?
// Periodic box n^3, Q_p vector H1 (byNODES), the 2p rule (the solver's mass
// rule), true dofs on both sides.
//
//   mpirun -np 1 dfem_probe -n 32 -p 3 -d cuda -r 200

#include "mesh/periodic_box.hpp"
#include "operators/kernel_specializations.hpp"
#include "util/device.hpp"
#include "mfem.hpp"

// The quadrature-point function is a device lambda: nvcc must compile this file
// in a CUDA build (apps/CMakeLists.txt's LANGUAGE CUDA).
#if defined(MFEM_USE_CUDA) && !defined(__CUDACC__)
#error "dfem_probe.cpp has a device lambda and must be compiled by nvcc -- see apps/CMakeLists.txt"
#endif

#include <cmath>
#include <iomanip>
#include <iostream>
#include <vector>

using namespace mfem;
using namespace mfem::future;

namespace
{

double Norm(const Vector& v)
{
   return std::sqrt(InnerProduct(MPI_COMM_WORLD, v, v));
}

// Seconds per call of @p apply: 5 untimed calls, then @p reps timed ones
// between device synchronizations.
template <typename F>
double TimePerCall(int reps, F&& apply)
{
   for (int i = 0; i < 5; ++i) { apply(); }
   incns::DeviceSync();
   MPI_Barrier(MPI_COMM_WORLD);
   const double t0 = MPI_Wtime();
   for (int i = 0; i < reps; ++i) { apply(); }
   incns::DeviceSync();
   MPI_Barrier(MPI_COMM_WORLD);
   return (MPI_Wtime() - t0) / reps;
}

} // namespace

int main(int argc, char* argv[])
{
   Mpi::Init(argc, argv);
   Hypre::Init();

   int n = 16, p = 3, reps = 100;
   const char* device = "cpu";
   OptionsParser args(argc, argv);
   args.AddOption(&n, "-n", "--elements", "Elements per direction.");
   args.AddOption(&p, "-p", "--order", "Velocity order.");
   args.AddOption(&reps, "-r", "--reps", "Timed applications of each operator.");
   args.AddOption(&device, "-d", "--device", "MFEM backend: cpu, cuda, ...");
   args.ParseCheck();
   incns::ConfigureDevice(device);
   incns::AddKernelSpecializations();

   constexpr int DIM = 3;
   incns::BoxSpec spec;
   spec.dim = DIM;
   spec.num_elems = {n, n, n};
   spec.lengths = {2.0 * M_PI, 2.0 * M_PI, 2.0 * M_PI};
   Mesh serial = incns::MakeBoxMesh(spec);
   ParMesh pmesh(MPI_COMM_WORLD, serial);
   serial.Clear();
   pmesh.EnsureNodes();
   auto* nodes = static_cast<ParGridFunction*>(pmesh.GetNodes());

   H1_FECollection fec(p, DIM);
   ParFiniteElementSpace fes(&pmesh, &fec, DIM);
   const IntegrationRule& ir = IntRules.Get(Geometry::CUBE, 2 * p);
   const HYPRE_BigInt dofs = fes.GlobalTrueVSize();

   Vector X(fes.GetTrueVSize()), Y(fes.GetTrueVSize()), Z(fes.GetTrueVSize());
   X.Randomize(1);
   X.UseDevice(true);
   Y.UseDevice(true);
   Z.UseDevice(true);

   // MFEM: partial assembly, the true-dof operator P^T M P.
   incns::DeviceSync();
   const double mem0 = incns::DeviceMemoryInUse();
   ParBilinearForm m(&fes);
   auto* vmi = new VectorMassIntegrator;
   vmi->SetIntRule(&ir);
   m.AddDomainIntegrator(vmi);
   m.SetAssemblyLevel(AssemblyLevel::PARTIAL);
   m.Assemble();
   OperatorPtr M;
   Array<int> no_ess;
   m.FormSystemMatrix(no_ess, M);
   M->Mult(X, Y);
   incns::DeviceSync();
   const double mem1 = incns::DeviceMemoryInUse();

   // dFEM: the same integrand as a quadrature-point function, u w det(J).
   static constexpr int U = 0, Coords = 1;
   const auto fields = std::vector{FieldDescriptor{U, &fes}};
   DifferentiableOperator dop(fields, {{Coords, nodes->ParFESpace()}}, pmesh);
   const auto mass_qf =
      [] MFEM_HOST_DEVICE(const tensor<real_t, DIM>& u,
                          const tensor<real_t, DIM, DIM>& J, const real_t& w)
   {
      return tuple{u * (w * det(J))};
   };
   Array<int> all_attr(pmesh.attributes.Max());
   all_attr = 1;
   dop.AddDomainIntegrator(mass_qf,
                           tuple{Value<U>{}, Gradient<Coords>{}, Weight{}},
                           tuple{Value<U>{}}, ir, all_attr);
   dop.SetParameters({nodes});
   dop.Mult(X, Z);
   incns::DeviceSync();
   const double mem2 = incns::DeviceMemoryInUse();

   Vector D(Y);
   D -= Z;
   const double rel = Norm(D) / Norm(Y);

   const double t_mfem = TimePerCall(reps, [&] { M->Mult(X, Y); });
   const double t_dfem = TimePerCall(reps, [&] { dop.Mult(X, Z); });

   if (Mpi::Root())
   {
      const double mib = 1024.0 * 1024.0;
      std::cout << std::setprecision(3)
                << "dfem_probe: device " << device << ", ranks " << Mpi::WorldSize()
                << ", " << n << "^3 Q" << p << " vector mass, " << dofs
                << " true dofs, rule order " << 2 * p << " ("
                << ir.GetNPoints() << " points)\n"
                << "  |M x - dFEM x| / |M x| = " << rel << "\n"
                << "  per apply: MFEM PA " << t_mfem * 1e3 << " ms, dFEM "
                << t_dfem * 1e3 << " ms (dFEM / MFEM " << t_dfem / t_mfem
                << ")\n"
                << "  throughput: MFEM PA " << dofs / t_mfem << " dofs/s, dFEM "
                << dofs / t_dfem << " dofs/s\n"
                << "  device memory (setup + first apply): MFEM PA "
                << (mem1 - mem0) / mib << " MiB, dFEM " << (mem2 - mem1) / mib
                << " MiB\n";
   }
   return 0;
}
