// Micro-benchmark of the hyperelastic operators (docs/design/hyperelasticity_spec.md,
// the memory and performance step after M1): for each order and tangent storage,
//   - the bytes the operator owns per vector dof (HyperelasticOperator::MemoryBytes)
//     and, on a GPU, the device memory it adds (driver-reported, construction and
//     first linearization);
//   - the time of a residual apply, a linearization and a tangent apply, as
//     milliseconds and million vector dofs per second;
// and, for reference, MFEM's ElasticityIntegrator in partial assembly (the linear
// solver's operator; skipped on a GPU where it cannot launch, 3D order >= 6).
// 3D Cartesian hexes sized for about -dofs vector dofs. Not a ctest -- run by hand:
//   mpirun -np 1 build/cpu/bench/bench_hyperelastic -dofs 1e6
//   INCNS_DEVICE=cuda mpirun -np 1 build/cuda-43/bench/bench_hyperelastic -dofs 2e7

#include "quadrature/rule_book.hpp"
#include "solid/hyperelastic_operator.hpp"
#include "util/device.hpp"
#include "mfem.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

using namespace mfem;

namespace
{

/// Seconds per call of @p f over @p reps calls, after one warm-up call.
double Time(int reps, const std::function<void()>& f)
{
   f();
   incns::DeviceSync();
   MPI_Barrier(MPI_COMM_WORLD);
   const auto t0 = std::chrono::steady_clock::now();
   for (int r = 0; r < reps; ++r) { f(); }
   incns::DeviceSync();
   MPI_Barrier(MPI_COMM_WORLD);
   const auto t1 = std::chrono::steady_clock::now();
   return std::chrono::duration<double>(t1 - t0).count() / reps;
}

double MiB(double bytes) { return bytes / (1024.0 * 1024.0); }

const char* Name(incns::TangentStorage s)
{
   switch (s)
   {
      case incns::TangentStorage::StressAndDeformation: return "F,S";
      case incns::TangentStorage::Deformation: return "F";
      case incns::TangentStorage::Displacement: return "u";
   }
   return "?";
}

} // namespace

int main(int argc, char** argv)
{
   incns::ConfigureMpiTransport();
   Mpi::Init(argc, argv);
   Hypre::Init();
   double target = 1e6;
   int reps = 20;
   const char* orders = "2 3 4 5 6 7";
   OptionsParser args(argc, argv);
   args.AddOption(&target, "-dofs", "--dofs", "Target vector dofs per run.");
   args.AddOption(&reps, "-reps", "--reps", "Applies timed per measurement.");
   args.AddOption(&orders, "-p", "--orders", "Orders, e.g. \"2 4 6\".");
   args.Parse();
   if (!args.Good())
   {
      if (Mpi::Root()) { args.PrintUsage(std::cout); }
      return 1;
   }
   const char* env_dev = std::getenv("INCNS_DEVICE");
   incns::ConfigureDevice(env_dev ? env_dev : "cpu");
   const bool gpu = Device::Allows(Backend::DEVICE_MASK);
   if (Mpi::Root())
   {
      std::printf("%-3s %-4s %10s %-4s %9s %9s %9s %9s | %9s %9s %9s | %9s %9s %9s\n",
                  "p", "n", "dofs", "tan", "B/dof", "B/lent", "MiB+ctor", "MiB+lin",
                  "res ms", "lin ms", "tan ms", "res Md/s", "lin Md/s", "tan Md/s");
   }

   std::istringstream in(orders);
   int p;
   while (in >> p)
   {
      const int n = std::max(1,
                             static_cast<int>(std::lround(std::cbrt(target / 3.0) / p)));
      Mesh serial = Mesh::MakeCartesian3D(n, n, n, Element::HEXAHEDRON, 1.0, 1.0,
                                          1.0);
      ParMesh mesh(MPI_COMM_WORLD, serial);
      serial.Clear();
      H1_FECollection fec(p, 3);
      ParFiniteElementSpace fes(&mesh, &fec, 3, Ordering::byNODES);
      const HYPRE_BigInt ndof = fes.GlobalTrueVSize();
      incns::RuleBook rules;
      const IntegrationRule& ir = rules.Get(Geometry::CUBE, 2 * p + 1);
      Vector lambda(mesh.GetNE()), mu(mesh.GetNE());
      lambda = 1.7;
      mu = 0.9;
      const int nt = fes.GetTrueVSize();
      Vector u(nt), w(nt), y(nt);
      u.Randomize(1);
      u *= 0.01;
      w.Randomize(2);
      for (Vector* v : {&u, &w, &y}) { v->UseDevice(true); }

      incns::ElementScratch scratch(fes); // for the borrowed-buffer column
      for (incns::TangentStorage s :
           {
              incns::TangentStorage::StressAndDeformation,
              incns::TangentStorage::Deformation,
              incns::TangentStorage::Displacement
           })
      {
         incns::DeviceSync();
         const double m0 = incns::DeviceMemoryInUse();
         incns::HyperelasticOperator op(fes, ir,
                                        incns::HyperelasticMaterial::StVenantKirchhoff,
                                        lambda, mu, s);
         incns::DeviceSync();
         const double m1 = incns::DeviceMemoryInUse();
         Operator& K = op.GetGradient(u);
         incns::DeviceSync();
         const double m2 = incns::DeviceMemoryInUse();
         const double bytes = static_cast<double>(op.MemoryBytes());
         double all = 0.0;
         MPI_Allreduce(&bytes, &all, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
         double lent_all = 0.0;
         {
            incns::HyperelasticOperator lent(fes, ir,
                                             incns::HyperelasticMaterial::StVenantKirchhoff,
                                             lambda, mu, s, &scratch);
            lent.GetGradient(u);
            const double lb = static_cast<double>(lent.MemoryBytes());
            MPI_Allreduce(&lb, &lent_all, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
         }
         const double tr = Time(reps, [&] { op.Mult(u, y); });
         const double tl = Time(reps, [&] { op.GetGradient(u); });
         const double tt = Time(reps, [&] { K.Mult(w, y); });
         if (Mpi::Root())
         {
            std::printf("%-3d %-4d %10lld %-4s %9.1f %9.1f %9.1f %9.1f | %9.3f %9.3f %9.3f | %9.1f %9.1f %9.1f\n",
                        p, n, static_cast<long long>(ndof), Name(s), all / ndof, lent_all / ndof,
                        gpu ? MiB(m1 - m0) : 0.0, gpu ? MiB(m2 - m1) : 0.0,
                        1e3 * tr, 1e3 * tl, 1e3 * tt, 1e-6 * ndof / tr, 1e-6 * ndof / tl,
                        1e-6 * ndof / tt);
            std::fflush(stdout);
         }
      }

      // Reference: MFEM's linear elasticity PA (cannot launch on a GPU at 3D p >= 6).
      if (!(gpu && p >= 6))
      {
         incns::DeviceSync();
         const double m0 = incns::DeviceMemoryInUse();
         ConstantCoefficient cl(1.7), cm(0.9);
         ParBilinearForm a(&fes);
         auto* integ = new ElasticityIntegrator(cl, cm);
         integ->SetIntRule(&ir);
         a.AddDomainIntegrator(integ);
         a.SetAssemblyLevel(AssemblyLevel::PARTIAL);
         a.Assemble();
         Array<int> empty;
         OperatorPtr A;
         a.FormSystemMatrix(empty, A);
         incns::DeviceSync();
         const double m1 = incns::DeviceMemoryInUse();
         const double ta = Time(reps, [&] { A->Mult(w, y); });
         if (Mpi::Root())
         {
            std::printf("%-3d %-4d %10lld %-4s %9s %9s %9.1f %9s | %9s %9s %9.3f | %9s %9s %9.1f   (MFEM linear PA)\n",
                        p, n, static_cast<long long>(ndof), "lin", "-", "-",
                        gpu ? MiB(m1 - m0) : 0.0, "-", "-", "-", 1e3 * ta, "-", "-",
                        1e-6 * ndof / ta);
            std::fflush(stdout);
         }
      }
      mesh.DeleteGeometricFactors(); // keyed by rules' rules
   }
   return 0;
}
