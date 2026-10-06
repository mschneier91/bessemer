// bench_rotation_pc -- velocity-block preconditioners with the semi-implicit
// rotation term (rotational_convection_pa_spec.md Table A2 setup).
//
// A = sigma M + nu K + N(omega), 3D, p = 4, curved mesh M2 (3x3x3), Dirichlet
// on the whole boundary, each term at the solver's Gauss-Legendre rule (mass
// order 2p, diffusion 2p + 2, N the dealiased 3p its kernels are specialized
// for), sigma = 1/dt (dt = 1). GMRES(50) to 1e-10 on a hashed-noise
// right-hand side, preconditioned by:
//   jacobi    scalar Jacobi (sees nothing of N)
//   lor       LOR-AMG on sigma M + nu K (today's LOR path; sees nothing of N)
//   lor+N     LOR-AMG on sigma M + nu K + N (w copied to the LOR space)
//   pbj       point-block Jacobi
// over max|omega| dt in {0, 0.1, 1, 10, 100} and nu dt / h^2 in {0.01, 1, 100}.
// Prints iteration counts ("fail" = no convergence in 3000).
//
// Usage: mpirun -np <=4 build/cpu/bench/bench_rotation_pc [p] [n]

#include "operators/convection.hpp"
#include "operators/rotational_convection.hpp"
#include "precond/point_block_jacobi.hpp"
#include "mfem.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <string>

using namespace mfem;
using incns::PointBlockJacobi;
using incns::VectorRotationalConvectionIntegrator;

namespace
{
void vel_fn(const Vector& x, Vector& u) // spec App. B, 3D
{
   const real_t X = x(0), Y = x(1), Z = x(2);
   u(0) = sin(M_PI * X) * cos(1.3 * M_PI * Y) * cos(0.6 * Z) + 0.3 * Y;
   u(1) = cos(0.7 * M_PI * X) * sin(M_PI * Y) * sin(0.9 * Z) - 0.2 * X * X;
   u(2) = sin(0.8 * X + 1.1 * Y) * cos(M_PI * Z) + 0.1 * Z * Y;
}

void noise_fn(const Vector& x, Vector& u)
{
   for (int c = 0; c < u.Size(); ++c)
   {
      const real_t s = sin(12.9898 * x(0) + 78.233 * x(1) + 37.719 * x(2) +
                           19.17 * c) * 43758.5453;
      u(c) = s - std::floor(s) - 0.5;
   }
}

Mesh MakeCurvedMesh(int n, int order) // M2 (spec App. B), 3D
{
   Mesh mesh = Mesh::MakeCartesian3D(n, n, n, Element::HEXAHEDRON, 1.0, 1.0, 1.0);
   mesh.SetCurvature(order, false, 3, Ordering::byNODES);
   mesh.Transform([](const Vector & x, Vector & y)
   {
      y = x;
      y(1) += 0.2 * x(0);
      y(2) += 0.3 * x(0);
      y(0) += 0.04 * sin(M_PI * x(1)) * sin(M_PI * x(0));
      y(1) += 0.04 * sin(M_PI * x(0)) * sin(2 * M_PI * x(1));
      y(2) += 0.03 * sin(M_PI * x(0)) * sin(M_PI * x(2));
   });
   return mesh;
}

// max |curl w| over the points of ir in every element, global over ranks,
// from MFEM's GridFunction::GetCurl (2D: the scalar vorticity).
real_t MaxVorticity(const ParGridFunction& w, const IntegrationRule& ir)
{
   ParMesh& mesh = *w.ParFESpace()->GetParMesh();
   w.HostRead();
   Vector curl;
   real_t mx = 0.0;
   for (int e = 0; e < mesh.GetNE(); ++e)
   {
      ElementTransformation& T = *mesh.GetElementTransformation(e);
      for (int q = 0; q < ir.GetNPoints(); ++q)
      {
         T.SetIntPoint(&ir.IntPoint(q));
         w.GetCurl(T, curl);
         mx = std::max(mx, curl.Norml2());
      }
   }
   MPI_Allreduce(MPI_IN_PLACE, &mx, 1, MPITypeMap<real_t>::mpi_type, MPI_MAX,
                 mesh.GetComm());
   return mx;
}
} // namespace

int main(int argc, char** argv)
{
   Mpi::Init(argc, argv);
   Hypre::Init();
   const int p = (argc > 1) ? std::atoi(argv[1]) : 4;
   const int n = (argc > 2) ? std::atoi(argv[2]) : 3;

   Mesh serial = MakeCurvedMesh(n, p);
   ParMesh mesh(MPI_COMM_WORLD, serial);
   H1_FECollection fec(p, 3);
   ParFiniteElementSpace fes(&mesh, &fec, 3, Ordering::byNODES);
   // The solver's rules (StokesOperator).
   const IntegrationRule& ir_mass = IntRules.Get(Geometry::CUBE, 2 * p);
   const IntegrationRule& ir_diff = IntRules.Get(Geometry::CUBE, 2 * p + 2);
   const IntegrationRule& ir_rot =
      IntRules.Get(Geometry::CUBE, incns::Convection::DealiasedOrder(p));
   Array<int> ess, bdr(mesh.bdr_attributes.Max());
   bdr = 1;
   fes.GetEssentialTrueDofs(bdr, ess);

   ParGridFunction w(&fes), bgf(&fes);
   VectorFunctionCoefficient wc(3, vel_fn), bc(3, noise_fn);
   w.ProjectCoefficient(wc);
   bgf.ProjectCoefficient(bc);
   Vector W(fes.GetTrueVSize()), b(fes.GetTrueVSize());
   w.GetTrueDofs(W);
   w.SetFromTrueDofs(W);
   bgf.GetTrueDofs(b);

   const real_t om_max = MaxVorticity(w, ir_rot);

   ParLORDiscretization lor_disc(fes);
   ParGridFunction w_lor(&lor_disc.GetParFESpace());
   w_lor.SetFromTrueDofs(W);

   const real_t h = 1.0 / n, dt = 1.0;
   if (Mpi::Root())
   {
      std::printf("3D p=%d n=%d (%d^3 elements), %lld velocity dofs, curved M2, "
                  "Dirichlet, solver rules, GMRES(50) to 1e-10, np=%d\n", p, n,
                  n, static_cast<long long>(fes.GlobalTrueVSize()),
                  Mpi::WorldSize());
      std::printf("%-12s %-10s %8s %8s %8s %8s\n", "nu dt/h^2", "|w| dt",
                  "jacobi", "lor", "lor+N", "pbj");
   }
   for (real_t vr : {0.01, 1.0, 100.0})
   {
      for (real_t rot : {0.0, 0.1, 1.0, 10.0, 100.0})
      {
         ConstantCoefficient sigma(1.0 / dt), nu(vr * h * h / dt);
         const real_t alpha = rot / (om_max * dt);

         // The operator (PA) and its diagonal.
         ParBilinearForm a(&fes);
         auto* m = new VectorMassIntegrator(sigma);
         auto* k = new VectorDiffusionIntegrator(nu);
         auto* r = new VectorRotationalConvectionIntegrator(w, alpha);
         m->SetIntRule(&ir_mass);
         k->SetIntRule(&ir_diff);
         r->SetIntRule(&ir_rot);
         a.AddDomainIntegrator(m);
         a.AddDomainIntegrator(k);
         a.AddDomainIntegrator(r);
         a.SetAssemblyLevel(AssemblyLevel::PARTIAL);
         a.Assemble();
         OperatorPtr A;
         a.FormSystemMatrix(ess, A);
         Vector diag(fes.GetTrueVSize());
         a.AssembleDiagonal(diag);

         // Symmetric LOR source and the LOR source with N.
         ParBilinearForm sym(&fes), full(&fes);
         sym.AddDomainIntegrator(new VectorMassIntegrator(sigma));
         sym.AddDomainIntegrator(new VectorDiffusionIntegrator(nu));
         full.AddDomainIntegrator(new VectorMassIntegrator(sigma));
         full.AddDomainIntegrator(new VectorDiffusionIntegrator(nu));
         full.AddDomainIntegrator(
                new VectorRotationalConvectionIntegrator(w_lor, alpha));

         OperatorJacobiSmoother jac(diag, ess);
         LORSolver<HypreBoomerAMG> lor(sym, ess);
         lor_disc.AssembleSystem(full, ess);
         LORSolver<HypreBoomerAMG> lorn(lor_disc.GetAssembledMatrix(), lor_disc);
         for (auto* s : {&lor, &lorn})
         {
            s->GetSolver().SetSystemsOptions(3, true);
            s->GetSolver().SetPrintLevel(0);
            s->GetSolver().iterative_mode = false;
         }
         PointBlockJacobi pbj(fes, *r, ess);
         pbj.SetDiagonal(diag);
         pbj.UpdateSkew();

         std::string row;
         for (Solver* pc : std::initializer_list<Solver*> {&jac, &lor, &lorn, &pbj})
         {
            GMRESSolver gmres(MPI_COMM_WORLD);
            gmres.SetKDim(50);
            gmres.SetRelTol(1e-10);
            gmres.SetAbsTol(0.0);
            gmres.SetMaxIter(3000);
            gmres.SetPrintLevel(-1);
            gmres.SetOperator(*A);
            gmres.SetPreconditioner(*pc);
            Vector x(b.Size());
            x = 0.0;
            gmres.Mult(b, x);
            char cell[16];
            if (gmres.GetConverged())
            {
               std::snprintf(cell, sizeof(cell), " %8d", gmres.GetNumIterations());
            }
            else { std::snprintf(cell, sizeof(cell), " %8s", "fail"); }
            row += cell;
         }
         if (Mpi::Root())
         {
            std::printf("%-12g %-10g%s\n", vr, rot, row.c_str());
            std::fflush(stdout);
         }
      }
   }
   return 0;
}
