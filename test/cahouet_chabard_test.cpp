// CC.5 -- the CC Schur PC (SPEC par.6.1, T1d validation, T2 singular
// correctness):
//
//  * T1d: every not-in-v1 / inconsistent config throws std::invalid_argument.
//  * sigma = 0 short-circuits to the pure mass PC (z = nu * M_p^-1 r exactly).
//  * Scaling guard: with a tight inner solve, w = (z - nu Mp^-1 r)/sigma
//    satisfies ||S w - r|| << ||r|| -- pinning that sigma multiplies the
//    POISSON term and nu the mass term (a swapped scaling fails wildly).
//  * Reset(sigma, nu) updates only scalars: results reproduce bitwise-level.
//  * T2b drift: NullspaceMode::ForceOff on an enclosed domain (projections
//    disabled) pollutes the Poisson term along the constant mode (or NaNs);
//    Auto keeps it clean to roundoff.
//  * LaplacianLegacy applies through the same interface and differs.

#include <gtest/gtest.h>

#include "bc/boundary_conditions.hpp"
#include "mesh/periodic_box.hpp"
#include "operators/stokes_operator.hpp"
#include "precond/cahouet_chabard.hpp"
#include "quadrature/rule_book.hpp"
#include "spaces/mixed_spaces.hpp"
#include "mfem.hpp"

#include <cmath>
#include <memory>
#include <stdexcept>

using namespace mfem;
using incns::BoundaryConditions;
using incns::BoxSpec;
using incns::CahouetChabardConfig;
using incns::CahouetChabardSchurPC;
using incns::ConstantPressureProjector;
using incns::MakeBoxMesh;
using incns::MassInverse;
using incns::MassInvType;
using incns::MixedPoissonOperator;
using incns::MixedSpaces;
using incns::NullspaceMode;
using incns::RuleBook;
using incns::SchurModel;
using incns::StokesOperator;
using incns::StokesOperatorOptions;

namespace
{

struct SettingCase
{
   ParMesh mesh;
   std::unique_ptr<MixedSpaces> spaces;
   RuleBook rules;
   std::unique_ptr<BoundaryConditions> bc;
   std::unique_ptr<StokesOperator> op;
   Array<int> outflow_attrs; // empty: enclosed

   SettingCase()
   {
      BoxSpec s;
      s.dim = 2;
      s.num_elems = {4, 4, 0};
      s.lengths = {1.0, 1.0};
      s.periodic = {false, false, false};
      Mesh serial = MakeBoxMesh(s);
      mesh = ParMesh(MPI_COMM_WORLD, serial);
      spaces = std::make_unique<MixedSpaces>(mesh, 3, 2);
      bc = std::make_unique<BoundaryConditions>(spaces->Velocity());
      for (int a = 1; a <= 4; ++a) { bc->AddNoSlip(a); }
      StokesOperatorOptions oo;
      oo.nu = 1.0;
      oo.collocated_mass = true;
      op = std::make_unique<StokesOperator>(*spaces, rules, oo,
                                            &bc->EssentialTrueDofs());
   }

   std::unique_ptr<CahouetChabardSchurPC> MakePC(const CahouetChabardConfig& c)
   {
      return std::make_unique<CahouetChabardSchurPC>(
                c, spaces->Velocity(), spaces->Pressure(), rules,
                op->Divergence(), op->Mass(), op->MassDiagonal(),
                outflow_attrs, bc->PressureNullspaceExists());
   }
};

double Norm(const Vector& v)
{
   return std::sqrt(InnerProduct(MPI_COMM_WORLD, v, v));
}

} // namespace

TEST(CahouetChabard, ConfigValidationThrows)
{
   auto bad = [](auto mutate)
   {
      CahouetChabardConfig c;
      c.sigma = 1.0;
      mutate(c);
      EXPECT_THROW(c.Validate(false), std::invalid_argument);
   };
   bad([](CahouetChabardConfig & c) { c.sigma = -1.0; });
   bad([](CahouetChabardConfig & c) { c.nu = 0.0; });
   bad([](CahouetChabardConfig & c) { c.schur_model = SchurModel::LumpedBMB; });
   bad([](CahouetChabardConfig & c)
   { c.visc_form = incns::ViscousForm::SymGradient; });
   bad([](CahouetChabardConfig & c)
   { c.pc_mass_coeff = incns::PcMassCoeff::ReciprocalNuField; });
   bad([](CahouetChabardConfig & c) { c.lp_pc = incns::LpPC::PMG; });
   bad([](CahouetChabardConfig & c) { c.a_pc = incns::APC::PMGChebyshev; });
   bad([](CahouetChabardConfig & c)
   { c.pc_precision = incns::PcPrecision::FP32PC; });
   bad([](CahouetChabardConfig & c) { c.n_inner = 0; });
   bad([](CahouetChabardConfig & c)
   { c.inner_stop = incns::InnerStop::RelTol; c.tol_inner = 2.0; });
   bad([](CahouetChabardConfig & c) { c.lp_vcycles = 0; });

   // The default config (with a legal sigma) validates cleanly.
   CahouetChabardConfig ok;
   ok.sigma = 1.0;
   EXPECT_NO_THROW(ok.Validate(false));
}

TEST(CahouetChabard, DgPressureRejected)
{
   SettingCase sc;
   L2_FECollection l2(2, 2);
   ParFiniteElementSpace l2fes(&sc.mesh, &l2);
   CahouetChabardConfig c;
   c.sigma = 1.0;
   EXPECT_THROW(CahouetChabardSchurPC(c, sc.spaces->Velocity(), l2fes,
                                      sc.rules, sc.op->Divergence(),
                                      sc.op->Mass(), sc.op->MassDiagonal(),
                                      sc.outflow_attrs, true),
                std::invalid_argument);
}

TEST(CahouetChabard, SigmaZeroIsPureMass)
{
   SettingCase sc;
   CahouetChabardConfig c;
   c.sigma = 0.0;
   c.nu = 0.7;
   auto pc = sc.MakePC(c);
   const int np = sc.spaces->Pressure().GetTrueVSize();

   Vector r(np), z(np), z_ref(np);
   r.Randomize(3);
   pc->Mult(r, z);

   // Independent reference: nu * M_p^-1 r with the same (deterministic)
   // Chebyshev construction.
   ParBilinearForm mp(&sc.spaces->Pressure());
   auto* mi = new MassIntegrator;
   mi->SetIntRule(&sc.rules.Get(Geometry::SQUARE, 4));
   mp.AddDomainIntegrator(mi);
   mp.SetAssemblyLevel(AssemblyLevel::PARTIAL);
   mp.Assemble();
   Array<int> empty;
   OperatorPtr Mp;
   mp.FormSystemMatrix(empty, Mp);
   Vector diag(np);
   mp.AssembleDiagonal(diag);
   MassInverse mp_inv(*Mp.Ptr(), diag, MPI_COMM_WORLD, MassInvType::Auto, 3);
   mp_inv.Mult(r, z_ref);
   z_ref *= 0.7;

   z_ref -= z;
   EXPECT_LE(Norm(z_ref), 1e-13 * Norm(z));
}

TEST(CahouetChabard, ScalingGuardSigmaOnPoissonTerm)
{
   SettingCase sc;
   const double sigma = 150.0, nu = 0.7;
   CahouetChabardConfig c;
   c.sigma = sigma;
   c.nu = nu;
   c.n_inner = 60; // tight inner solve so w ~ S^-1 r-hat
   auto pc = sc.MakePC(c);
   ASSERT_TRUE(pc->Singular()); // enclosed

   const int np = sc.spaces->Pressure().GetTrueVSize();
   ConstantPressureProjector proj(sc.spaces->Pressure());
   Vector r(np);
   r.Randomize(5);
   proj.Project(r); // keep the algebra clean: r in the range of S

   Vector z(np), t(np), w(np);
   pc->Mult(r, z);

   // Extract w = (z - nu * Mp^-1 r) / sigma and verify S w ~ r.
   CahouetChabardConfig c0 = c;
   c0.sigma = 0.0;
   auto pc0 = sc.MakePC(c0);
   pc0->Mult(r, t); // = nu * Mp^-1 r
   w = z;
   w -= t;
   w *= 1.0 / sigma;

   MassInverse mv_inv(sc.op->Mass(), sc.op->MassDiagonal(), MPI_COMM_WORLD,
                      MassInvType::Auto);
   MixedPoissonOperator S(sc.op->Divergence(), mv_inv);
   Vector Sw(np);
   S.Mult(w, Sw);
   Sw -= r;
   proj.Project(Sw);
   const double rel = Norm(Sw) / Norm(r);
   if (Mpi::Root())
   {
      mfem::out << "[cc] scaling guard ||S w - r||/||r|| = " << rel << std::endl;
   }
   EXPECT_LE(rel, 1e-5); // wildly violated if sigma/nu were swapped
}

TEST(CahouetChabard, ResetUpdatesOnlyScalars)
{
   SettingCase sc;
   CahouetChabardConfig c;
   c.sigma = 100.0;
   c.nu = 1.0;
   auto pc = sc.MakePC(c);
   const int np = sc.spaces->Pressure().GetTrueVSize();

   Vector r(np), z1(np), z2(np), z3(np);
   r.Randomize(7);
   pc->Mult(r, z1);

   // Reset to steady (sigma = 0): pure mass path.
   pc->Reset(0.0, 1.0);
   pc->Mult(r, z2);
   CahouetChabardConfig c0 = c;
   c0.sigma = 0.0;
   auto pc0 = sc.MakePC(c0);
   Vector z_ref(np);
   pc0->Mult(r, z_ref);
   z_ref -= z2;
   EXPECT_LE(Norm(z_ref), 1e-14 * Norm(z2));

   // Reset back: reproduces the original application (deterministic reuse).
   pc->Reset(100.0, 1.0);
   pc->Mult(r, z3);
   z3 -= z1;
   EXPECT_LE(Norm(z3), 1e-14 * Norm(z1));

   EXPECT_THROW(pc->Reset(-1.0, 1.0), std::invalid_argument);
   EXPECT_THROW(pc->Reset(1.0, 0.0), std::invalid_argument);
}

TEST(CahouetChabard, DriftWithoutProjection)
{
   SettingCase sc;
   const int np = sc.spaces->Pressure().GetTrueVSize();
   ConstantPressureProjector proj(sc.spaces->Pressure());
   Vector r(np);
   r.Randomize(9); // deliberately NOT projected

   CahouetChabardConfig c;
   c.sigma = 100.0;
   c.nu = 1.0;
   c.n_inner = 40;

   // Auto (singular path, all projections): the Poisson term stays clean.
   auto pc_auto = sc.MakePC(c);
   ASSERT_TRUE(pc_auto->Singular());
   Vector z_auto(np), t(np);
   pc_auto->Mult(r, z_auto);
   CahouetChabardConfig c0 = c;
   c0.sigma = 0.0;
   sc.MakePC(c0)->Mult(r, t);
   Vector w_auto = z_auto;
   w_auto -= t; // = sigma * w
   const double c_rel_auto =
      std::abs(proj.ConstantComponent(w_auto)) / (Norm(w_auto) + 1e-300);
   EXPECT_LE(c_rel_auto, 1e-10);

   // Health metric: does the extracted w actually solve S w = P(r)? (The
   // consistent part of the singular system -- what the Poisson term is FOR.)
   MassInverse mv_inv(sc.op->Mass(), sc.op->MassDiagonal(), MPI_COMM_WORLD,
                      MassInvType::Auto);
   MixedPoissonOperator S(sc.op->Divergence(), mv_inv);
   Vector r_hat = r;
   proj.Project(r_hat);
   auto poisson_residual = [&](Vector & sw, double sigma_val)
   {
      Vector wloc = sw;
      wloc *= 1.0 / sigma_val;
      Vector res(np);
      S.Mult(wloc, res);
      res -= r_hat;
      proj.Project(res);
      return Norm(res) / Norm(r_hat);
   };
   const double res_auto = poisson_residual(w_auto, c.sigma);
   EXPECT_LE(res_auto, 1e-2); // n_inner = 40: a genuine approximate solve

   // ForceOff (projections disabled on a singular domain): the inner stack
   // breaks -- in practice hypre's direct coarse solve on the raw singular
   // Neumann L_p produces garbage and the CG collapses (w = 0), or the
   // constant mode pollutes the iterate. Either way the Poisson term FAILS
   // to solve the consistent system -- that is what the projections buy.
   CahouetChabardConfig coff = c;
   coff.nullspace = NullspaceMode::ForceOff;
   auto pc_off = sc.MakePC(coff);
   ASSERT_FALSE(pc_off->Singular());
   Vector z_off(np);
   pc_off->Mult(r, z_off);
   Vector w_off = z_off;
   w_off -= t;
   const double res_off = std::isfinite(Norm(w_off))
                          ? poisson_residual(w_off, c.sigma)
                          : 1e30;
   if (Mpi::Root())
   {
      mfem::out << "[cc] drift: poisson residual auto=" << res_auto
                << "  forceoff=" << res_off << "  (const fraction auto="
                << c_rel_auto << ")" << std::endl;
   }
   const bool off_healthy = std::isfinite(res_off) && res_off < 1e-2;
   EXPECT_FALSE(off_healthy)
         << "disabling projection should break the singular Poisson solve";
}

TEST(CahouetChabard, LaplacianLegacyDiffers)
{
   SettingCase sc;
   CahouetChabardConfig c;
   c.sigma = 100.0;
   c.nu = 1.0;
   auto pc_cons = sc.MakePC(c);
   CahouetChabardConfig cl = c;
   cl.schur_model = SchurModel::LaplacianLegacy;
   auto pc_leg = sc.MakePC(cl);

   const int np = sc.spaces->Pressure().GetTrueVSize();
   Vector r(np), zc(np), zl(np);
   r.Randomize(11);
   pc_cons->Mult(r, zc);
   pc_leg->Mult(r, zl);
   zl -= zc;
   EXPECT_GT(Norm(zl), 1e-3 * Norm(zc)); // genuinely different operators
}

// pc_quadrature = GllCollocated: the PC builds its OWN diagonal masses; the
// scaling-guard algebra must hold against the COLLOCATED reference operator
// S~ = B M~^-1 B^T (not the consistent one), and contradictory mass-inverse
// requests throw. The system operator is untouched by construction.
TEST(CahouetChabard, GllCollocatedPcQuadrature)
{
   SettingCase sc;
   const double sigma = 150.0;
   CahouetChabardConfig c;
   c.sigma = sigma;
   c.nu = 0.7;
   c.n_inner = 60;
   c.pc_quadrature = incns::PcQuadrature::GllCollocated;
   auto pc = sc.MakePC(c);

   const int np = sc.spaces->Pressure().GetTrueVSize();
   ConstantPressureProjector proj(sc.spaces->Pressure());
   Vector r(np);
   r.Randomize(5);
   proj.Project(r);

   Vector z(np), t(np), w(np);
   pc->Mult(r, z);
   CahouetChabardConfig c0 = c;
   c0.sigma = 0.0;
   sc.MakePC(c0)->Mult(r, t);
   w = z;
   w -= t;
   w *= 1.0 / sigma;

   // Collocated reference S~: BM~^-1B^T with the PC's own GLL-diagonal mass.
   ParBilinearForm mv(&sc.spaces->Velocity());
   auto* mvi = new VectorMassIntegrator;
   mvi->SetIntRule(&sc.rules.CollocatedMass(Geometry::SQUARE, 3));
   mv.AddDomainIntegrator(mvi);
   mv.SetAssemblyLevel(AssemblyLevel::PARTIAL);
   mv.Assemble();
   Array<int> empty;
   OperatorPtr Mv;
   mv.FormSystemMatrix(empty, Mv);
   Vector diag(sc.spaces->Velocity().GetTrueVSize());
   mv.AssembleDiagonal(diag);
   MassInverse mv_inv(*Mv.Ptr(), diag, MPI_COMM_WORLD, MassInvType::DiagDirect);
   // NOTE: the reference must use the ELIMINATED B, like the PC does.
   MixedPoissonOperator S(sc.op->Divergence(), mv_inv);
   Vector Sw(np);
   S.Mult(w, Sw);
   Sw -= r;
   proj.Project(Sw);
   EXPECT_LE(Norm(Sw), 1e-5 * Norm(r));

   // Contradictory request: collocated PC masses + Chebyshev inverse.
   CahouetChabardConfig bad = c;
   bad.mv_inv = incns::MassInvType::Chebyshev;
   EXPECT_THROW(bad.Validate(false), std::invalid_argument);
}
