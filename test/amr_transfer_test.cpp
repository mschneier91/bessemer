// AMR.3 -- MeshAdapter: refine (and rebalance) in place, carrying fields
// across. Refinement nests the spaces, so the transfer must be EXACT.
//  X1 velocity and pressure fields keep ||.||_L2 (integrated exactly) through
//     two refinement passes, stay conforming (P R x = x), and the block
//     offsets / essential dofs are rebuilt -- 2D/3D, periodic box and walled
//     stretched box, isotropic and anisotropic;
//  X2 the same with rebalancing (np > 1), plus balanced element counts.

#include <gtest/gtest.h>

#include "amr/mesh_adapter.hpp"
#include "amr/refinement_marker.hpp"
#include "amr_test_util.hpp"
#include "bc/boundary_conditions.hpp"
#include "spaces/mixed_spaces.hpp"
#include "mfem.hpp"

#include <cmath>
#include <memory>

using namespace mfem;
using incns::BoundaryConditions;
using incns::MeshAdapter;
using incns::MixedSpaces;

namespace
{

double GMax(double v)
{
   MPI_Allreduce(MPI_IN_PLACE, &v, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
   return v;
}

// ||x_h||_L2 with a rule exact for |x_h|^2 on affine elements (both boxes are
// affine): refinement cannot change it unless the transfer changes x_h.
double ExactNorm(ParGridFunction& x, int order)
{
   const int vd = x.VectorDim();
   const IntegrationRule* irs[Geometry::NumGeom] = {nullptr};
   irs[Geometry::SQUARE] = &IntRules.Get(Geometry::SQUARE, 2 * order + 2);
   irs[Geometry::CUBE] = &IntRules.Get(Geometry::CUBE, 2 * order + 2);
   if (vd == 1)
   {
      ConstantCoefficient zero(0.0);
      return x.ComputeL2Error(zero, irs);
   }
   Vector z(vd);
   z = 0.0;
   VectorConstantCoefficient zero(z);
   return x.ComputeL2Error(zero, irs);
}

// max |P R x - x| / max |x|.
double ConformityDefect(ParGridFunction& x)
{
   ParGridFunction y(x.ParFESpace());
   Vector t;
   x.GetTrueDofs(t);
   y.SetFromTrueDofs(t);
   y -= x;
   return GMax(y.Normlinf()) / GMax(x.Normlinf());
}

void SetConforming(ParGridFunction& gf, Coefficient* c, VectorCoefficient* vc)
{
   if (c) { gf.ProjectCoefficient(*c); }
   else { gf.ProjectCoefficient(*vc); }
   Vector t;
   gf.GetTrueDofs(t);
   gf.SetFromTrueDofs(t);
}

void RunTransfer(int dim, bool periodic, bool aniso, bool rebalance)
{
   SCOPED_TRACE("dim=" + std::to_string(dim) + (periodic ? " periodic" :
                " walled+stretched") + (aniso ? " aniso" : " iso") +
                (rebalance ? " rebalance" : ""));
   const double L = periodic ? 2.0 * M_PI : 1.0;
   incns::BoxSpec spec = amr_test::Box(dim, dim == 2 ? 4 : 3, periodic, L);
   if (!periodic)
   {
      spec.stretch[1] = incns::Stretch::TwoSidedTanh;
      spec.stretch_beta[1] = 2.0;
   }
   auto mesh = amr_test::NcBox(spec);
   const int ku = (dim == 2) ? 3 : 2;
   MixedSpaces spaces(*mesh, ku, ku - 1);

   std::unique_ptr<BoundaryConditions> bc;
   if (!periodic)
   {
      bc = std::make_unique<BoundaryConditions>(spaces.Velocity());
      for (int a = 1; a <= 2 * dim; ++a) { bc->AddNoSlip(a); }
   }

   ParGridFunction u(&spaces.Velocity()), p(&spaces.Pressure());
   VectorFunctionCoefficient uc(dim, [](const Vector & x, Vector & v)
   {
      const double X = x(0), Y = x(1), Z = (x.Size() == 3) ? x(2) : 0.0;
      v(0) = std::sin(X) * std::cos(Y) + 0.3 * std::cos(Z);
      v(1) = std::cos(2.0 * X) * std::sin(Y);
      if (v.Size() == 3) { v(2) = std::sin(X + Z) * std::cos(Y); }
   });
   FunctionCoefficient pc([](const Vector & x)
   {
      return std::cos(x(0)) * std::sin(2.0 * x(1)) + 0.1;
   });
   SetConforming(u, nullptr, &uc);
   SetConforming(p, &pc, nullptr);
   const double nu0 = ExactNorm(u, ku), np0 = ExactNorm(p, ku - 1);
   const long long ne0 = mesh->GetGlobalNE();

   // Pass 1: x band; pass 2: y band (X / Y splits, or isotropic).
   const char iso = static_cast<char>(dim == 3 ? Refinement::XYZ : Refinement::XY);
   for (int pass = 0; pass < 2; ++pass)
   {
      Array<Refinement> refs;
      Vector c(dim);
      for (int e = 0; e < mesh->GetNE(); ++e)
      {
         mesh->GetElementCenter(e, c);
         if (c(pass) < 0.35 * L || c(pass) > 0.85 * L)
         {
            refs.Append(Refinement(e, aniso ? static_cast<char>(1 << pass) : iso));
         }
      }
      incns::RefinementMarker::ResolveAnisotropicConflicts(*mesh, refs);
      MeshAdapter::Refine(*mesh, spaces, bc.get(), refs, 1, rebalance, {&u, &p});
   }
   EXPECT_GT(mesh->GetGlobalNE(), ne0);

   // Exact transfer; conforming fields.
   EXPECT_NEAR(ExactNorm(u, ku), nu0, 1e-12 * nu0);
   EXPECT_NEAR(ExactNorm(p, ku - 1), np0, 1e-12 * np0);
   EXPECT_LE(ConformityDefect(u), 1e-13);
   EXPECT_LE(ConformityDefect(p), 1e-13);

   // Rebuilt bookkeeping.
   const Array<int>& off = spaces.BlockTrueOffsets();
   EXPECT_EQ(off[1], spaces.Velocity().GetTrueVSize());
   EXPECT_EQ(off[2] - off[1], spaces.Pressure().GetTrueVSize());
   if (bc)
   {
      Array<int> marker(mesh->bdr_attributes.Max()), fresh;
      marker = 1;
      spaces.Velocity().GetEssentialTrueDofs(marker, fresh);
      const Array<int>& ess = bc->EssentialTrueDofs();
      ASSERT_EQ(ess.Size(), fresh.Size());
      for (int i = 0; i < ess.Size(); ++i) { EXPECT_EQ(ess[i], fresh[i]); }
   }

   if (rebalance && mesh->GetNRanks() > 1)
   {
      long long ne = mesh->GetNE(), mx = ne, mn = ne;
      MPI_Allreduce(MPI_IN_PLACE, &mx, 1, MPI_LONG_LONG, MPI_MAX, MPI_COMM_WORLD);
      MPI_Allreduce(MPI_IN_PLACE, &mn, 1, MPI_LONG_LONG, MPI_MIN, MPI_COMM_WORLD);
      EXPECT_LE(mx - mn, 2) << "elements per rank after rebalancing: " << mn
                            << ".." << mx;
   }
}

} // namespace

TEST(AmrTransfer, X1_ExactTransfer)
{
   for (int dim : {2, 3})
      for (bool periodic : {true, false})
         for (bool aniso : {false, true})
         {
            RunTransfer(dim, periodic, aniso, false);
         }
}

TEST(AmrTransfer, X2_ExactTransferWithRebalance)
{
   for (int dim : {2, 3})
      for (bool periodic : {true, false})
         for (bool aniso : {false, true})
         {
            RunTransfer(dim, periodic, aniso, true);
         }
}
