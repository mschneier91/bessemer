// AMR.3 -- MeshAdapter: refine (and rebalance) in place, carrying fields
// across. Refinement nests the spaces, so the transfer must be EXACT.
//  X1 velocity and pressure fields keep ||.||_L2 (integrated exactly) through
//     two refinement passes, stay conforming (P R x = x), and the block
//     offsets / essential dofs are rebuilt -- 2D/3D, periodic box and walled
//     stretched box, isotropic and anisotropic;
//  X2 the same with rebalancing (np > 1), plus balanced element counts;
//  X3 coarsening (MeshAdapter::Derefine) replaces a velocity by its L2
//     projection: the integral of each component (walled: and of its
//     products with the coordinates) is kept and the L2 norm does not grow,
//     which MFEM's interpolation does not do; the results stay conforming --
//     2D/3D (3D isotropic: MFEM's limit), periodic and walled, rebalanced or
//     not;
//  X4 refining a velocity the mesh holds, then merging every family until
//     none is left, gives back the original mesh and the same velocity.

#include <gtest/gtest.h>

#include "amr/mesh_adapter.hpp"
#include "amr/refinement_marker.hpp"
#include "amr_test_util.hpp"
#include "bc/boundary_conditions.hpp"
#include "spaces/mixed_spaces.hpp"
#include "mfem.hpp"

#include <algorithm>
#include <cmath>
#include <memory>
#include <vector>

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

// X3/X4's velocity.
void Trial(const Vector& x, Vector& v)
{
   const double X = x(0), Y = x(1), Z = (x.Size() == 3) ? x(2) : 0.0;
   v(0) = std::sin(X) * std::cos(Y) + 0.3 * std::cos(Z);
   v(1) = std::cos(2.0 * X) * std::sin(Y);
   if (v.Size() == 3) { v(2) = std::sin(X + Z) * std::cos(Y); }
}

// The integrals of each component of u and, with @p with_x, of its products
// with each coordinate (a rule exact for them on the affine boxes).
std::vector<double> Moments(ParGridFunction& u, int order, bool with_x)
{
   ParMesh& mesh = *u.ParFESpace()->GetParMesh();
   const int dim = mesh.Dimension(), vd = u.VectorDim();
   const int per = with_x ? dim + 1 : 1;
   std::vector<double> m(vd * per, 0.0);
   Vector val, x;
   u.HostRead();
   for (int e = 0; e < mesh.GetNE(); ++e)
   {
      ElementTransformation& T = *mesh.GetElementTransformation(e);
      const IntegrationRule& ir = IntRules.Get(T.GetGeometryType(),
                                  2 * order + 2);
      for (int q = 0; q < ir.GetNPoints(); ++q)
      {
         const IntegrationPoint& ip = ir.IntPoint(q);
         T.SetIntPoint(&ip);
         u.GetVectorValue(T, ip, val);
         T.Transform(ip, x);
         const double w = ip.weight * T.Weight();
         for (int c = 0; c < vd; ++c)
         {
            m[c * per] += w * val(c);
            for (int d = 0; with_x && d < dim; ++d)
            {
               m[c * per + 1 + d] += w * x(d) * val(c);
            }
         }
      }
   }
   MPI_Allreduce(MPI_IN_PLACE, m.data(), static_cast<int>(m.size()),
                 MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
   return m;
}

double MaxDev(const std::vector<double>& a, const std::vector<double>& b)
{
   double d = 0.0;
   for (std::size_t i = 0; i < a.size(); ++i)
   {
      d = std::max(d, std::abs(a[i] - b[i]));
   }
   return d;
}

// X3: merge every family in x < L/2 (where RefineBands refined).
void RunDerefine(int dim, bool periodic, bool aniso, bool rebalance)
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
   amr_test::RefineBands(*mesh, aniso, L);
   const int ku = (dim == 2) ? 3 : 2;
   MixedSpaces spaces(*mesh, ku, ku - 1);
   std::unique_ptr<BoundaryConditions> bc;
   if (!periodic)
   {
      bc = std::make_unique<BoundaryConditions>(spaces.Velocity());
      for (int a = 1; a <= 2 * dim; ++a) { bc->AddNoSlip(a); }
   }
   ParGridFunction u(&spaces.Velocity()), v(&spaces.Velocity()),
                   p(&spaces.Pressure());
   VectorFunctionCoefficient uc(dim, Trial);
   FunctionCoefficient pc([](const Vector & x)
   {
      return std::cos(x(0)) * std::sin(2.0 * x(1)) + 0.1;
   });
   SetConforming(u, nullptr, &uc);
   v = u;
   SetConforming(p, &pc, nullptr);
   const std::vector<double> m0 = Moments(u, ku, !periodic);
   const double n0 = ExactNorm(u, ku);
   const long long ne0 = mesh->GetGlobalNE();

   Vector values(mesh->GetNE()), c(dim);
   for (int e = 0; e < mesh->GetNE(); ++e)
   {
      mesh->GetElementCenter(e, c);
      values(e) = (c(0) < 0.5 * L) ? 0.0 : 1.0;
   }
   Array<int> gather, merge;
   const long long families =
      MeshAdapter::SelectDerefinements(*mesh, values, 0.5, 1, gather, merge);
   EXPECT_GT(families, 0);
   EXPECT_TRUE(MeshAdapter::Derefine(*mesh, spaces, bc.get(), gather, merge, 1,
                                     rebalance, {&v, &p}, {&u}));
   EXPECT_LT(mesh->GetGlobalNE(), ne0);

   // |int u_c| <= ||u|| |Omega|^(1/2), |int x_d u_c| <= L times that
   // (Cauchy-Schwarz): the periodic trial field integrates to zero, so the
   // moments are not measured against themselves.
   const double scale = n0 * std::pow(L, 0.5 * dim) * std::max(1.0, L);
   const double dev_u = MaxDev(Moments(u, ku, !periodic), m0);
   const double dev_v = MaxDev(Moments(v, ku, !periodic), m0);
   const double n1 = ExactNorm(u, ku);
   if (Mpi::Root())
   {
      mfem::out << "[x3] " << families << " families merged; moments off by "
                << dev_u / scale << " (L2 projection), " << dev_v / scale
                << " (interpolation); L2 norm ratio " << n1 / n0 << std::endl;
   }
   EXPECT_LE(dev_u, 1e-10 * scale);
   EXPECT_LE(n1, n0 * (1.0 + 1e-12));
   EXPECT_LE(ConformityDefect(u), 1e-13);
   EXPECT_LE(ConformityDefect(p), 1e-13);
   if (bc)
   {
      Array<int> marker(mesh->bdr_attributes.Max()), fresh;
      marker = 1;
      spaces.Velocity().GetEssentialTrueDofs(marker, fresh);
      const Array<int>& ess = bc->EssentialTrueDofs();
      ASSERT_EQ(ess.Size(), fresh.Size());
      for (int i = 0; i < ess.Size(); ++i) { EXPECT_EQ(ess[i], fresh[i]); }
   }
}

// X4: refine a velocity the original mesh holds (exactly), then merge every
// family until none is left.
void RunUndo(int dim, bool periodic, bool aniso, bool rebalance)
{
   SCOPED_TRACE("dim=" + std::to_string(dim) + (periodic ? " periodic" :
                " walled") + (aniso ? " aniso" : " iso") +
                (rebalance ? " rebalance" : ""));
   const double L = periodic ? 2.0 * M_PI : 1.0;
   auto mesh = amr_test::NcBox(amr_test::Box(dim, dim == 2 ? 4 : 3, periodic,
                               L));
   const int ku = (dim == 2) ? 3 : 2;
   MixedSpaces spaces(*mesh, ku, ku - 1);
   ParGridFunction u(&spaces.Velocity());
   VectorFunctionCoefficient uc(dim, Trial);
   SetConforming(u, nullptr, &uc);
   const long long ne0 = mesh->GetGlobalNE();
   const char iso = static_cast<char>(dim == 3 ? Refinement::XYZ : Refinement::XY);
   for (int pass = 0; pass < 2; ++pass)
   {
      Array<Refinement> refs;
      Vector c(dim);
      for (int e = 0; e < mesh->GetNE(); ++e)
      {
         mesh->GetElementCenter(e, c);
         if (c(pass) < 0.5 * L)
         {
            refs.Append(Refinement(e, aniso ? static_cast<char>(1 << pass) : iso));
         }
      }
      MeshAdapter::Refine(*mesh, spaces, nullptr, refs, 1, rebalance, {&u});
   }
   EXPECT_GT(mesh->GetGlobalNE(), ne0);
   int rounds = 0;
   for (; rounds < 10; ++rounds)
   {
      Vector values(mesh->GetNE());
      values = 0.0;
      Array<int> gather, merge;
      if (MeshAdapter::SelectDerefinements(*mesh, values, 0.5, 1, gather,
                                           merge) == 0) { break; }
      MeshAdapter::Derefine(*mesh, spaces, nullptr, gather, merge, 1, rebalance,
                            {}, {&u});
   }
   EXPECT_EQ(mesh->GetGlobalNE(), ne0) << "after " << rounds << " rounds";
   ParGridFunction w(&spaces.Velocity());
   SetConforming(w, nullptr, &uc);
   w -= u;
   // To the mass solves' tolerance (rtol 1e-12, a few rounds).
   EXPECT_LE(GMax(w.Normlinf()), 1e-10 * GMax(u.Normlinf()));
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

TEST(AmrTransfer, X3_DerefinementIsTheL2Projection)
{
   for (int dim : {2, 3})
      for (bool periodic : {true, false})
         for (bool aniso : {false, true})
            for (bool rebalance : {false, true})
            {
               if (dim == 3 && aniso) { continue; } // MFEM: no 3D anisotropic
               RunDerefine(dim, periodic, aniso, rebalance);
            }
}

TEST(AmrTransfer, X4_RefineThenMergeEverythingIsExact)
{
   for (int dim : {2, 3})
      for (bool periodic : {true, false})
         for (bool aniso : {false, true})
            for (bool rebalance : {false, true})
            {
               if (dim == 3 && aniso) { continue; }
               RunUndo(dim, periodic, aniso, rebalance);
            }
}
