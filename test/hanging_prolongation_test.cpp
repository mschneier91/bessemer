// HangingProlongation (src/spaces): a nonconforming space's P as copies and
// weighted sums, at np 1/2/4.
//  HP1 Mult, MultTranspose, AbsMult and AbsMultTranspose equal MFEM's
//      assembled P's (2D and 3D, two refinement levels, refined after the
//      partition so that masters sit on other ranks; scalar Q1 and Q3, vector
//      Q2 ordered by nodes);
//  HP2 MixedSpaces hands out MFEM's prolongation on a conforming mesh and a
//      HangingProlongation on a nonconforming one, rebuilt after Update().

#include <gtest/gtest.h>

#include "spaces/hanging_prolongation.hpp"
#include "spaces/mixed_spaces.hpp"
#include "mfem.hpp"

#include <algorithm>
#include <cmath>
#include <memory>
#include <string>

using namespace mfem;
using incns::HangingProlongation;

namespace
{

/// The largest |a - b| over the largest |b|, across ranks.
double RelMax(const Vector& a, const Vector& b)
{
   const real_t* pa = a.HostRead();
   const real_t* pb = b.HostRead();
   double m[2] = {0.0, 0.0};
   for (int i = 0; i < a.Size(); ++i)
   {
      m[0] = std::max(m[0], std::abs(pa[i] - pb[i]));
      m[1] = std::max(m[1], std::abs(pb[i]));
   }
   MPI_Allreduce(MPI_IN_PLACE, m, 2, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
   return m[0] / m[1];
}

/// Refine the local elements whose centres pass @p pick (collective).
template <class F>
void RefineWhere(ParMesh& mesh, F&& pick)
{
   Array<int> elems;
   Vector c;
   for (int e = 0; e < mesh.GetNE(); ++e)
   {
      mesh.GetElementCenter(e, c);
      if (pick(c)) { elems.Append(e); }
   }
   mesh.GeneralRefinement(elems, 1, 0); // isotropic, no nc_limit
}

/// A nonconforming mesh with hanging nodes on rank interfaces: a corner
/// refined twice, after the partition.
std::unique_ptr<ParMesh> NcMesh(int dim)
{
   Mesh serial = dim == 2 ?
                 Mesh::MakeCartesian2D(4, 4, Element::QUADRILATERAL, true, 1.0, 1.0) :
                 Mesh::MakeCartesian3D(3, 3, 2, Element::HEXAHEDRON, 1.0, 1.0, 1.0);
   serial.EnsureNCMesh();
   auto mesh = std::make_unique<ParMesh>(MPI_COMM_WORLD, serial);
   RefineWhere(*mesh, [](const Vector & c) { return c(0) < 0.5 && c(1) < 0.6; });
   RefineWhere(*mesh, [](const Vector & c) { return c(0) < 0.3 && c(1) < 0.3; });
   return mesh;
}

} // namespace

TEST(HangingProlongation, HP1_MatchesMfemsAssembledP)
{
   struct Space { int order, vdim; };
   for (int dim : {2, 3})
   {
      auto mesh = NcMesh(dim);
      for (const Space sp : {Space{1, 1}, Space{3, 1}, Space{2, dim}})
      {
         H1_FECollection fec(sp.order, dim);
         ParFiniteElementSpace fes(mesh.get(), &fec, sp.vdim, Ordering::byNODES);
         ASSERT_FALSE(fes.Conforming());
         const HypreParMatrix& P = *fes.Dof_TrueDof_Matrix();
         HangingProlongation hp(fes);
         int nh = hp.NumHanging();
         MPI_Allreduce(MPI_IN_PLACE, &nh, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
         ASSERT_GT(nh, 0) << "no hanging dofs: the test would prove nothing";

         const int nl = fes.GetVSize(), nt = fes.GetTrueVSize();
         Vector x(nt), xl(nl), y1(nl), y2(nl), z1(nt), z2(nt);
         for (Vector* v : {&x, &xl, &y1, &y2, &z1, &z2}) { v->UseDevice(true); }
         x.Randomize(11 + Mpi::WorldRank());
         xl.Randomize(23 + Mpi::WorldRank());
         x -= 0.5; // signed: AbsMult must differ where weights are negative
         xl -= 0.5;
         const std::string what = "dim " + std::to_string(dim) + " order " +
                                  std::to_string(sp.order) + " vdim " +
                                  std::to_string(sp.vdim);

         hp.Mult(x, y1);
         P.Mult(x, y2);
         EXPECT_LT(RelMax(y1, y2), 1e-15) << "Mult, " << what;
         hp.MultTranspose(xl, z1);
         P.MultTranspose(xl, z2);
         EXPECT_LT(RelMax(z1, z2), 1e-14) << "MultTranspose, " << what;
         hp.AbsMult(x, y1);
         P.AbsMult(1.0, x, 0.0, y2);
         EXPECT_LT(RelMax(y1, y2), 1e-15) << "AbsMult, " << what;
         hp.AbsMultTranspose(xl, z1);
         P.AbsMultTranspose(1.0, xl, 0.0, z2);
         EXPECT_LT(RelMax(z1, z2), 1e-14) << "AbsMultTranspose, " << what;
      }
   }
}

TEST(HangingProlongation, HP2_MixedSpacesHandOutTheRightOne)
{
   {
      Mesh serial = Mesh::MakeCartesian2D(4, 4, Element::QUADRILATERAL, true, 1.0,
                                          1.0);
      ParMesh mesh(MPI_COMM_WORLD, serial);
      incns::MixedSpaces spaces(mesh, 3, 2);
      EXPECT_EQ(spaces.VelocityProlongation(),
                spaces.Velocity().GetProlongationMatrix());
      EXPECT_EQ(spaces.PressureProlongation(),
                spaces.Pressure().GetProlongationMatrix());
   }
   for (int dim : {2, 3})
   {
      auto mesh = NcMesh(dim);
      incns::MixedSpaces spaces(*mesh, 2, 1);
      const auto* vp = dynamic_cast<const HangingProlongation*>(
                          spaces.VelocityProlongation());
      const auto* pp = dynamic_cast<const HangingProlongation*>(
                          spaces.PressureProlongation());
      ASSERT_TRUE(vp && pp) << "dim " << dim;
      EXPECT_EQ(vp, spaces.VelocityProlongation()) << "built once";
      EXPECT_EQ(vp->Height(), spaces.Velocity().GetVSize());
      EXPECT_EQ(pp->Width(), spaces.Pressure().GetTrueVSize());

      // An AMR event: the next call builds the new spaces' P.
      RefineWhere(*mesh, [](const Vector & c) { return c(0) > 0.7; });
      spaces.Update();
      spaces.UpdatesFinished();
      const auto* vp2 = dynamic_cast<const HangingProlongation*>(
                           spaces.VelocityProlongation());
      ASSERT_TRUE(vp2);
      EXPECT_EQ(vp2->Height(), spaces.Velocity().GetVSize());
      EXPECT_EQ(vp2->Width(), spaces.Velocity().GetTrueVSize());

      const int nt = spaces.Velocity().GetTrueVSize(),
                nl = spaces.Velocity().GetVSize();
      Vector x(nt), y1(nl), y2(nl);
      for (Vector* v : {&x, &y1, &y2}) { v->UseDevice(true); }
      x.Randomize(5 + Mpi::WorldRank());
      vp2->Mult(x, y1);
      spaces.Velocity().Dof_TrueDof_Matrix()->Mult(x, y2);
      EXPECT_LT(RelMax(y1, y2), 1e-15) << "after Update(), dim " << dim;
   }
}
