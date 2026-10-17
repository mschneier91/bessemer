// The collocated mass lumped at hanging nodes, b = P^T d
// (operators/collocated_mass: AssembleCollocatedMassDiagonal, LumpedMassRatio).
// On affine elements with one hanging level it equals the consistent mass's
// row sums -- the integrals of the constrained basis functions -- at every
// order (the positivity analysis, 2026-10-17). Under the refinements the
// solver allows (amr.nc_limit 1: isotropic, anisotropic toward a wall, random
// anisotropic; distorted and curved elements) its ratio to the conforming
// weight, b / ref, stays above kMinLumpedMassRatio (at least 1 where it is
// proved). The ratios are printed.

#include <gtest/gtest.h>

#include "operators/collocated_mass.hpp"
#include "quadrature/rule_book.hpp"
#include "mfem.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <functional>
#include <memory>

using namespace mfem;

namespace
{

// A fixed pseudo-random choice per element (rank-independent per index).
bool Pick(int e, int salt, int percent)
{
   std::uint32_t h = static_cast<std::uint32_t>(e) * 2654435761u +
                     static_cast<std::uint32_t>(salt) * 40503u;
   h ^= h >> 13;
   h *= 0x5bd1e995u;
   h ^= h >> 15;
   return static_cast<int>(h % 100u) < percent;
}

// The box, its map, then the refinements on the serial mesh (MFEM's parallel
// anisotropic refinement cannot force neighbour refinements), partitioned.
std::unique_ptr<ParMesh> Box(int dim, int n, int curvature,
                             const std::function<void(const Vector&, Vector&)>& map,
                             const std::function<void(Mesh&)>& refine)
{
   Mesh serial = (dim == 3)
                 ? Mesh::MakeCartesian3D(n, n, n, Element::HEXAHEDRON)
                 : Mesh::MakeCartesian2D(n + 1, n, Element::QUADRILATERAL, true);
   if (curvature > 0) { serial.SetCurvature(curvature); }
   if (map) { serial.Transform(map); }
   serial.EnsureNCMesh();
   refine(serial);
   return std::make_unique<ParMesh>(MPI_COMM_WORLD, serial);
}

// b, and the smallest ratio b / ref, on V (order k, GLL, vdim = dim).
struct Lump
{
   Vector b;
   double ratio = 0.0;
};

Lump Lumped(ParFiniteElementSpace& V, int k)
{
   incns::RuleBook rules;
   const Geometry::Type g =
      (V.GetMesh()->Dimension() == 3) ? Geometry::CUBE : Geometry::SQUARE;
   const IntegrationRule& ir = rules.CollocatedMass(g, k);
   Lump l;
   incns::AssembleCollocatedMassDiagonal(V, ir, l.b);
   l.ratio = incns::LumpedMassRatio(V, ir, l.b);
   return l;
}

void Report(const char* what, int dim, int k, double ratio)
{
   if (Mpi::Root())
   {
      mfem::out << "[hanging mass] " << what << ", dim " << dim << ", order " << k
                << ": min b / ref = " << ratio << std::endl;
   }
}

} // namespace

// Affine elements, one hanging level (isotropic, nc_limit 1): b equals the
// consistent mass's row sums (the vector linear form of 1, exactly
// integrated), to roundoff, and is at least the conforming weight.
TEST(HangingMass, AffineSingleLevelEqualsTheRowSums)
{
   for (const int dim : {2, 3})
   {
      for (const int k : {2, 4, 6, 8})
      {
         if (dim == 3 && k == 8) { continue; } // the test's cost
         SCOPED_TRACE(testing::Message() << "dim " << dim << ", order " << k);
         std::unique_ptr<ParMesh> pm = Box(dim, dim == 3 ? 3 : 4, 0, nullptr,
                                           [](Mesh & m)
         {
            Array<Refinement> refs;
            for (int e = 0; e < m.GetNE(); ++e)
            {
               if (Pick(e, 1, 35)) { refs.Append(Refinement(e)); }
            }
            m.GeneralRefinement(refs, 1, 1);
         });
         ParMesh& mesh = *pm;
         ASSERT_FALSE(mesh.Conforming());
         H1_FECollection fec(k, dim, BasisType::GaussLobatto);
         ParFiniteElementSpace V(&mesh, &fec, dim);
         const Lump l = Lumped(V, k);
         Report("affine, one level", dim, k, l.ratio);
         EXPECT_GE(l.ratio, 1.0 - 1e-12); // b >= ref: proved here

         ParLinearForm f(&V);
         Vector one(dim);
         one = 1.0;
         VectorConstantCoefficient c(one);
         auto* integ = new VectorDomainLFIntegrator(c);
         const Geometry::Type g = (dim == 3) ? Geometry::CUBE : Geometry::SQUARE;
         integ->SetIntRule(&IntRules.Get(g, 2 * k + 2));
         f.AddDomainIntegrator(integ);
         f.Assemble();
         Vector rows(V.GetTrueVSize());
         f.ParallelAssemble(rows);
         Vector d(l.b);
         d -= rows;
         double diff = d.Normlinf(), size = rows.Normlinf();
         MPI_Allreduce(MPI_IN_PLACE, &diff, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
         MPI_Allreduce(MPI_IN_PLACE, &size, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
         EXPECT_GT(size, 0.0);
         EXPECT_LE(diff, 1e-12 * size);
      }
   }
}

// Anisotropic refinement toward a wall (y = 0), several rounds, and a
// neighbour split only tangentially: the boundary-layer pattern. With
// nc_limit 1 (what the solver enforces) the lump stays well above the
// threshold; the deep nesting nc_limit 8 allows is printed for the record.
TEST(HangingMass, AnisotropicTowardAWall)
{
   for (const int dim : {2, 3})
   {
      for (const int k : {2, 4, 6})
      {
         for (const int limit : {1, 8})
         {
            SCOPED_TRACE(testing::Message() << "dim " << dim << ", order " << k
                         << ", nc_limit " << limit);
            std::unique_ptr<ParMesh> pm = Box(dim, 3, 0, nullptr, [&](Mesh & m)
            {
               for (int round = 0; round < 4; ++round)
               {
                  // The elements touching y = 0, split in y only.
                  Array<Refinement> refs;
                  double ymin = 1e300;
                  Vector c(dim);
                  for (int e = 0; e < m.GetNE(); ++e)
                  {
                     m.GetElementCenter(e, c);
                     ymin = std::min(ymin, c(1));
                  }
                  for (int e = 0; e < m.GetNE(); ++e)
                  {
                     m.GetElementCenter(e, c);
                     if (c(1) <= ymin + 1e-12) { refs.Append(Refinement(e, Refinement::Y)); }
                  }
                  // And once, an element above them split in x only.
                  if (round == 0)
                  {
                     for (int e = 0; e < m.GetNE(); ++e)
                     {
                        m.GetElementCenter(e, c);
                        if (c(1) > 0.4 && c(1) < 0.6 && c(0) < 0.4)
                        {
                           refs.Append(Refinement(e, Refinement::X));
                           break;
                        }
                     }
                  }
                  m.GeneralRefinement(refs, 1, limit);
               }
            });
            ParMesh& mesh = *pm;
            H1_FECollection fec(k, dim, BasisType::GaussLobatto);
            ParFiniteElementSpace V(&mesh, &fec, dim);
            const Lump l = Lumped(V, k);
            Report(limit == 1 ? "toward a wall, nc_limit 1"
                   : "toward a wall, nc_limit 8", dim, k, l.ratio);
            if (limit == 1) { EXPECT_GE(l.ratio, incns::kMinLumpedMassRatio); }
         }
      }
   }
}

// Random anisotropic refinement, 3D, three rounds, nc_limit 1.
TEST(HangingMass, RandomAnisotropic3D)
{
   const char types[] = {Refinement::X, Refinement::Y, Refinement::Z,
                         Refinement::XY, Refinement::XZ, Refinement::YZ,
                         Refinement::XYZ
                        };
   for (const int k : {2, 4, 6})
   {
      SCOPED_TRACE(testing::Message() << "order " << k);
      std::unique_ptr<ParMesh> pm = Box(3, 3, 0, nullptr, [&](Mesh & m)
      {
         for (int round = 0; round < 3; ++round)
         {
            Array<Refinement> refs;
            for (int e = 0; e < m.GetNE(); ++e)
            {
               if (Pick(e, 10 + round, 30))
               {
                  refs.Append(Refinement(e, types[(e * 7 + round) % 7]));
               }
            }
            m.GeneralRefinement(refs, 1, 1);
         }
      });
      ParMesh& mesh = *pm;
      H1_FECollection fec(k, 3, BasisType::GaussLobatto);
      ParFiniteElementSpace V(&mesh, &fec, 3);
      const Lump l = Lumped(V, k);
      Report("random anisotropic, nc_limit 1", 3, k, l.ratio);
      EXPECT_GE(l.ratio, incns::kMinLumpedMassRatio);
   }
}

// Distorted straight-sided (trilinear) and curved elements, isotropic
// refinement, nc_limit 1.
TEST(HangingMass, DistortedAndCurved)
{
   auto distort = [](const Vector & x, Vector & y)
   {
      y = x;
      y(0) += 0.08 * std::sin(2.3 * x(1));
      y(1) += 0.06 * x(0) * x(0);
      if (x.Size() == 3) { y(2) += 0.05 * x(0) * x(1); }
   };
   for (const int dim : {2, 3})
   {
      for (const int k : {3, 6})
      {
         for (const bool curved : {false, true})
         {
            SCOPED_TRACE(testing::Message() << "dim " << dim << ", order " << k
                         << (curved ? ", curved" : ", trilinear"));
            std::unique_ptr<ParMesh> pm = Box(dim, 3, curved ? k : 0, distort,
                                              [](Mesh & m)
            {
               Array<Refinement> refs;
               for (int e = 0; e < m.GetNE(); ++e)
               {
                  if (Pick(e, 3, 40)) { refs.Append(Refinement(e)); }
               }
               m.GeneralRefinement(refs, 1, 1);
            });
            ParMesh& mesh = *pm;
            H1_FECollection fec(k, dim, BasisType::GaussLobatto);
            ParFiniteElementSpace V(&mesh, &fec, dim);
            const Lump l = Lumped(V, k);
            Report(curved ? "curved, nc_limit 1" : "trilinear, nc_limit 1", dim, k,
                   l.ratio);
            EXPECT_GE(l.ratio, incns::kMinLumpedMassRatio);
         }
      }
   }
}
