// TrueDofExchange (operators/true_dof_exchange.hpp): P and P^T of a conforming
// parallel space without the L-vector, against MFEM's prolongation:
//  X1 the gather: owned dofs read from the true-dof vector and ghosts from
//     Ghosts() are exactly P x;
//  X2 the reduce: owned dofs written into the true-dof vector, the ghosts'
//     contributions summed by their owners, give P^T y (to roundoff: another
//     summation order), and every true dof is written;
//  X3 two reduces agree bitwise.
// 3D hexes, vector (vdim 3, by nodes) and scalar spaces, orders 1 to 3. Run at
// np 1, 2 and 4 on the CPU (np 4 has dofs that three or more ranks share) and
// at np 1 and 2 on GPUs (INCNS_DEVICE=cuda).

#include <gtest/gtest.h>

#include "operators/true_dof_exchange.hpp"
#include "mfem.hpp"

#include <cmath>
#include <limits>
#include <memory>
#include <vector>

using namespace mfem;
using incns::TrueDofExchange;

namespace
{

struct Space
{
   Space(int vdim, int order)
   {
      Mesh serial = Mesh::MakeCartesian3D(4, 3, 3, Element::HEXAHEDRON, 1.0, 0.8,
                                          0.7);
      mesh = std::make_unique<ParMesh>(MPI_COMM_WORLD, serial);
      fec = std::make_unique<H1_FECollection>(order, 3);
      fes = std::make_unique<ParFiniteElementSpace>(mesh.get(), fec.get(), vdim,
            Ordering::byNODES);
   }
   std::unique_ptr<ParMesh> mesh;
   std::unique_ptr<H1_FECollection> fec;
   std::unique_ptr<ParFiniteElementSpace> fes;
};

Vector Random(int n, int seed)
{
   Vector v(n);
   v.Randomize(seed + 17 * Mpi::WorldRank());
   v -= 0.5;
   v.UseDevice(true);
   return v;
}

const int kCases[][2] = {{3, 3}, {1, 2}, {3, 1}}; // vdim, order

/// The exchange's reduce of the L-vector @p yl into y (true dofs): owned
/// values written, ghosts' contributions sent to their owners.
void Reduce(const TrueDofExchange& ex, const ParFiniteElementSpace& fes,
            const Vector& yl, Vector& y)
{
   const int nd = fes.GetNDofs(), vd = ex.VDim(), ntd = ex.TrueStride();
   y.SetSize(fes.GetTrueVSize());
   y.UseDevice(true);
   y = std::numeric_limits<double>::quiet_NaN();
   const int* s = ex.LocalSource().HostRead();
   const real_t* l = yl.HostRead();
   real_t* h = y.HostReadWrite();
   real_t* g = ex.GhostSums().HostReadWrite();
   for (int j = 0; j < nd; j++)
   {
      for (int c = 0; c < vd; c++)
      {
         if (s[j] >= 0) { h[s[j] + ntd * c] = l[j + nd * c]; }
         else { g[vd * (-1 - s[j]) + c] = l[j + nd * c]; }
      }
   }
   ex.ReduceBegin();
   ex.ReduceEnd(y);
}

} // namespace

TEST(TrueDofExchange, X1_GatherIsTheProlongation)
{
   for (const auto& cs : kCases)
   {
      Space sp(cs[0], cs[1]);
      ParFiniteElementSpace& fes = *sp.fes;
      TrueDofExchange ex(fes);
      const Vector x = Random(fes.GetTrueVSize(), 3);
      Vector l(fes.GetVSize());
      l.UseDevice(true);
      fes.GetProlongationMatrix()->Mult(x, l);
      ex.GatherBegin(x);
      ex.GatherEnd();
      const int nd = fes.GetNDofs(), vd = ex.VDim(), ntd = ex.TrueStride();
      const int* s = ex.LocalSource().HostRead();
      const real_t* xh = x.HostRead();
      const real_t* gh = ex.Ghosts().HostRead();
      const real_t* lh = l.HostRead();
      for (int j = 0; j < nd; j++)
      {
         for (int c = 0; c < vd; c++)
         {
            const real_t v = s[j] >= 0 ? xh[s[j] + ntd * c] : gh[vd * (-1 - s[j]) + c];
            ASSERT_EQ(v, lh[j + nd * c]) << "vdim " << vd << " order " << cs[1]
                                         << " ldof " << j << " component " << c;
         }
      }
   }
}

TEST(TrueDofExchange, X2_ReduceIsTheTransposedProlongation)
{
   for (const auto& cs : kCases)
   {
      Space sp(cs[0], cs[1]);
      ParFiniteElementSpace& fes = *sp.fes;
      TrueDofExchange ex(fes);
      const Vector yl = Random(fes.GetVSize(), 5);
      Vector ref(fes.GetTrueVSize()), y;
      ref.UseDevice(true);
      fes.GetProlongationMatrix()->MultTranspose(yl, ref);
      Reduce(ex, fes, yl, y);
      const real_t* r = ref.HostRead();
      const real_t* h = y.HostRead();
      for (int i = 0; i < ref.Size(); i++)
      {
         ASSERT_TRUE(std::isfinite(h[i])) << "true dof " << i << " not written";
         ASSERT_NEAR(h[i], r[i], 1e-14 * (1.0 + std::abs(r[i]))) << "vdim " << cs[0]
               << " order " << cs[1] << " true dof " << i;
      }
   }
}

TEST(TrueDofExchange, X3_ReduceIsDeterministic)
{
   Space sp(3, 3);
   ParFiniteElementSpace& fes = *sp.fes;
   TrueDofExchange ex(fes);
   const Vector yl = Random(fes.GetVSize(), 9);
   Vector y1, y2;
   Reduce(ex, fes, yl, y1);
   Reduce(ex, fes, yl, y2);
   const real_t* a = y1.HostRead();
   const real_t* b = y2.HostRead();
   for (int i = 0; i < y1.Size(); i++) { ASSERT_EQ(a[i], b[i]); }
}
