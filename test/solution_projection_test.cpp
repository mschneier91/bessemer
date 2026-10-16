// incns::SolutionProjection (solver/solution_projection): the least-squares
// guess from earlier solutions, its images kept orthonormal (Gram-Schmidt,
// twice).
//  P1 a right-hand side in the span of the kept solutions' images gives back
//     the combination of the solutions (the guess is exact);
//  P2 a full basis restarts from the newest solution: the newest is still
//     reproduced, an older one no longer is;
//  P3 Rebuild with a changed operator: exact again for the new operator;
//  P4 the same sequence twice gives the same guess, bitwise;
//  P5 a basis restored from Solutions() and Images() (a checkpoint) guesses,
//     adds and rebuilds as the original does, bitwise.

#include <gtest/gtest.h>

#include "solver/solution_projection.hpp"
#include "mfem.hpp"

#include <algorithm>
#include <cmath>
#include <vector>

using namespace mfem;
using incns::SolutionProjection;

namespace
{

// y = d .* x on the device: an SPD stand-in for the Stokes block.
class DiagOp : public Operator
{
public:
   explicit DiagOp(const Vector& d) : Operator(d.Size()), d_(d) {}
   void Mult(const Vector& x, Vector& y) const override
   {
      y = x;
      y *= d_;
   }
private:
   const Vector& d_;
};

Vector Random(int n, int seed)
{
   Vector v(n);
   v.Randomize(seed);
   v -= 0.5;
   v.UseDevice(true);
   return v;
}

double Norm(const Vector& v) { return std::sqrt(InnerProduct(MPI_COMM_WORLD, v, v)); }

// || guess(A x) - x || / || x ||.
double GuessError(const SolutionProjection& p, const Operator& A,
                  const Vector& x)
{
   Vector b(x.Size()), g(x.Size());
   b.UseDevice(true);
   g.UseDevice(true);
   A.Mult(x, b);
   EXPECT_TRUE(p.Guess(b, g));
   g -= x;
   return Norm(g) / Norm(x);
}

} // namespace

TEST(SolutionProjection, GuessesFromTheKeptSolutions)
{
   int rank = 0;
   MPI_Comm_rank(MPI_COMM_WORLD, &rank);
   const int n = 20000 + 1013 * rank;
   Vector d = Random(n, 1 + rank);
   d += 1.5; // 1 .. 2: well conditioned
   DiagOp A(d);

   const int k = 6;
   std::vector<Vector> xs;
   for (int j = 0; j < k + 2; ++j) { xs.push_back(Random(n, 100 + 13 * j + 7 * rank)); }

   SolutionProjection p(k, MPI_COMM_WORLD);
   for (int j = 0; j < 5; ++j) { p.Add(A, xs[j]); }
   EXPECT_EQ(p.Size(), 5);

   // P1: a combination of the kept solutions comes back.
   Vector comb(n);
   comb.UseDevice(true);
   comb = 0.0;
   for (int j = 0; j < 5; ++j) { comb.Add(0.3 * (j + 1) * (j % 2 ? -1.0 : 1.0), xs[j]); }
   EXPECT_LT(GuessError(p, A, comb), 1e-10);
   EXPECT_LT(GuessError(p, A, xs[2]), 1e-10);

   // P3: a changed operator, images recomputed.
   Vector d2(d);
   d2 *= 1.3;
   d2 += 0.1;
   DiagOp A2(d2);
   p.Rebuild(A2);
   EXPECT_LT(GuessError(p, A2, comb), 1e-10);

   // P2: full at k, the next Add restarts from the newest.
   p.Add(A2, xs[5]);
   EXPECT_EQ(p.Size(), k);
   p.Add(A2, xs[6]);
   EXPECT_EQ(p.Size(), 1);
   EXPECT_LT(GuessError(p, A2, xs[6]), 1e-12);
   EXPECT_GT(GuessError(p, A2, xs[0]), 1e-3);

   // P4: the same sequence, the same bits.
   auto run = [&](Vector & g)
   {
      SolutionProjection q(k, MPI_COMM_WORLD);
      for (int j = 0; j < 5; ++j) { q.Add(A, xs[j]); }
      Vector b(n);
      b.UseDevice(true);
      A.Mult(comb, b);
      g.SetSize(n);
      g.UseDevice(true);
      q.Guess(b, g);
   };
   Vector g1, g2;
   run(g1);
   run(g2);
   const double* a = g1.HostRead();
   const double* c = g2.HostRead();
   for (int i = 0; i < n; ++i) { ASSERT_EQ(a[i], c[i]); }

   // P5: a restored copy of a basis (3 of k) follows the original, bitwise:
   // the guess, then one more solution, then a rebuild for A2.
   SolutionProjection orig(k, MPI_COMM_WORLD);
   for (int j = 0; j < 3; ++j) { orig.Add(A, xs[j]); }
   std::vector<Vector> sx(orig.Size()), sw(orig.Size());
   for (int i = 0; i < orig.Size(); ++i)
   {
      sx[i] = orig.Solutions()[i];
      sw[i] = orig.Images()[i];
      sx[i].UseDevice(true);
      sw[i].UseDevice(true);
   }
   SolutionProjection copy(k, MPI_COMM_WORLD);
   copy.Restore(sx, sw);
   EXPECT_EQ(copy.Size(), orig.Size());
   auto same_guess = [&](const Operator & op)
   {
      Vector b(n), go(n), gc(n);
      b.UseDevice(true);
      go.UseDevice(true);
      gc.UseDevice(true);
      op.Mult(comb, b);
      orig.Guess(b, go);
      copy.Guess(b, gc);
      const double* o = go.HostRead();
      const double* r = gc.HostRead();
      for (int i = 0; i < n; ++i) { ASSERT_EQ(o[i], r[i]); }
   };
   same_guess(A);
   orig.Add(A, xs[3]);
   copy.Add(A, xs[3]);
   same_guess(A);
   orig.Rebuild(A2);
   copy.Rebuild(A2);
   same_guess(A2);
}
