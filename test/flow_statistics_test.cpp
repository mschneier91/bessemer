// Flow statistics (post/flow_statistics, post/flow_sampling) and their place
// in a run:
//  F1 the nodal running moments: uneven sample times, right-endpoint
//     weights, the means and Reynolds stresses of u = A + a(t) B exact at
//     every node; a sample before start_time or at a repeated time ignored;
//  F2 Save()/Load(): continuing from a saved state is bitwise the
//     uninterrupted accumulation;
//  F3 lines and azimuthally averaged profiles of fields the Q2 space holds
//     exactly: velocity, pressure, Reynolds and viscous stresses, Cartesian
//     and cylindrical; a point outside the mesh is NaN;
//  F4 wall bins: area, mean pressure and wall shear stress of a Poiseuille
//     profile (2D both walls, 3D the cross component), a wall facing along
//     the axis skipped;
//  F5 the half-window split: the second part's means and Reynolds stresses
//     exact, the relative differences, the CSV files, Save()/Load() of the
//     split;
//  F6 in a run: the statistics leave the solution alone (bitwise on a host),
//     a restart continues them, and the summary and CSV files carry them.

#include <gtest/gtest.h>

#include "bc/boundary_names.hpp"
#include "config/parameters.hpp"
#include "mesh/case_mesh.hpp"
#include "post/flow_sampling.hpp"
#include "post/flow_statistics.hpp"
#include "repro_tolerance.hpp"
#include "solver/case.hpp"
#include "spaces/mixed_spaces.hpp"
#include "mfem.hpp"

#include <yaml-cpp/yaml.h>

#include <cmath>
#include <filesystem>
#include <string>
#include <vector>

using namespace mfem;

namespace
{
const double kNu = 0.01;

// A box [0, L] (no periodicity) and the parameters that made it.
std::unique_ptr<ParMesh> Box(int dim, const std::string& elements,
                             const std::string& lengths, incns::Parameters* out)
{
   const std::string deck =
      "mesh:\n  dim: " + std::to_string(dim) + "\n  elements: " + elements +
      "\n  lengths: " + lengths + "\n  periodic: " +
      (dim == 3 ? "[false, false, false]" : "[false, false]") + "\n";
   const incns::Parameters p = incns::Parameters::LoadYAMLString(deck);
   if (out) { *out = p; }
   return incns::MakeCaseMesh(p);
}

std::string Tag() { return "_np" + std::to_string(Mpi::WorldSize()); }

// Rank 0 creates @p dir; everyone waits.
void MakeDir(const std::string& dir)
{
   if (Mpi::Root()) { std::filesystem::create_directories(dir); }
   MPI_Barrier(MPI_COMM_WORLD);
}

void RemoveDir(const std::string& dir)
{
   MPI_Barrier(MPI_COMM_WORLD);
   if (Mpi::Root()) { std::filesystem::remove_all(dir); }
}

// The value of table column @p name in row @p r.
double Col(const incns::SampleTable& t, int r, const std::string& name)
{
   for (std::size_t c = 0; c < t.columns.size(); ++c)
   {
      if (t.columns[c] == name) { return t.values[r * t.columns.size() + c]; }
   }
   ADD_FAILURE() << "no column " << name << " in " << t.name;
   return 0.0;
}
} // namespace

TEST(FlowStatistics, F1_RunningMomentsAreExact)
{
   auto mesh = Box(3, "[2, 2, 2]", "[1.0, 1.0, 1.0]", nullptr);
   incns::MixedSpaces spaces(*mesh, 2, 1);
   ParFiniteElementSpace& V = spaces.Velocity();
   ParFiniteElementSpace& Q = spaces.Pressure();
   ParGridFunction A(&V), B(&V), u(&V), pa(&Q), pb(&Q), p(&Q);
   VectorFunctionCoefficient ca(3, [](const Vector & x, Vector & v)
   {
      v(0) = 1.0 + x(1) * x(1);
      v(1) = x(0) - x(2);
      v(2) = 0.5 * x(0) * x(1);
   });
   VectorFunctionCoefficient cb(3, [](const Vector & x, Vector & v)
   {
      v(0) = x(2);
      v(1) = 1.0 - x(0) * x(0);
      v(2) = 0.3 + x(1);
   });
   FunctionCoefficient qa([](const Vector & x) { return x(0) + x(1) * x(2); });
   FunctionCoefficient qb([](const Vector & x) { return 1.0 - x(2); });
   A.ProjectCoefficient(ca);
   B.ProjectCoefficient(cb);
   pa.ProjectCoefficient(qa);
   pb.ProjectCoefficient(qb);

   incns::FlowStatistics fs(spaces.Velocity(), spaces.Pressure(), 0.1);
   // t = 0 is before start_time and the second t = 0.3 repeats a time.
   const std::vector<double> t = {0.0, 0.1, 0.3, 0.3, 0.35, 0.8, 1.0};
   const std::vector<double> a = {9.0, 1.0, -2.0, 7.0, 0.5, 3.0, -1.0};
   for (std::size_t i = 0; i < t.size(); ++i)
   {
      u = A;
      u.Add(a[i], B);
      p = pa;
      p.Add(a[i], pb);
      EXPECT_EQ(fs.Sample(u, p, t[i]), i != 0 && i != 3) << i;
   }
   EXPECT_EQ(fs.Samples(), 5);
   EXPECT_NEAR(fs.AveragingTime(), 0.9, 1e-15);
   EXPECT_EQ(fs.FirstTime(), 0.1);
   EXPECT_EQ(fs.LastTime(), 1.0);

   // The weights: 0 for t = 0.1, then the time since the previous sample.
   const std::vector<double> w = {0.2, 0.05, 0.45, 0.2}, av = {-2.0, 0.5, 3.0, -1.0};
   double W = 0.0, mean = 0.0, var = 0.0;
   for (std::size_t i = 0; i < w.size(); ++i) { W += w[i]; mean += w[i] * av[i]; }
   mean /= W;
   for (std::size_t i = 0; i < w.size(); ++i) { var += w[i] * (av[i] - mean) * (av[i] - mean); }
   var /= W;

   const int N = spaces.Velocity().GetNDofs();
   const double* mu = fs.MeanVelocity().HostRead();
   const double* rn = fs.NormalStress().HostRead();
   const double* rs = fs.ShearStress().HostRead();
   const double* mp = fs.MeanPressure().HostRead();
   const double* Ad = A.HostRead();
   const double* Bd = B.HostRead();
   for (int n = 0; n < N; ++n)
   {
      const double b[3] = {Bd[n], Bd[N + n], Bd[2 * N + n]};
      for (int c = 0; c < 3; ++c)
      {
         EXPECT_NEAR(mu[c * N + n], Ad[c * N + n] + mean * b[c], 1e-13);
         EXPECT_NEAR(rn[c * N + n], var * b[c] * b[c], 1e-13);
      }
      EXPECT_NEAR(rs[n], var * b[0] * b[1], 1e-13);
      EXPECT_NEAR(rs[N + n], var * b[0] * b[2], 1e-13);
      EXPECT_NEAR(rs[2 * N + n], var * b[1] * b[2], 1e-13);
   }
   const double* pad = pa.HostRead();
   const double* pbd = pb.HostRead();
   for (int n = 0; n < spaces.Pressure().GetNDofs(); ++n)
   {
      EXPECT_NEAR(mp[n], pad[n] + mean * pbd[n], 1e-13);
   }
}

TEST(FlowStatistics, F2_SaveLoadContinuesExactly)
{
   auto mesh = Box(2, "[3, 2]", "[1.5, 1.0]", nullptr);
   incns::MixedSpaces spaces(*mesh, 3, 2);
   ParGridFunction u(&spaces.Velocity()), p(&spaces.Pressure());
   const std::vector<double> times = {0.0, 0.2, 0.5, 0.6, 1.3, 1.4};
   auto fill = [&](double t)
   {
      VectorFunctionCoefficient cu(2, [t](const Vector & x, Vector & v)
      {
         v(0) = std::sin(3.0 * t + x(0)) * x(1);
         v(1) = std::cos(2.0 * t - x(1)) + x(0) * t;
      });
      FunctionCoefficient cp([t](const Vector & x) { return std::cos(t * x(0)) - x(1); });
      u.ProjectCoefficient(cu);
      p.ProjectCoefficient(cp);
   };
   incns::FlowStatistics whole(spaces.Velocity(), spaces.Pressure(), 0.0);
   for (double t : times)
   {
      fill(t);
      whole.Sample(u, p, t);
   }
   const std::string dir = "flowstats_f2" + Tag();
   MakeDir(dir);
   {
      incns::FlowStatistics first(spaces.Velocity(), spaces.Pressure(), 0.0);
      for (int i = 0; i < 3; ++i)
      {
         fill(times[i]);
         first.Sample(u, p, times[i]);
      }
      first.Save(dir);
   }
   MPI_Barrier(MPI_COMM_WORLD);
   ASSERT_TRUE(incns::FlowStatistics::Saved(dir));
   incns::FlowStatistics second(spaces.Velocity(), spaces.Pressure(), 0.0);
   second.Load(dir);
   for (std::size_t i = 3; i < times.size(); ++i)
   {
      fill(times[i]);
      second.Sample(u, p, times[i]);
   }
   RemoveDir(dir);

   EXPECT_EQ(second.Samples(), whole.Samples());
   EXPECT_EQ(second.AveragingTime(), whole.AveragingTime());
   EXPECT_EQ(second.FirstTime(), whole.FirstTime());
   auto same = [](const Vector & a, const Vector & b)
   {
      ASSERT_EQ(a.Size(), b.Size());
      const double* x = a.HostRead();
      const double* y = b.HostRead();
      for (int i = 0; i < a.Size(); ++i) { ASSERT_EQ(x[i], y[i]) << i; }
   };
   same(second.MeanVelocity(), whole.MeanVelocity());
   same(second.MeanPressure(), whole.MeanPressure());
   same(second.NormalStress(), whole.NormalStress());
   same(second.ShearStress(), whole.ShearStress());
}

namespace
{
// F3's fields about the axis x = y = 1: a swirl, a paraboloid along z and a
// fluctuation V; their samples u = U0 +- V give mean U0 and R = V V^T.
void U0(const Vector& x, Vector& v)
{
   const double X = x(0) - 1.0, Y = x(1) - 1.0;
   v(0) = -Y;
   v(1) = X;
   v(2) = 1.0 - X * X - Y * Y;
}
void Vf(const Vector& x, Vector& v)
{
   v(0) = x(0) - 1.0;
   v(1) = 0.0;
   v(2) = 1.0;
}
double P0(const Vector& x) { return (x(0) - 1.0) + 2.0 * (x(1) - 1.0) + x(2); }
} // namespace

TEST(FlowStatistics, F3_LinesAndProfilesAreExact)
{
   auto mesh = Box(3, "[2, 2, 2]", "[2.0, 2.0, 2.0]", nullptr);
   incns::MixedSpaces spaces(*mesh, 2, 1);
   ParFiniteElementSpace& V = spaces.Velocity();
   ParFiniteElementSpace& Q = spaces.Pressure();
   ParGridFunction u0(&V), v(&V), u(&V), p0(&Q), p(&Q);
   VectorFunctionCoefficient cu(3, U0), cv(3, Vf);
   FunctionCoefficient cp(P0);
   u0.ProjectCoefficient(cu);
   v.ProjectCoefficient(cv);
   p0.ProjectCoefficient(cp);
   incns::FlowStatistics fs(spaces.Velocity(), spaces.Pressure(), 0.0);
   for (int k = 0; k < 3; ++k)
   {
      const double s = (k == 1) ? 1.0 : -1.0; // t = 0 has weight 0
      u = u0;
      u.Add(s, v);
      p = p0;
      p += 0.3 * s;
      fs.Sample(u, p, static_cast<double>(k));
   }

   incns::FlowStatisticsParameters sp;
   incns::StatisticsLine diag;
   diag.name = "diag";
   diag.from = {0.5, 0.5, 0.2};
   diag.to = {1.5, 1.5, 1.8};
   diag.points = 11;
   incns::StatisticsLine out;
   out.name = "outside";
   out.from = {0.5, 1.0, 1.0};
   out.to = {2.5, 1.0, 1.0};
   out.points = 5;
   sp.lines = {diag, out};
   incns::StatisticsProfiles pr;
   pr.name = "pipe";
   pr.origin = {1.0, 1.0, 0.0};
   pr.axis = {0.0, 0.0, 2.0};
   pr.stations = {0.5, 1.5};
   pr.radius = 0.8;
   pr.points = 5;
   pr.azimuths = 8;
   sp.profiles = {pr};
   incns::FlowSampler sampler(*mesh, sp, kNu, 2);
   const incns::SampleValues values = sampler.Evaluate(fs);
   const std::vector<incns::SampleTable> tables = sampler.Tabulate(values);
   if (!Mpi::Root())
   {
      EXPECT_TRUE(tables.empty());
      return;
   }
   EXPECT_EQ(sampler.PointsNotFound(), 1); // x = 2.5
   ASSERT_EQ(tables.size(), 3u);

   const double tol = 1e-10;
   const incns::SampleTable& t = tables[0];
   ASSERT_EQ(t.Rows(), 11);
   for (int r = 0; r < t.Rows(); ++r)
   {
      Vector x(3), uv(3), vv(3);
      x(0) = Col(t, r, "x");
      x(1) = Col(t, r, "y");
      x(2) = Col(t, r, "z");
      U0(x, uv);
      Vf(x, vv);
      const double X = x(0) - 1.0, Y = x(1) - 1.0;
      EXPECT_NEAR(Col(t, r, "s"), std::sqrt(1.0 + 1.0 + 1.6 * 1.6) * r / 10.0, 1e-13);
      EXPECT_NEAR(Col(t, r, "u"), uv(0), tol);
      EXPECT_NEAR(Col(t, r, "v"), uv(1), tol);
      EXPECT_NEAR(Col(t, r, "w"), uv(2), tol);
      EXPECT_NEAR(Col(t, r, "p"), P0(x), tol);
      EXPECT_NEAR(Col(t, r, "R_xx"), vv(0) * vv(0), tol);
      EXPECT_NEAR(Col(t, r, "R_yy"), 0.0, tol);
      EXPECT_NEAR(Col(t, r, "R_zz"), 1.0, tol);
      EXPECT_NEAR(Col(t, r, "R_xy"), 0.0, tol);
      EXPECT_NEAR(Col(t, r, "R_xz"), vv(0), tol);
      EXPECT_NEAR(Col(t, r, "R_yz"), 0.0, tol);
      EXPECT_NEAR(Col(t, r, "tau_xx"), 0.0, tol);
      EXPECT_NEAR(Col(t, r, "tau_xy"), 0.0, tol); // the swirl is a rotation
      EXPECT_NEAR(Col(t, r, "tau_xz"), -2.0 * kNu * X, tol);
      EXPECT_NEAR(Col(t, r, "tau_yz"), -2.0 * kNu * Y, tol);
   }
   const incns::SampleTable& o = tables[1];
   for (int r = 0; r < 4; ++r) { EXPECT_FALSE(std::isnan(Col(o, r, "u"))) << r; }
   EXPECT_TRUE(std::isnan(Col(o, 4, "u")));
   EXPECT_TRUE(std::isnan(Col(o, 4, "tau_yz")));

   // Cylindrical, averaged over 8 angles from x: u_a = 1 - r^2, u_t = r; with
   // V = (r cos, 0, 1): R_rr = 3 r^2 / 8, R_tt = r^2 / 8, R_ar = r / 2.
   const incns::SampleTable& q = tables[2];
   ASSERT_EQ(q.Rows(), 2 * 5);
   for (int row = 0; row < q.Rows(); ++row)
   {
      const double z = Col(q, row, "station"), r = Col(q, row, "r");
      EXPECT_DOUBLE_EQ(z, row < 5 ? 0.5 : 1.5);
      EXPECT_NEAR(r, 0.2 * (row % 5), 1e-14);
      EXPECT_EQ(Col(q, row, "n"), 8.0);
      EXPECT_NEAR(Col(q, row, "u_a"), 1.0 - r * r, tol);
      EXPECT_NEAR(Col(q, row, "u_r"), 0.0, tol);
      EXPECT_NEAR(Col(q, row, "u_t"), r, tol);
      EXPECT_NEAR(Col(q, row, "p"), z, tol);
      EXPECT_NEAR(Col(q, row, "R_aa"), 1.0, tol);
      EXPECT_NEAR(Col(q, row, "R_rr"), 0.375 * r * r, tol);
      EXPECT_NEAR(Col(q, row, "R_tt"), 0.125 * r * r, tol);
      EXPECT_NEAR(Col(q, row, "R_ar"), 0.5 * r, tol);
      EXPECT_NEAR(Col(q, row, "R_at"), 0.0, tol);
      EXPECT_NEAR(Col(q, row, "R_rt"), 0.0, tol);
      EXPECT_NEAR(Col(q, row, "tau_ar"), -2.0 * kNu * r, tol);
      EXPECT_NEAR(Col(q, row, "tau_at"), 0.0, tol);
      EXPECT_NEAR(Col(q, row, "tau_rt"), 0.0, tol);
      EXPECT_NEAR(Col(q, row, "tau_aa"), 0.0, tol);
   }
}

TEST(FlowStatistics, F4_WallBins)
{
   // 2D: u = 4 y (1 - y) on [0, 4] x [0, 1]: wall shear 4 nu along x on both
   // walls (sign into the fluid); p = 1 + x.
   {
      incns::Parameters bp;
      auto mesh = Box(2, "[4, 2]", "[4.0, 1.0]", &bp);
      incns::MixedSpaces spaces(*mesh, 2, 1);
      ParGridFunction u(&spaces.Velocity()), p(&spaces.Pressure());
      VectorFunctionCoefficient cu(2, [](const Vector & x, Vector & v)
      {
         v(0) = 4.0 * x(1) * (1.0 - x(1));
         v(1) = 0.0;
      });
      FunctionCoefficient cp([](const Vector & x) { return 1.0 + x(0); });
      u.ProjectCoefficient(cu);
      p.ProjectCoefficient(cp);
      incns::FlowStatistics fs(spaces.Velocity(), spaces.Pressure(), 0.0);
      fs.Sample(u, p, 0.0);
      fs.Sample(u, p, 1.0);

      incns::FlowStatisticsParameters sp;
      incns::StatisticsWall both;
      both.name = "walls";
      both.boundaries = {"ymin", "ymax"};
      both.attributes = {incns::NamedBoundaryAttribute(bp, *mesh, "ymin"),
                         incns::NamedBoundaryAttribute(bp, *mesh, "ymax")
                        };
      both.axis = {2.0, 0.0, 0.0};
      both.from = 0.0;
      both.to = 4.0;
      both.bins = 4; // on the element edges: the quadrature splits exactly
      incns::StatisticsWall inlet = both;
      inlet.name = "with_inlet";
      inlet.boundaries = {"xmin", "ymin"};
      inlet.attributes = {incns::NamedBoundaryAttribute(bp, *mesh, "xmin"),
                          incns::NamedBoundaryAttribute(bp, *mesh, "ymin")
                         };
      inlet.bins = 4;
      sp.walls = {both, inlet};
      incns::FlowSampler sampler(*mesh, sp, kNu, 2);
      const incns::SampleValues v = sampler.Evaluate(fs);
      const std::vector<incns::SampleTable> t = sampler.Tabulate(v);
      if (Mpi::Root())
      {
         ASSERT_EQ(t.size(), 2u);
         ASSERT_EQ(t[0].Rows(), 4);
         for (int b = 0; b < 4; ++b)
         {
            EXPECT_NEAR(Col(t[0], b, "xi"), b + 0.5, 1e-14);
            EXPECT_NEAR(Col(t[0], b, "area"), 2.0, 1e-13); // both walls
            EXPECT_NEAR(Col(t[0], b, "p"), 1.5 + b, 1e-12);
            EXPECT_NEAR(Col(t[0], b, "wss_axial"), 4.0 * kNu, 1e-12);
         }
         EXPECT_EQ(v.wall_skipped[0], 0);
         EXPECT_GT(v.wall_skipped[1], 0); // xmin's normal is the axis
         for (int b = 0; b < 4; ++b)
         {
            EXPECT_NEAR(Col(t[1], b, "area"), 1.0, 1e-13);
            EXPECT_NEAR(Col(t[1], b, "p"), 1.5 + b, 1e-12);
            EXPECT_NEAR(Col(t[1], b, "wss_axial"), 4.0 * kNu, 1e-12);
         }
      }
   }
   // 3D: u = w = 4 y (1 - y): at y = 0 the cross direction n x e_x is -z.
   {
      incns::Parameters bp;
      auto mesh = Box(3, "[2, 2, 2]", "[2.0, 1.0, 1.0]", &bp);
      incns::MixedSpaces spaces(*mesh, 2, 1);
      ParGridFunction u(&spaces.Velocity()), p(&spaces.Pressure());
      VectorFunctionCoefficient cu(3, [](const Vector & x, Vector & v)
      {
         v(0) = 4.0 * x(1) * (1.0 - x(1));
         v(1) = 0.0;
         v(2) = v(0);
      });
      u.ProjectCoefficient(cu);
      p = 2.0;
      incns::FlowStatistics fs(spaces.Velocity(), spaces.Pressure(), 0.0);
      fs.Sample(u, p, 0.0);
      incns::FlowStatisticsParameters sp;
      incns::StatisticsWall bottom;
      bottom.name = "bottom";
      bottom.boundaries = {"ymin"};
      bottom.attributes = {incns::NamedBoundaryAttribute(bp, *mesh, "ymin")};
      bottom.axis = {1.0, 0.0, 0.0};
      bottom.from = 0.0;
      bottom.to = 2.0;
      bottom.bins = 2;
      sp.walls = {bottom};
      incns::FlowSampler sampler(*mesh, sp, kNu, 2);
      const incns::SampleValues v = sampler.Evaluate(fs);
      const std::vector<incns::SampleTable> t = sampler.Tabulate(v);
      if (Mpi::Root())
      {
         for (int b = 0; b < 2; ++b)
         {
            EXPECT_NEAR(Col(t[0], b, "area"), 1.0, 1e-13);
            EXPECT_NEAR(Col(t[0], b, "p"), 2.0, 1e-13);
            EXPECT_NEAR(Col(t[0], b, "wss_axial"), 4.0 * kNu, 1e-12);
            EXPECT_NEAR(Col(t[0], b, "wss_cross"), -4.0 * kNu, 1e-12);
         }
      }
   }
}

TEST(FlowStatistics, F5_HalfWindowSplit)
{
   auto mesh = Box(3, "[2, 2, 2]", "[1.0, 1.0, 1.0]", nullptr);
   incns::MixedSpaces spaces(*mesh, 2, 1);
   ParGridFunction b(&spaces.Velocity()), u(&spaces.Velocity());
   ParGridFunction q(&spaces.Pressure()), p(&spaces.Pressure());
   // B linear, so B_i B_j is in Q2 and the nodal stresses interpolate exactly.
   auto B = [](const Vector & x, Vector & v)
   {
      v(0) = 1.0 + x(0);
      v(1) = x(1);
      v(2) = 0.5;
   };
   VectorFunctionCoefficient cb(3, B);
   FunctionCoefficient cq([](const Vector & x) { return 1.0 + x(0); });
   b.ProjectCoefficient(cb);
   q.ProjectCoefficient(cq);

   incns::FlowStatisticsParameters sp;
   incns::StatisticsLine line;
   line.name = "line";
   line.from = {0.1, 0.2, 0.3};
   line.to = {0.9, 0.8, 0.7};
   line.points = 7;
   sp.lines = {line};
   incns::FlowSampler sampler(*mesh, sp, kNu, 2);
   incns::FlowStatistics fs(spaces.Velocity(), spaces.Pressure(), 0.0);
   // Scales 1, 3 up to the split (mean 2, variance 1), then 2, 6 (mean 4,
   // variance 4); t = 0 has weight 0.
   const std::vector<double> s = {5.0, 1.0, 3.0, 2.0, 6.0};
   for (int k = 0; k < 5; ++k)
   {
      u.Set(s[k], b);
      p.Set(s[k], q);
      fs.Sample(u, p, static_cast<double>(k));
      if (k == 2) { sampler.Snapshot(fs); }
   }
   EXPECT_TRUE(sampler.HasSplit());
   EXPECT_EQ(sampler.SplitWeight(), 2.0);
   EXPECT_EQ(sampler.SplitTime(), 2.0);

   const incns::SampleValues all = sampler.Evaluate(fs);
   const std::string dir = "flowstats_f5" + Tag();
   MakeDir(dir);
   sampler.WriteCsv(all, fs, dir + "/run_stats");
   sampler.Save(dir);
   MPI_Barrier(MPI_COMM_WORLD);
   incns::FlowSampler reloaded(*mesh, sp, kNu, 2);
   reloaded.Load(dir);
   EXPECT_TRUE(reloaded.HasSplit());
   EXPECT_EQ(reloaded.SplitWeight(), 2.0);
   EXPECT_EQ(reloaded.SplitTime(), 2.0);
   if (Mpi::Root())
   {
      const std::vector<std::string> files = {"/run_stats_line.csv",
                                              "/run_stats_line_half1.csv",
                                              "/run_stats_line_half2.csv"
                                             };
      for (const std::string& f : files)
      {
         EXPECT_TRUE(std::filesystem::exists(dir + f)) << f;
      }
      ASSERT_EQ(reloaded.FirstPart().points.size(), 1u);
      const std::vector<double>& a = sampler.FirstPart().points[0];
      const std::vector<double>& c = reloaded.FirstPart().points[0];
      ASSERT_EQ(a.size(), c.size());
      for (std::size_t i = 0; i < a.size(); ++i) { EXPECT_EQ(a[i], c[i]) << i; }
   }
   RemoveDir(dir);
   if (!Mpi::Root()) { return; }

   using Tables = std::vector<incns::SampleTable>;
   const Tables t1 = sampler.Tabulate(sampler.FirstPart());
   const Tables t2 =
      sampler.Tabulate(sampler.SecondPart(all, fs.AveragingTime()));
   const Tables ta = sampler.Tabulate(all);
   const double tol = 1e-10;
   const char* ax = "xyz";
   for (int r = 0; r < 7; ++r)
   {
      Vector x(3), bv(3);
      x(0) = Col(ta[0], r, "x");
      x(1) = Col(ta[0], r, "y");
      x(2) = Col(ta[0], r, "z");
      B(x, bv);
      for (int c = 0; c < 3; ++c)
      {
         const std::string uc(1, "uvw"[c]);
         EXPECT_NEAR(Col(t1[0], r, uc), 2.0 * bv(c), tol);
         EXPECT_NEAR(Col(t2[0], r, uc), 4.0 * bv(c), tol);
         EXPECT_NEAR(Col(ta[0], r, uc), 3.0 * bv(c), tol);
         for (int d = c; d < 3; ++d)
         {
            const std::string rc = std::string("R_") + ax[c] + ax[d];
            EXPECT_NEAR(Col(t1[0], r, rc), 1.0 * bv(c) * bv(d), tol) << rc;
            EXPECT_NEAR(Col(t2[0], r, rc), 4.0 * bv(c) * bv(d), tol) << rc;
            EXPECT_NEAR(Col(ta[0], r, rc), 3.5 * bv(c) * bv(d), tol) << rc;
         }
      }
      EXPECT_NEAR(Col(t2[0], r, "p"), 4.0 * (1.0 + x(0)), tol);
      EXPECT_NEAR(Col(t2[0], r, "tau_xx"), 4.0 * 2.0 * kNu, tol);
   }
   // max |2B - 4B| / max |3B| = 2/3; stresses (4 - 1) / 3.5.
   const std::vector<incns::HalfWindowDifference> diffs =
      sampler.Compare(all, fs.AveragingTime());
   for (const incns::HalfWindowDifference& d : diffs)
   {
      EXPECT_EQ(d.set, "line");
      if (d.group == "velocity" || d.group == "viscous_stress")
      {
         EXPECT_NEAR(d.relative, 2.0 / 3.0, 1e-10) << d.group;
      }
      if (d.group == "reynolds_stress") { EXPECT_NEAR(d.relative, 3.0 / 3.5, 1e-10); }
      if (d.group == "pressure") { EXPECT_TRUE(std::isfinite(d.relative)); }
   }
}

namespace
{
// A 2D channel with a parabolic inflow, fixed steps.
incns::Parameters ChannelDeck(const std::string& out, int steps,
                              const std::string& extra,
                              const std::string& output_keys = "")
{
   const std::string deck =
      "equation: navier_stokes\n"
      "physics:\n  nu: 0.05\n"
      "mesh:\n  dim: 2\n  elements: [8, 2]\n  lengths: [4.0, 1.0]\n"
      "  periodic: [false, false]\n"
      "time:\n  dt: 0.01\n  t_final: " + std::to_string(0.01 * steps) + "\n"
      "  step_control: fixed\n"
      "boundary_conditions:\n"
      "  - {select: [xmin], type: velocity, profile: parabolic, u_max: 1.0, "
      "height: 1.0}\n"
      "  - {select: [xmax], type: outflow}\n"
      "  - {select: [ymin, ymax], type: no_slip}\n"
      // See channel_statistics_test S5: a tight rtol keeps a GPU's
      // run-to-run floor small; on a host the runs are bitwise.
      "solver:\n  rtol: 1.0e-10\n"
      "output:\n  path: " + out + "\n  name: fs\n" + output_keys + extra;
   return incns::Parameters::LoadYAMLString(deck);
}

const char* kStats =
   "flow_statistics:\n  enabled: true\n"
   "  lines:\n    - {name: center, from: [0.0, 0.5], to: [4.0, 0.5], points: 21}\n"
   "  walls:\n    - {name: wall, boundaries: [ymin, ymax], origin: [0.0, 0.0], "
   "axis: [1.0, 0.0], from: 0.0, to: 4.0, bins: 8}\n";

struct RunResult
{
   Vector u;                    ///< Final velocity, true dofs.
   Vector mean_u, normal, shear; ///< Statistics, local.
   long samples = 0;
};

RunResult RunChannel(const incns::Parameters& p, const std::string& summary)
{
   auto mesh = incns::MakeCaseMesh(p);
   incns::Case flow(*mesh, p);
   flow.Run();
   if (!summary.empty()) { flow.WriteSummary(summary, "ok"); }
   RunResult r;
   r.u.SetSize(flow.Velocity().ParFESpace()->GetTrueVSize());
   flow.Velocity().GetTrueDofs(r.u);
   if (flow.FlowStats())
   {
      r.mean_u = flow.FlowStats()->MeanVelocity();
      r.normal = flow.FlowStats()->NormalStress();
      r.shear = flow.FlowStats()->ShearStress();
      r.samples = flow.FlowStats()->Samples();
   }
   return r;
}

void ExpectClose(const Vector& a, const Vector& b, double tol, const char* what)
{
   ASSERT_EQ(a.Size(), b.Size()) << what;
   const double* x = a.HostRead();
   const double* y = b.HostRead();
   double scale = 0.0;
   for (int i = 0; i < a.Size(); ++i) { scale = std::max(scale, std::abs(x[i])); }
   MPI_Allreduce(MPI_IN_PLACE, &scale, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
   for (int i = 0; i < a.Size(); ++i)
   {
      ASSERT_NEAR(x[i], y[i], tol * scale) << what << " " << i;
   }
}
} // namespace

TEST(FlowStatistics, F6_InARun)
{
   const std::string base = "flowstats_run" + Tag();
   const std::string chk = base + "/chk";
   const double tol = incns_test::ReproTol(0.0);

   // The statistics are a passenger: same solution with and without them
   // (the ParaView output carries the fields too).
   const RunResult plain = RunChannel(ChannelDeck(base + "/plain", 6, ""), "");
   const incns::Parameters ref_deck =
      ChannelDeck(base + "/ref", 6, kStats, "  enabled: true\n  interval: 3\n");
   const RunResult ref = RunChannel(ref_deck, base + "/ref_summary.json");
   ExpectClose(ref.u, plain.u, tol, "velocity");
   EXPECT_EQ(ref.samples, 7); // the initial state + 6 steps

   // 3 steps and a checkpoint, then a restart to step 6.
   RunChannel(ChannelDeck(base + "/a", 3, std::string(kStats) +
                          "checkpoint:\n  enabled: true\n  interval: 3\n  path: " +
                          chk + "\n"), "");
   const std::string resume = std::string(kStats) + "restart: " + chk + "\n";
   const RunResult restarted = RunChannel(ChannelDeck(base + "/b", 6, resume), "");
   EXPECT_EQ(restarted.samples, ref.samples);
   const double rtol = incns_test::ReproTol(1e-13);
   ExpectClose(restarted.mean_u, ref.mean_u, rtol, "mean velocity");
   ExpectClose(restarted.normal, ref.normal, rtol, "normal stresses");
   ExpectClose(restarted.shear, ref.shear, rtol, "shear stresses");

   MPI_Barrier(MPI_COMM_WORLD);
   const YAML::Node sum = YAML::LoadFile(base + "/ref_summary.json");
   const YAML::Node st = sum["flow_statistics"];
   ASSERT_TRUE(st.IsDefined());
   EXPECT_EQ(st["samples"].as<int>(), 7);
   EXPECT_NEAR(st["averaging_time"].as<double>(), 0.06, 1e-12);
   // The first sample at or past the middle of [0, 0.06] (step 3, or 4 when
   // the summed time lands just short of 0.03).
   const double split = st["split"]["time"].as<double>();
   EXPECT_GE(split, 0.03 - 1e-9);
   EXPECT_LE(split, 0.04 + 1e-9);
   EXPECT_NEAR(st["split"]["first_averaging_time"].as<double>(), split, 1e-12);
   for (const char* g : {"velocity", "pressure", "reynolds_stress", "viscous_stress"})
   {
      EXPECT_GE(st["half_window_difference"]["center"][g].as<double>(), 0.0) << g;
   }
   EXPECT_GE(st["half_window_difference"]["wall"]["wss"].as<double>(), 0.0);
   const std::vector<std::string> files = {"/ref/fs_stats_center.csv",
                                           "/ref/fs_stats_center_half1.csv",
                                           "/ref/fs_stats_center_half2.csv",
                                           "/ref/fs_stats_wall.csv"
                                          };
   for (const std::string& f : files)
   {
      EXPECT_TRUE(std::filesystem::exists(base + f)) << f;
   }
   RemoveDir(base);
}
