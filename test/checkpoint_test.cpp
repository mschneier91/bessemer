// Sprint 1.13 -- checkpoint/restart (same-np contract). The green criterion:
// a march interrupted by a checkpoint and restarted in a FRESH case (mesh
// regenerated from the same parameters) finishes with the same solution as the
// uninterrupted march. The restored BDF history/times/dt reproduce the exact
// stepping sequence -- fixed-step to roundoff, adaptive including the PI
// memory so the dt sequence itself continues identically.

#include <gtest/gtest.h>

#include "config/initial_conditions.hpp"
#include "config/parameters.hpp"
#include "mesh/periodic_box.hpp"
#include "post/checkpoint.hpp"
#include "solver/case.hpp"
#include "mfem.hpp"

#include <cmath>

using namespace mfem;
using incns::MakeBoxMesh;
using incns::Parameters;
using incns::Case;

namespace
{

Parameters TgvParams(bool adaptive)
{
   Parameters p;
   p.nu = 1.0;
   p.mesh.dim = 2;
   p.mesh.num_elems = {8, 8, 8};
   p.dt = 0.02;
   p.t_final = 0.2;
   p.initial_velocity = "taylor_green_2d";
   p.adaptive = adaptive;
   if (adaptive)
   {
      p.controller.atol = 2e-4;
      p.controller.rtol = 1e-16;
   }
   p.Normalize();
   return p;
}

struct FinalState
{
   Vector u;
   double t;
   int steps;
};

// March a case to t_final; optionally checkpointing, optionally restarting.
FinalState MarchCase(const Parameters& params)
{
   Mesh serial = MakeBoxMesh(params.mesh);
   ParMesh mesh(MPI_COMM_WORLD, serial);
   Case flow(mesh, params);
   auto u0 = incns::MakeInitialVelocity(params);
   flow.SetInitialVelocity(*u0);
   flow.Run();

   FinalState s;
   s.u.SetSize(flow.Spaces().Velocity().GetTrueVSize());
   flow.Velocity().GetTrueDofs(s.u);
   s.t = flow.Time();
   s.steps = flow.Integrator().StepCount();
   return s;
}

double RelDiff(const Vector& a, const Vector& b)
{
   Vector d(a);
   d -= b;
   const double dn = std::sqrt(InnerProduct(MPI_COMM_WORLD, d, d));
   const double an = std::sqrt(InnerProduct(MPI_COMM_WORLD, a, a));
   return dn / an;
}

void CheckInterruptedMatchesUninterrupted(bool adaptive,
      const std::string& tag)
{
   // Uninterrupted reference.
   const FinalState ref = MarchCase(TgvParams(adaptive));

   // Interrupted run: SAME t_final as the reference (an adaptive march clamps
   // its last steps to land on t_final, so truncating t_final would change the
   // trajectory BEFORE the checkpoint) -- march manually to mid-run and write
   // the checkpoint there, exactly as a killed job's rolling checkpoint would.
   const std::string dir = "chk_" + tag;
   double halted_t = 0.0;
   {
      Parameters first = TgvParams(adaptive);
      Mesh serial = MakeBoxMesh(first.mesh);
      ParMesh mesh(MPI_COMM_WORLD, serial);
      Case flow(mesh, first);
      auto u0 = incns::MakeInitialVelocity(first);
      flow.SetInitialVelocity(*u0);
      while (flow.Time() < 0.1) { flow.Step(); }
      incns::Checkpoint::Write(dir, flow.Integrator(), first);
      halted_t = flow.Time();
   }

   // Fresh case (fresh mesh, fresh operators) restarted from the checkpoint.
   Parameters second = TgvParams(adaptive);
   second.restart_from = dir;
   const FinalState restarted = MarchCase(second);

   EXPECT_NEAR(restarted.t, ref.t, 1e-10);
   EXPECT_EQ(restarted.steps, ref.steps); // identical step sequence
   const double rd = RelDiff(ref.u, restarted.u);
   if (Mpi::Root())
   {
      mfem::out << "[checkpoint:" << tag << "] halted at t=" << halted_t
                << ", restarted -> reldiff vs uninterrupted = " << rd
                << std::endl;
   }
   // Bitwise-identical arithmetic path after restore + binary-exact I/O; the
   // bound is pure-roundoff headroom.
   EXPECT_LE(rd, 1e-13);
}

} // namespace

TEST(Checkpoint, FixedStepRestartMatchesUninterrupted)
{
   CheckInterruptedMatchesUninterrupted(false, "fixed");
}

TEST(Checkpoint, AdaptiveRestartMatchesUninterrupted)
{
   CheckInterruptedMatchesUninterrupted(true, "adaptive");
}

// The deck-driven rolling-checkpoint path (checkpoint.enabled + interval)
// writes a loadable checkpoint during Run(); restarting from it reproduces
// the reference tail. Fixed step with t_final on the step grid, so the
// interrupted trajectory is identical to the reference by construction.
TEST(Checkpoint, DeckDrivenRollingCheckpoint)
{
   const FinalState ref = MarchCase(TgvParams(false));

   Parameters first = TgvParams(false);
   first.t_final = 0.1; // exact multiple of dt: no trajectory change
   first.checkpoint.enabled = true;
   first.checkpoint.path = "chk_deck";
   first.checkpoint.interval = 1;
   MarchCase(first);

   Parameters second = TgvParams(false);
   second.restart_from = "chk_deck";
   const FinalState restarted = MarchCase(second);

   EXPECT_EQ(restarted.steps, ref.steps);
   EXPECT_LE(RelDiff(ref.u, restarted.u), 1e-13);
}
