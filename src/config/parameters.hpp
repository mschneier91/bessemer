/**
 * @file parameters.hpp
 * @brief Case parameters (physics, discretization, mesh, time, solver, output)
 *        and the YAML deck loader.
 */
#pragma once

#include "amr/amr_parameters.hpp"
#include "config/nondimensionalization.hpp"
#include "mesh/periodic_box.hpp"
#include "operators/convection.hpp" // ConvectiveForm
#include "operators/grad_div_scale.hpp"
#include "post/body_force.hpp"
#include "precond/cahouet_chabard.hpp"
#include "precond/rotational_schur.hpp"
#include "solver/velocity_preconditioner.hpp"
#include "time/adaptive_controller.hpp"

#include <string>
#include <vector>

namespace incns
{

/// The equation set a case solves -- a runtime option; the unified Case
/// dispatches on it (never a class per physics).
enum class Equation
{
   Stokes,      ///< Unsteady Stokes (Sprint 1).
   NavierStokes ///< Incompressible Navier-Stokes (Sprint 2.2; IMEX BDF/AB).
};

/// How the time step is chosen (deck `time.step_control`).
enum class StepControl
{
   Fixed, ///< dt = time.dt throughout.
   /// Error control: BDF2 (EXT ext_order) advances, an embedded BDF3/EXT3
   /// candidate supplies the estimate (AdaptiveController, time.atol/rtol).
   Error,
   /// Nek-style CFL control: dt = cfl_target / c before every step, c the
   /// convective CFL rate (time/cfl.hpp); one solve per step. Navier-Stokes
   /// only -- Stokes has no convective CFL and steps at time.dt.
   Cfl
};

/// The kind of boundary condition a deck-declared group carries.
enum class BcType
{
   VelocityDirichlet, ///< Prescribed velocity (the field is bound in the driver).
   NoSlip,            ///< Zero velocity wall (no field needed).
   Outflow            ///< Do-nothing / natural traction (no field).
};

/**
 * @brief A boundary-condition group declared in the deck: a label, a boundary
 *        selection, and a type. The field for a Dirichlet group is bound in the
 *        driver by group name (fields never live in the deck).
 *
 * The selection mixes box face names (@c "xmin" ... resolved geometrically),
 * explicit boundary attribute integers, or @c select_all for every real
 * boundary -- MFEM attribute-marker style, by name or number.
 */
struct BcSpec
{
   std::string group;              ///< Label for binding a Dirichlet field.
   BcType type = BcType::VelocityDirichlet; ///< Dirichlet or outflow.
   bool select_all = false;        ///< Select every real (non-periodic) face.
   std::vector<std::string> faces; ///< Face-name selectors (box convenience).
   std::vector<int> attributes;    ///< Explicit boundary attribute selectors.
};

/// Rolling checkpoint settings (see post/checkpoint).
struct CheckpointParameters
{
   bool enabled = false;     ///< Write a rolling checkpoint during Run().
   std::string path = "chk"; ///< Checkpoint directory (overwritten each write).
   int interval = 10;        ///< Write every this many accepted steps.
};

/// ParaView output settings (see post/output).
struct OutputParameters
{
   bool enabled = false;      ///< Write ParaView output during Run().
   std::string path = ".";    ///< Prefix directory for the collection.
   std::string name = "case"; ///< Collection name.
   int interval = 1;          ///< Save every this many accepted steps.
   bool diagnostics = false;  ///< Also log KE/dissipation/||div u|| to a CSV.
};

/**
 * @brief Everything a case needs: physics, discretization, mesh, time
 *        integration, Krylov, initial condition, and output settings.
 *
 * A case is set up by a thin driver -- a .cpp program or a YAML deck via
 * LoadYAML() -- and both go through the same library surface (Case); the
 * solver core is never edited to run a new case. Every field has a sensible
 * default, so a deck only states what differs.
 */
struct Parameters
{
   // --- backend -------------------------------------------------------------
   /// MFEM device backend ("cpu", "cuda", "hip", ...); configured once at case
   /// setup before any mesh/space is built. Default CPU.
   std::string device = "cpu";

   // --- physics -------------------------------------------------------------
   /// Equation set to solve. NavierStokes (Sprint 2.2) adds the dealiased
   /// convection term, extrapolated to t^{n+1} by AB/EXT and carried on the
   /// right-hand side; the implicit block solve is identical to Stokes.
   Equation equation = Equation::Stokes;
   /// Kinematic viscosity. In Dimensionless mode this is read as 1/Re (a deck
   /// may equivalently give `physics: Re`); in Dimensional mode it carries
   /// units and Normalize() converts it to 1/Re.
   double nu = 1.0;
   double grad_div = 0.0; ///< Grad-div scale c_gd; 0 = off.
   /// Grad-div scaling (deck `physics.grad_div_scale: h|nu`).
   GradDivScale grad_div_scale = GradDivScale::OrderH;
   /// Nonlinear-term treatment for NavierStokes (deck
   /// `physics.convective_form: convective|rotational`); see ConvectiveForm.
   ConvectiveForm convective_form = ConvectiveForm::Convective;
   /// Outflow condition on do-nothing boundaries with the convective form
   /// (deck `physics.outflow: directional|classical`; default directional,
   /// human decision 2026-10-08). The rotational form ignores it (classical).
   OutflowCondition outflow = OutflowCondition::Directional;

   /// Input scaling: mode + reference scales (see nondimensionalization.hpp).
   Nondimensionalization nondim;

   // --- discretization (default Q3/Q2 Taylor-Hood) ---------------------------
   int order_u = 3;              ///< Velocity polynomial order k_u.
   int order_p = 2;              ///< Pressure polynomial order k_p.
   bool collocated_mass = false; ///< GLL collocated (diagonal) mass option.

   /// Box mesh specification (quads/hexes; per-direction periodicity).
   BoxSpec mesh;

   /// Adaptive mesh refinement (deck section `amr:`; off by default).
   AmrParameters amr;

   /// Lift/drag on a body (deck section `forces:`; off by default).
   ForceParameters forces;

   // --- time integration ------------------------------------------------------
   /// Fixed step size; the first step in Cfl mode (capped by cfl_target);
   /// the initial guess in Error mode.
   double dt = 1e-2;
   double t_final = 1.0; ///< End time.
   int time_order = 2;   ///< BDF order: 2 production, 3 test-only.
   /// Step-size control (deck `time.step_control: fixed|error|cfl`; the older
   /// `time.adaptive: true|false` means error|fixed). CASE-LEVEL DEFAULT: Cfl
   /// (human decision 2026-10-07: run Navier-Stokes like Nek, at a target CFL
   /// number). Stokes steps at time.dt under Cfl (no convective CFL). The
   /// low-level TimeIntegratorOptions keep fixed steps, so integrator unit
   /// tests pin what they test (the `schur` precedent).
   StepControl step_control = StepControl::Cfl;
   /// Cfl mode: the target CFL number c*dt (deck `time.cfl_target`, > 0). c
   /// is Nek5000's directional CFL rate (time/cfl.hpp), so this is Nek's
   /// `targetCFL`: the step as a multiple of the CFL = 1 step. Explicit
   /// convection bounds it below ~1 (DFG 2D-3: unstable at ~0.8-1.1); larger
   /// multiples need sub-stepped (OIFS) convection, which this code lacks.
   /// See TimeIntegratorOptions::cfl_target for the growth rules.
   double cfl_target = 0.5;
   /// Cfl mode: dt cap (deck `time.dt_max`; 0 = none). Scaled like dt.
   double dt_max = 0.0;
   /// Extrapolation order of the explicit / lagged nonlinear term, 2 or 3
   /// (deck `time.ext_order`): BDF2/EXT2 or BDF2/EXT3. CASE-LEVEL DEFAULT 2
   /// (human decision 2026-10-08, from the DFG 2D-3 study,
   /// docs/imex_vs_semi_implicit.md section 7): on that flow EXT3 fails
   /// silently at CFL 0.7 in BOTH schemes while EXT2 stays accurate, and EXT3
   /// buys no accuracy there (the error is spatial). EXT3's stability region
   /// covers a stretch of the imaginary axis (nearly undamped advection, where
   /// Nek uses it); for damped spectra EXT2's is larger.
   int ext_order = 2;
   AdaptiveControllerOptions controller; ///< Adaptive tolerances/constants.
   /// Convective CFL limit for Navier-Stokes (deck `time.cfl_max`; 0 = off),
   /// BOTH forms: the IMEX convective form transports velocity explicitly,
   /// and the semi-implicit rotational form transports vorticity explicitly
   /// (DFG 2D-3 study: same stability threshold; the rotational failure is
   /// silent). The directional CFL number c*dt (see time/cfl.hpp) may not
   /// exceed it.
   /// Adaptive mode caps the step; fixed-step mode aborts at setup or after
   /// an AMR event that would exceed it (it never changes dt silently).
   double cfl_max = 0.0;

   // --- Krylov ----------------------------------------------------------------
   double krylov_rtol = 1e-10; ///< FGMRES relative tolerance.
   double krylov_atol = 0.0;   ///< FGMRES absolute tolerance.
   int max_iter = 2000;        ///< FGMRES iteration cap.
   int kdim = 200;             ///< FGMRES restart size.
   int print_level = -1;       ///< Solver print level.
   /// Velocity-block preconditioner (deck `solver.preconditioner: jacobi|amg`).
   VelocityPreconditioner velocity_prec = VelocityPreconditioner::Jacobi;
   /// AMG only: reuse (freeze) the LOR hierarchy across Delta-t changes for
   /// cheap adaptive stepping (deck `solver.amg_reuse: true`).
   bool amg_reuse = false;
   /// Velocity-block PC with the rotational form (deck
   /// `solver.rotation_pc: symmetric|pbj_only|pbj_krylov`).
   RotationVelocityPC rotation_pc = RotationVelocityPC::Symmetric;
   /// Rotational form + LOR-AMG velocity PC: put the rotation term into the
   /// LOR operator too, re-set-up every step (deck `solver.rotation_lor`).
   bool rotation_in_lor = false;
   /// Rotational form, CC Schur path: the pressure Schur PC -- deck
   /// `solver.rotation_schur: cc|tensor|auto`, or a map with `mode`,
   /// `criterion: max_mu|volume_fraction`, `mu_on`, `mu_off`, `vol_on`,
   /// `vol_off`, `inner_iterations`. Default cc (today's preconditioner)
   /// until calibrated; see precond/rotational_schur.hpp.
   RotationalSchurPreconditioner::Options rotation_schur;
   /// Rotational form: write the per-step rotation log (rotation number,
   /// viscous ratio, Schur mode, iteration counts, timings) to
   /// `<output.path>/<output.name>_rotation.csv` every this many steps (deck
   /// `solver.rotation_log_interval`; 0 = off).
   int rotation_log_interval = 0;
   /// Pressure Schur block (deck `solver.schur: mass|cc|laplacian_legacy`).
   /// DEFAULT: CahouetChabard (human decision, 2026-07-17) -- Delta-t-robust,
   /// the right default for the adaptive-primary workflow. `mass` remains
   /// available (and is what the low-level solver unit tests pin explicitly).
   SchurBlockType schur = SchurBlockType::CahouetChabard;
   /// CC knobs (deck `solver.n_inner`, `solver.lp_vcycles`,
   /// `solver.block_shape: diag|lower|upper`, `solver.a_pc:
   /// loramg|jacobi_chebyshev|jacobi_pcg`, `solver.a_pcg_rtol`,
   /// `solver.a_pcg_max_iter`); sigma/nu are auto-filled.
   /// CASE-LEVEL DEFAULT velocity PC: JacobiPCG (human decision 2026-10-07,
   /// measured with bench/bench_velocity_pc: never the worst of the three,
   /// 4-8x faster than LOR-AMG at the mass-dominated steps of CFL-limited
   /// NSE, and robust with grad-div, which the LOR operator omits). The
   /// low-level CahouetChabardConfig default stays LORAMG so solver unit
   /// tests and their baselines pin what they test (the `schur` precedent).
   CahouetChabardConfig cc = CaseDefaultCc();

   /// @return The case-level CC configuration defaults (a_pc = JacobiPCG).
   static CahouetChabardConfig CaseDefaultCc()
   {
      CahouetChabardConfig c;
      c.a_pc = APC::JacobiPCG;
      return c;
   }

   // --- case data ---------------------------------------------------------------
   /// Named initial velocity: "zero" or "taylor_green_2d" (uses nu). Decks
   /// select analytic ICs by name so no recompilation is needed per case.
   std::string initial_velocity = "zero";

   /// Boundary-condition groups (topology only; Dirichlet fields are bound in
   /// the driver by group name). Empty means the driver sets BCs explicitly (or
   /// a fully periodic mesh needs none).
   std::vector<BcSpec> boundary_conditions;

   OutputParameters output; ///< ParaView output settings.

   CheckpointParameters checkpoint; ///< Rolling checkpoint settings.
   /// When non-empty, restore the marching state from this checkpoint
   /// directory before stepping (same-np restart; see post/checkpoint).
   std::string restart_from;

   /// @return True when the step is CFL-controlled: StepControl::Cfl on a
   ///         Navier-Stokes case (Stokes steps at time.dt under Cfl).
   bool CflSteps() const
   {
      return step_control == StepControl::Cfl &&
             equation == Equation::NavierStokes;
   }

   /**
    * @brief Apply the convective nondimensionalization in place (idempotent).
    *
    * Dimensionless mode: records Re = 1/nu, nothing else changes. Dimensional
    * mode: rescales mesh lengths (/L_ref), dt and t_final (*U_ref/L_ref),
    * nu -> 1/Re with Re = U_ref*L_ref/nu, and the adaptive atol (/U_ref --
    * the LTE is velocity-normed); named initial conditions other than "zero"
    * are rejected (registry entries are inherently nondimensional -- provide
    * dimensional data through WrapDimensionalVelocity instead).
    *
    * MUST run after the parameters are filled and BEFORE the mesh is built
    * (Case verifies this). LoadYAML() calls it automatically; in-code
    * drivers call it themselves.
    */
   void Normalize();

   /**
    * @brief Load a YAML deck, overriding the defaults field by field, and
    *        Normalize() the result.
    * @param path Path to the YAML file.
    * @return The populated, normalized parameters (throws on malformed input).
    */
   static Parameters LoadYAML(const std::string& path);
};

} // namespace incns
