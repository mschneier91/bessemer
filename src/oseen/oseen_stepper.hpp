/**
 * @file oseen_stepper.hpp
 * @brief Semi-implicit and implicit BDF2 for Navier-Stokes with every linear
 *        solve an Oseen solve (docs/design/oseen_bdf2.md).
 */
#pragma once

#include "bc/boundary_conditions.hpp"
#include "operators/stokes_operator.hpp"
#include "oseen/oseen_convection.hpp"
#include "oseen/oseen_lor_pc.hpp"
#include "oseen/oseen_pcd.hpp"
#include "oseen/oseen_wind.hpp"
#include "precond/block_stokes_pc.hpp"
#include "precond/cahouet_chabard.hpp"
#include "quadrature/rule_book.hpp"
#include "spaces/mixed_spaces.hpp"
#include "mfem.hpp"

#include <memory>
#include <vector>

namespace incns
{

/// How the convection of the Oseen schemes is linearized.
enum class OseenScheme
{
   /// One Oseen solve per step, the wind extrapolated: w = 2 u^n - u^{n-1}.
   SemiImplicit,
   /// Picard iterations, the wind the current iterate, to a nonlinear
   /// tolerance or a fixed count; the first iteration is the semi-implicit
   /// step's solve with the wind of the extrapolated guess.
   Implicit
};

/// The pressure Schur preconditioner of the Oseen schemes.
enum class OseenSchur
{
   /// OseenPcdSchurPC: PCD (or, with pcd_convection off, Cahouet-Chabard with
   /// a constant nu + gamma).
   Pcd,
   /// bessemer's CahouetChabardSchurPC with the grad-div weighting (the
   /// pressure mass by 1/(nu + gamma(x))): with a strong grad-div term, the
   /// augmented-Lagrangian Schur approximation.
   CahouetChabardGradDiv
};

/// Options of OseenStepper.
struct OseenStepperOptions
{
   double nu = 1.0;          ///< Kinematic viscosity.
   double dt = 1e-2;         ///< Step size (fixed).
   OseenScheme scheme = OseenScheme::SemiImplicit;
   double lin_rtol = 1e-6;   ///< FGMRES tolerance, semi-implicit.
   /// FGMRES tolerance of each Picard iteration (implicit).
   double picard_lin_rtol = 1e-1;
   /// Picard stopping: ||r|| <= nl_rtol ||r_0||; <= 0: exactly nl_maxit
   /// iterations, not counted as failures.
   double nl_rtol = 1e-3;
   int nl_maxit = 10;        ///< Picard iteration cap.
   int ns_reuse = 10;        ///< Rebuild the LOR operator every this many steps.
   int kdim = 200;           ///< FGMRES restart.
   int max_iter = 500;       ///< FGMRES iteration cap.
   /// Grad-div gamma = grad_div_nu * nu (the recipe's gamma = nu): the
   /// componentwise velocity preconditioner leaves it out, so it must stay
   /// of the viscosity's size.
   double grad_div_nu = 1.0;
   /// Grad-div's part proportional to the element size, gamma += c h_K
   /// (bessemer's grad_div_h; 6 / k_u is its default): 0 = none.
   double grad_div_h = 0.0;
   /// The Schur preconditioner.
   OseenSchur schur = OseenSchur::Pcd;
   /// Give each velocity component's LOR operator its diagonal grad-div block
   /// gamma (d_c u_c, d_c v_c) (with grad-div on).
   bool lor_grad_div = true;
   /// How the velocity preconditioner couples the components:
   /// UpperGaussSeidel (modified AL) keeps the grad-div blocks above the
   /// diagonal; it needs lor_grad_div and a grad-div term.
   OseenVelocityCoupling velocity_coupling = OseenVelocityCoupling::BlockJacobi;
   /// > 0: each component's LOR block solved by GMRES to this tolerance
   /// instead of one V-cycle (diagnostics; see OseenLorPreconditioner).
   double lor_block_rtol = 0.0;
   int lor_block_maxit = 100; ///< Its iteration cap.
   /// The LOR streamline diffusion's factor (OseenLorPreconditioner::Options).
   double lor_sd_scale = 1.0;
   bool lor_half_div = false; ///< 1/2 (div w) u in the LOR operator.
   bool oss = true;          ///< OSS stabilization (momentum and N_p).
   OseenAmgMode amg_mode = OseenAmgMode::Auto; ///< The velocity V-cycle.
   int n_inner = 2;          ///< PCD: CG iterations on A_p.
   /// PCD's convection N_p; false: Cahouet-Chabard (comparisons).
   bool pcd_convection = true;
   /// Stabilize N_p (with oss); false: N_p without OSS (diagnostics).
   bool pcd_oss = true;
   /// > 0: the velocity block solved by an inner FGMRES on F to this
   /// relative tolerance (preconditioned by the V-cycles) instead of one
   /// V-cycle -- a diagnostic that takes the velocity preconditioner out of
   /// the outer iteration count.
   double velocity_inner_rtol = 0.0;
   int velocity_inner_maxit = 100; ///< Its iteration cap.
   /// A constant body force per unit mass (dim values; empty: none), e.g. a
   /// channel's mean pressure gradient.
   std::vector<double> body_force;
   int print_level = -1;     ///< FGMRES print level.
};

/**
 * @brief BDF2 (BDF1 first step) for Navier-Stokes in the skew-symmetric
 *        form with OSS stabilization, each step one Oseen solve
 *        (semi-implicit) or Picard iterations (implicit), on the device.
 *
 * Defect-correction form: the guess X = 2 X^n - X^{n-1} carries the
 * Dirichlet data at t^{n+1}; K delta = r with r = b - K X zeroed on the
 * Dirichlet dofs, delta = 0 there, X += delta; tolerances are relative to
 * ||r|| of the guess. The system is [F B^T; B 0] on p~ = -p (BlockStokesPC's
 * convention), F = sigma M + nu K + gamma G + N(w) with the collocated
 * velocity mass; preconditioned by BlockStokesPC (upper triangular) with
 * OseenLorPreconditioner and OseenPcdSchurPC, in FGMRES.
 */
class OseenStepper
{
public:
   /// What the last step did.
   struct Stats
   {
      int linear_iterations = 0; ///< FGMRES iterations, all Picard steps.
      int picard_iterations = 0; ///< Oseen solves.
      bool converged = true;     ///< Every FGMRES solve converged.
      double r0 = 0.0;           ///< ||r|| of the extrapolated guess.
      double r_final = 0.0;      ///< ||r|| after the last correction.
      bool air = false;          ///< The velocity V-cycle was AIR.
      double cell_re = 0.0;      ///< Its cell Reynolds number.
      double t_lor = 0.0;        ///< LOR assembly + V-cycle setup (s; 0 if reused).
      double t_step = 0.0;       ///< The whole step (s).
      long velocity_inner_iterations = 0; ///< Inner velocity FGMRES iterations.
      long vcycles = 0;          ///< Velocity V-cycles, all components.
   };

   /**
    * @param spaces Mixed spaces (borrowed).
    * @param rules  Quadrature source (borrowed).
    * @param bc     Boundary conditions (borrowed; SetTime is driven here).
    * @param opts   Options.
    */
   OseenStepper(MixedSpaces& spaces, const RuleBook& rules,
                BoundaryConditions& bc, const OseenStepperOptions& opts);
   ~OseenStepper();

   /// u^0 (t = 0); p^0 = 0.
   void SetInitialVelocity(mfem::VectorCoefficient& u0);

   /// One step: t -> t + dt.
   void Step();

   double Time() const { return t_; }            ///< Current time.
   int StepCount() const { return step_; }       ///< Steps taken.
   const Stats& LastStats() const { return stats_; } ///< The last step.
   mfem::ParGridFunction& Velocity() { return u_; }  ///< u^n.
   mfem::ParGridFunction& Pressure() { return p_; }  ///< p^n (physical).

   /**
    * @brief The momentum residual of the last step on velocity true dofs,
    *        unconstrained: r = F u + B^T p~ - b with the operator the step
    *        ended with -- minus the reaction at the Dirichlet dofs, what
    *        BodyForce turns into lift and drag.
    */
   void MomentumResidual(mfem::Vector& r) const;

private:
   /// The wind-dependent operators for wind true dofs @p W.
   void SetWind(const mfem::Vector& W);
   /// R = b - K X (Dirichlet rows of the momentum part zeroed).
   void Residual(const mfem::Vector& X, mfem::Vector& R) const;
   /// Global 2-norm.
   double Norm(const mfem::Vector& v) const;
   /// (Re)build the block operator after the momentum block changed.
   void BuildSystem();

   MixedSpaces& spaces_;
   const RuleBook& rules_;
   BoundaryConditions& bc_;
   OseenStepperOptions opts_;
   int nu_ = 0, np_ = 0, dim_ = 0;
   mfem::Array<int> offsets_;
   mfem::Array<int> ess_;
   std::unique_ptr<StokesOperator> op_;
   std::unique_ptr<mfem::ParFiniteElementSpace> sfes_; ///< One component.
   std::unique_ptr<OseenWindData> wind_;
   std::unique_ptr<ScalarOseenOperator> nscalar_;
   std::unique_ptr<ComponentwiseOperator> N_;
   std::unique_ptr<mfem::ConstrainedOperator> Nc_;
   std::unique_ptr<mfem::SumOperator> F_, Func_;
   std::unique_ptr<mfem::TransposeOperator> Bt_;
   std::unique_ptr<mfem::BlockOperator> sys_;
   std::unique_ptr<OseenLorPreconditioner> lor_pc_;
   std::unique_ptr<OseenPcdSchurPC> pcd_;
   std::unique_ptr<CahouetChabardSchurPC>
   cc_; ///< OseenSchur::CahouetChabardGradDiv.
   std::unique_ptr<mfem::Solver> vel_count_;   ///< Counts the inner iterations.
   mutable long vel_inner_its_ = 0;
   std::unique_ptr<mfem::FGMRESSolver> vel_inner_; ///< velocity_inner_rtol > 0.
   std::unique_ptr<BlockStokesPC> pc_;
   std::unique_ptr<mfem::FGMRESSolver> fgmres_;
   mfem::ParGridFunction u_, p_, wind_gf_;
   mfem::Vector X_, Xn_, Xnm1_, R_, D_, b_, hist_, W_;
   mfem::Vector forcing_; ///< M f for the constant body force (empty: none).
   double t_ = 0.0, c0_ = -1.0;
   int step_ = 0, lor_age_ = 0;
   Stats stats_;
};

} // namespace incns
