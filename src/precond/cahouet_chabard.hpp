/**
 * @file cahouet_chabard.hpp
 * @brief The Cahouet-Chabard Schur preconditioner (consistent BM_v^-1 B^T
 *        variant) and its configuration (docs/design/SPEC_cahouet_chabard_mfem.md par.2.3,
 *        par.6, par.7; docs/precond_cc.md).
 */
#pragma once

#include "precond/lp_surrogate.hpp"
#include "precond/mass_inverse.hpp"
#include "precond/mixed_poisson_op.hpp"
#include "precond/nullspace.hpp"
#include "quadrature/rule_book.hpp"
#include "util/profiled_operator.hpp"
#include "mfem.hpp"

#include <memory>

namespace incns
{

/// Which pressure Schur block the Stokes solver uses (declared here so the
/// Parameters layer can reference it without pulling in the solver headers).
enum class SchurBlockType
{
   Mass,          ///< Sprint-1 default: S_hat^-1 = nu M_p^-1, block-diag PC.
   CahouetChabard ///< CC per docs/design/SPEC_cahouet_chabard_mfem.md (block-tri default);
   ///<              cc.schur_model picks ConsistentBMB vs LaplacianLegacy.
};

/// Shape of the block preconditioner on [A B^T; B 0] (CC.6).
enum class BlockPCShape { Diag, LowerTri, UpperTri };
/// Schur approximate-inverse model.
enum class SchurModel
{
   ConsistentBMB,   ///< nu M_p^-1 + sigma (B M_v^-1 B^T)^-1 -- the default.
   LumpedBMB,       ///< REJECTED in v1: with collocated GLL the consistent
   ///<               M_v is already diagonal; AbsLumped has no callers.
   LaplacianLegacy  ///< nu M_p^-1 + sigma L_p^-1 -- comparison mode ONLY (T5).
};
/// Viscous form of the SYSTEM operator (v1: Laplacian only).
enum class ViscousForm { Laplacian, SymGradient };
/// Pressure-mass coefficient model (v2 hook; v1: constant nu only).
enum class PcMassCoeff { ConstNu, ReciprocalNuField };
/// Inner CG stopping rule.
enum class InnerStop { FixedIters, RelTol };
/// Inner Poisson PC choice (v1: LOR-AMG only; PMG deferred with p-MG).
enum class LpPC { LORAMG, PMG };
/// A-block (velocity) preconditioner choice. PMG deferred (human decision
/// 2026-07-16). LORAMG: one LOR-BoomerAMG V-cycle -- robust when viscosity
/// dominates the block. JacobiChebyshev: a fixed order-4 Chebyshev polynomial
/// in the Jacobi-preconditioned block -- cheap and near-exact when the mass
/// term dominates (small steps). JacobiPCG: CG with Jacobi to a loose
/// tolerance (a_pcg_rtol, at most a_pcg_max_iter) -- adapts its work to the
/// regime; nonlinear, so only inside the flexible outer Krylov.
enum class APC { PMGChebyshev, LORAMG, JacobiChebyshev, JacobiPCG };
/// What preconditions the inner CG of APC::JacobiPCG. Auto: point-block
/// Jacobi when the velocity block has a grad-div term (its nodal coupling of
/// the components is then in the blocks), scalar Jacobi otherwise.
enum class APcgPC { Auto, Jacobi, PointBlockJacobi };
/// Nullspace handling mode.
enum class NullspaceMode { Auto, ForceOn, ForceOff };
/// Preconditioner arithmetic precision (v2 hook; v1: FP64 only).
enum class PcPrecision { FP64, FP32PC };
/// Quadrature of the PRECONDITIONER's internal mass matrices (SPEC par.3).
enum class PcQuadrature
{
   Inherit,      ///< Use the system's mass operators (default -- calibration
   ///<             matches the operator actually being solved).
   GllCollocated ///< The PC builds its OWN collocated-GLL velocity/pressure
   ///<             masses (exactly diagonal -> bare multiplies inside
   ///<             BM_v^-1B^T). The SYSTEM mass is untouched -- this cheapens
   ///<             only the preconditioner; requires the GLL nodal H1 basis.
};

/**
 * @brief Configuration of the CC preconditioner stage (SPEC par.7).
 *
 * The outer solver is FGMRES on the monolithic system -- not configurable.
 * Validate() throws std::invalid_argument (identically on every rank) on any
 * inconsistent or not-in-v1 setting; reserved enum values exist so decks stay
 * forward-compatible, but requesting one is a hard error, never a silent
 * fallback.
 */
struct CahouetChabardConfig
{
   // --- structure -----------------------------------------------------------
   BlockPCShape block_shape = BlockPCShape::UpperTri; ///< Block PC shape.
   SchurModel schur_model = SchurModel::ConsistentBMB; ///< Schur model.
   // --- physics / scaling ---------------------------------------------------
   double sigma = 0.0;    ///< gamma0/dt; 0 = steady Stokes (mass-only PC).
   double nu = 1.0;       ///< Constant viscosity of the system.
   /// <0 resolves to nu (for BOTH viscous forms -- C&G (4.4a)); an O(1)
   /// calibration dial around nu, NEVER a function of dt (scaling guard).
   double nu_pc = -1.0;
   ViscousForm visc_form = ViscousForm::Laplacian;   ///< System viscous form.
   PcMassCoeff pc_mass_coeff = PcMassCoeff::ConstNu; ///< v1: ConstNu only.
   // --- inner components ----------------------------------------------------
   /// PC-internal mass quadrature: Inherit the system mass (default) or build
   /// collocated-GLL (diagonal) masses for the PC only.
   PcQuadrature pc_quadrature = PcQuadrature::Inherit;
   MassInvType mv_inv = MassInvType::Auto; ///< Velocity mass-inverse strategy.
   int k_mv_chebyshev = 4;                 ///< Chebyshev order for M_v^-1.
   MassInvType mp_inv = MassInvType::Auto; ///< Pressure mass-inverse strategy.
   int k_mp_chebyshev = 3;                 ///< Chebyshev order for M_p^-1.
   InnerStop inner_stop = InnerStop::FixedIters; ///< Inner CG stopping rule.
   int n_inner = 10;        ///< Inner CG iterations (FixedIters).
   double tol_inner = 1e-3; ///< Inner CG relative tolerance (RelTol).
   LpPC lp_pc = LpPC::LORAMG; ///< Inner Poisson PC (v1: LORAMG only).
   int lp_vcycles = 1;        ///< V-cycles per L_p application.
   /// BoomerAMG strength threshold of the L_p AMG; < 0: 0.25 in 3D, 0.1 in
   /// 2D. Larger values give sparser coarse operators (less memory).
   double lp_strong_threshold = -1.0;
   /// Levels of aggressive coarsening in the L_p AMG (0 = none). Each cuts
   /// the coarse operators' size, at some cost in convergence.
   int lp_agg_levels = 0;
   int lp_print_level = 0; ///< hypre's print level for the L_p AMG setup.
   LpBC lp_bc = LpBC::Auto;   ///< L_p boundary-condition policy.
   mfem::Array<int> lp_dirichlet_attrs; ///< Attrs for DirichletOnAttrs.
   // --- A-block --------------------------------------------------------------
   APC a_pc = APC::LORAMG; ///< v1 default (spec's PMGChebyshev deferred).
   int a_vcycles = 1;      ///< V-cycles per A-hat application.
   double a_pcg_rtol = 1e-2; ///< JacobiPCG: inner relative tolerance.
   int a_pcg_max_iter = 50;  ///< JacobiPCG: inner iteration cap.
   APcgPC a_pcg_pc = APcgPC::Auto; ///< JacobiPCG: the inner CG's preconditioner.
   /// Put the grad-div coefficient in the Schur PC's mass term (the solver
   /// then calls CahouetChabardSchurPC::SetGradDiv). Off here so the solver
   /// tests' baselines pin what they test; the case-level default is on.
   bool schur_grad_div = false;
   // --- nullspace ------------------------------------------------------------
   NullspaceMode nullspace = NullspaceMode::Auto; ///< Detection override.
   int k_reproj = 5; ///< Spec parity; the implementation projects every apply.
   // NOTE (deliberate deletions vs SPEC par.7, human decision 2026-07-17):
   //  - no outer_* / fgmres_restart fields: the outer FGMRES knobs are
   //    StokesSolverOptions' (rtol/atol/max_iter/kdim) -- one source of truth;
   //  - no allow_pin_dof: pressure pinning is forbidden in this codebase and
   //    was never implemented, so the flag had nothing to enable.
   // --- execution ------------------------------------------------------------
   PcPrecision pc_precision = PcPrecision::FP64; ///< v1: FP64 only.
   int verbosity = 1;             ///< 0 silent .. 3 inner traces.
   bool collect_spectrum = false; ///< T3 Lanczos hooks (reserved).

   /**
    * @brief Fail-fast validation (SPEC par.7). Throws std::invalid_argument
    *        with a clear message; warns (rank 0) on LaplacianLegacy.
    * @param root True on the printing rank (for warnings).
    */
   void Validate(bool root) const;
};

/**
 * @brief The CC Schur approximate inverse @f$ P_S^{-1} = \nu_{pc} M_p^{-1} +
 *        \sigma\,(B M_v^{-1} B^T)^{-1} @f$ as an mfem::Solver (SPEC par.6.1).
 *
 * The Poisson term is an inner CG with a FIXED iteration count (default) on
 * the matrix-free MixedPoissonOperator, preconditioned by one symmetric
 * LOR-AMG V-cycle on L_p -- which makes this preconditioner NONLINEAR in its
 * input: the outer Krylov must be flexible (FGMRES), and no symmetry test
 * applies to this class. sigma = 0 short-circuits to the pure mass PC.
 *
 * On singular domains (enclosed / periodic) the inner solve's RHS is projected
 * onto the constant-mode complement, the CG operator is P.S (projected every
 * application -- stronger than the spec's every-k_reproj re-orthogonalization),
 * and the AMG is Ortho-wrapped inside LpSurrogate.
 *
 * The sign convention: this class applies the POSITIVE P_S^-1; the minus of
 * the block row [0 -S_hat] belongs to the block-triangular wrapper (CC.6),
 * where it appears exactly once.
 *
 * Reset(sigma, nu) updates only the two scalars -- the mass diagonals /
 * Chebyshev setups, B, and the L_p AMG hierarchy are sigma/nu-independent and
 * are reused always (SPEC par.9).
 */
class CahouetChabardSchurPC : public mfem::Solver
{
public:
   /**
    * @brief Build the Schur PC from the assembled (eliminated) pieces.
    * @param cfg           Validated configuration (copied; Validate() is
    *                      called again here -- construction is fail-fast).
    * @param vfes          Velocity space (borrowed; used to build the PC's own
    *                      collocated mass when pc_quadrature = GllCollocated).
    * @param pfes          Pressure space (borrowed).
    * @param rules         Quadrature source (borrowed; M_p rule).
    * @param B             ELIMINATED divergence operator (borrowed).
    * @param Mv            Velocity mass operator on true dofs (borrowed).
    * @param mv_diag       Assembled diagonal of @p Mv (borrowed).
    * @param outflow_attrs Boundary attributes carrying outflow BCs.
    * @param singular_auto Detection result (BoundaryConditions::
    *                      PressureNullspaceExists) -- used when
    *                      cfg.nullspace == Auto.
    * @param mv_diag_inv   1 / @p mv_diag when the caller keeps it (borrowed;
    *                      StokesOperator::MassDiagonalInverse()), or null.
    */
   CahouetChabardSchurPC(const CahouetChabardConfig& cfg,
                         mfem::ParFiniteElementSpace& vfes,
                         mfem::ParFiniteElementSpace& pfes,
                         const RuleBook& rules, const mfem::Operator& B,
                         const mfem::Operator& Mv, const mfem::Vector& mv_diag,
                         const mfem::Array<int>& outflow_attrs,
                         bool singular_auto,
                         const mfem::Vector* mv_diag_inv = nullptr);

   /**
    * @brief Apply @f$ z = \nu_{pc} M_p^{-1} r + \sigma\,w @f$, w from the
    *        inner CG on @f$ B M_v^{-1} B^T @f$ (SPEC par.6.1 pseudocode).
    * @param r Pressure-space residual (true dofs).
    * @param z Preconditioned output (true dofs).
    */
   void Mult(const mfem::Vector& r, mfem::Vector& z) const override;

   /// Required by mfem::Solver; structure is fixed at construction.
   void SetOperator(const mfem::Operator&) override {}

   /**
    * @brief Update the (sigma, nu) scalars for a new time step / viscosity.
    *        Everything structural (masses, B, L_p AMG) is reused.
    * @param sigma New gamma0/dt (>= 0; 0 = steady, drops the Poisson term).
    * @param nu    New viscosity (> 0); nu_pc re-resolves if it tracked nu.
    */
   void Reset(double sigma, double nu);

   /**
    * @brief Account for a grad-div term gamma (div u, div v) in the velocity
    *        block: the mass term becomes @f$ M_{1/(\nu_{pc}+\gamma)}^{-1} @f$,
    *        the inverse of the pressure mass weighted by 1/(nu_pc + gamma),
    *        in place of @f$ \nu_{pc} M_p^{-1} @f$.
    *
    * Grad-div acts on exactly the gradient directions B^T p, so the Schur
    * complement's high-frequency plateau is nu + gamma, not nu; with
    * gamma ~ h and a small viscosity the two differ by orders of magnitude
    * (measured 2026-10-12, Taylor-Green Re 1600 at 16^3: 28 outer iterations
    * without this against 7 with no grad-div). For a constant gamma this is
    * nu_pc -> nu_pc + gamma; a gamma that varies from element to element
    * goes into the mass form's coefficient. Same cost per apply.
    * @param gamma The grad-div coefficient (borrowed; must outlive this
    *              object). Reset() reassembles when nu changes.
    */
   void SetGradDiv(mfem::Coefficient& gamma);

   /// @return True if the singular (projected) path is active.
   bool Singular() const { return singular_; }

   /// @return The resolved nu_pc scaling of the mass term.
   double NuPc() const { return nu_pc_; }

   /// @return The current sigma (gamma0/dt).
   double Sigma() const { return sigma_; }

   /// @return The fixed pressure-mass inverse M_p^-1 of the mass term (for
   ///         the rotational Schur preconditioner's tensor mode).
   mfem::Solver& PressureMassInverse() const
   {
      MFEM_VERIFY(mp_inv_, "cc: the plain pressure mass was released");
      return *mp_inv_;
   }

   /**
    * @brief Free the plain pressure mass and its inverse: after SetGradDiv
    *        the mass term uses the weighted one, and only the rotation-aware
    *        Schur PC (PressureMassInverse) reads the plain one.
    * @pre SetGradDiv was called.
    */
   void ReleasePlainMass();

   /**
    * @brief Run B M_v^-1 B^T in two borrowed velocity-sized vectors from now
    *        on and free its own two (MixedPoissonOperator::BorrowScratch).
    *        The caller guarantees they are idle whenever this PC is applied.
    * @param a Scratch (borrowed, velocity true-dof sized, device-aware).
    * @param b A second one (borrowed).
    */
   void BorrowVelocityScratch(mfem::Vector& a, mfem::Vector& b)
   {
      bmb_->BorrowScratch(a, b);
   }

   /// @return The inner Poisson preconditioner (LOR-AMG V-cycle on the
   ///         UNSCALED L_p; for the rotational Schur preconditioner).
   LpSurrogate& Laplacian() const { return *lp_; }

private:
   CahouetChabardConfig cfg_;   ///< Configuration (copied).
   bool singular_;              ///< Resolved nullspace flag.
   double sigma_;               ///< gamma0/dt (Reset updates).
   double nu_pc_;               ///< Resolved mass-term scaling (Reset updates).

   /// PC-own collocated velocity mass (GllCollocated only; owned).
   std::unique_ptr<mfem::ParBilinearForm> mv_form_pc_;
   mfem::OperatorPtr Mv_pc_;            ///< Its true-dof operator.
   mfem::Vector mv_diag_pc_;            ///< Its (exact) diagonal.
   std::unique_ptr<mfem::ParBilinearForm> mp_form_; ///< Pressure mass form.
   mfem::OperatorPtr Mp_;               ///< Pressure mass on true dofs.
   mfem::Vector mp_diag_;               ///< Its assembled diagonal.
   std::unique_ptr<MassInverse> mp_inv_; ///< M_p^-1 (fixed op).
   /// The mass term with grad-div (SetGradDiv): the pressure mass weighted by
   /// 1/(nu_pc + gamma) and its inverse. All null without it.
   mfem::Coefficient* gamma_ = nullptr; ///< Grad-div coefficient (borrowed).
   double nu_weighted_ = -1.0; ///< The nu_pc the weighted mass was built with.
   mfem::ParFiniteElementSpace* pfes_ = nullptr; ///< Pressure space (borrowed).
   const mfem::IntegrationRule* mp_rule_ = nullptr; ///< M_p's rule (RuleBook's).
   std::unique_ptr<mfem::Coefficient> mpw_coeff_; ///< 1/(nu_pc + gamma).
   std::unique_ptr<mfem::ParBilinearForm> mpw_form_; ///< Weighted mass form.
   mfem::OperatorPtr Mpw_;  ///< Weighted mass on true dofs.
   mfem::Vector mpw_diag_;  ///< Its assembled diagonal.
   std::unique_ptr<MassInverse> mpw_inv_; ///< Its inverse.
   /// (Re)assemble the weighted mass for the current nu_pc_.
   void BuildWeightedMass();
   std::unique_ptr<MassInverse> mv_inv_; ///< M_v^-1 (fixed op, inside BMB).
   std::unique_ptr<MixedPoissonOperator> bmb_; ///< B M_v^-1 B^T (matrix-free).
   std::unique_ptr<ConstantPressureProjector> proj_; ///< l2 projector.
   std::unique_ptr<ProjectedOperator> bmb_proj_; ///< P.S (singular CG operator).
   std::unique_ptr<LpSurrogate> lp_;    ///< Inner Poisson PC (LOR-AMG).
   /// Named regions for the profile (no numerics): B and B^T, M_v^-1, the
   /// inner CG's operator and its preconditioner.
   std::unique_ptr<ProfiledOperator> b_prof_;
   std::unique_ptr<ProfiledOperator> mv_inv_prof_; ///< See b_prof_.
   std::unique_ptr<ProfiledOperator> bmb_prof_;    ///< See b_prof_.
   std::unique_ptr<ProfiledOperator> lp_prof_;     ///< See b_prof_.
   std::unique_ptr<mfem::CGSolver> inner_cg_; ///< The inner CG (fixed iters).

   mutable mfem::Vector z_mass_; ///< M_p^{-1} r, the viscous part (device).
   mutable mfem::Vector
   rhs_;    ///< Inner-solve right-hand side, projected (device).
   mutable mfem::Vector
   w_;      ///< (B M_v^{-1} B^T)^{-1} r, the mass part (device).
};

} // namespace incns
