/**
 * @file cahouet_chabard.hpp
 * @brief The Cahouet-Chabard Schur preconditioner (consistent BM_v^-1 B^T
 *        variant) and its configuration (SPEC_cahouet_chabard_mfem.md par.2.3,
 *        par.6, par.7; docs/precond_cc.md).
 */
#ifndef INCNS_PRECOND_CAHOUET_CHABARD_HPP
#define INCNS_PRECOND_CAHOUET_CHABARD_HPP

#include "precond/lp_surrogate.hpp"
#include "precond/mass_inverse.hpp"
#include "precond/mixed_poisson_op.hpp"
#include "precond/nullspace.hpp"
#include "quadrature/rule_book.hpp"
#include "mfem.hpp"

#include <memory>

namespace incns
{

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
/// A-block preconditioner choice (v1 default LORAMG; PMG deferred -- human
/// decision 2026-07-16).
enum class APC { PMGChebyshev, LORAMG, JacobiChebyshev };
/// Nullspace handling mode.
enum class NullspaceMode { Auto, ForceOn, ForceOff };
/// Preconditioner arithmetic precision (v2 hook; v1: FP64 only).
enum class PcPrecision { FP64, FP32PC };

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
   MassInvType mv_inv = MassInvType::Auto; ///< Velocity mass-inverse strategy.
   int k_mv_chebyshev = 4;                 ///< Chebyshev order for M_v^-1.
   MassInvType mp_inv = MassInvType::Auto; ///< Pressure mass-inverse strategy.
   int k_mp_chebyshev = 3;                 ///< Chebyshev order for M_p^-1.
   InnerStop inner_stop = InnerStop::FixedIters; ///< Inner CG stopping rule.
   int n_inner = 10;        ///< Inner CG iterations (FixedIters).
   double tol_inner = 1e-3; ///< Inner CG relative tolerance (RelTol).
   LpPC lp_pc = LpPC::LORAMG; ///< Inner Poisson PC (v1: LORAMG only).
   int lp_vcycles = 1;        ///< V-cycles per L_p application.
   LpBC lp_bc = LpBC::Auto;   ///< L_p boundary-condition policy.
   mfem::Array<int> lp_dirichlet_attrs; ///< Attrs for DirichletOnAttrs.
   // --- A-block --------------------------------------------------------------
   APC a_pc = APC::LORAMG; ///< v1 default (spec's PMGChebyshev deferred).
   int a_vcycles = 1;      ///< V-cycles per A-hat application.
   // --- nullspace ------------------------------------------------------------
   NullspaceMode nullspace = NullspaceMode::Auto; ///< Detection override.
   int k_reproj = 5; ///< Spec parity; the implementation projects every apply.
   bool allow_pin_dof = false; ///< Debug-only; warns loudly; NO-OP in v1.
   // --- outer solver ---------------------------------------------------------
   double outer_rtol = 1e-8; ///< FGMRES relative tolerance.
   double outer_atol = 0.0;  ///< FGMRES absolute tolerance.
   int outer_maxit = 200;    ///< FGMRES iteration cap.
   int fgmres_restart = 100; ///< FGMRES restart (Krylov subspace) size.
   // --- execution ------------------------------------------------------------
   PcPrecision pc_precision = PcPrecision::FP64; ///< v1: FP64 only.
   int verbosity = 1;             ///< 0 silent .. 3 inner traces.
   bool collect_spectrum = false; ///< T3 Lanczos hooks (reserved).

   /**
    * @brief Fail-fast validation (SPEC par.7). Throws std::invalid_argument
    *        with a clear message; warns (rank 0) on LaplacianLegacy and
    *        allow_pin_dof.
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
    * @param pfes          Pressure space (borrowed).
    * @param rules         Quadrature source (borrowed; M_p rule).
    * @param B             ELIMINATED divergence operator (borrowed).
    * @param Mv            Velocity mass operator on true dofs (borrowed).
    * @param mv_diag       Assembled diagonal of @p Mv (borrowed).
    * @param outflow_attrs Boundary attributes carrying outflow BCs.
    * @param singular_auto Detection result (BoundaryConditions::
    *                      PressureNullspaceExists) -- used when
    *                      cfg.nullspace == Auto.
    */
   CahouetChabardSchurPC(const CahouetChabardConfig& cfg,
                         mfem::ParFiniteElementSpace& pfes,
                         const RuleBook& rules, const mfem::Operator& B,
                         const mfem::Operator& Mv, const mfem::Vector& mv_diag,
                         const mfem::Array<int>& outflow_attrs,
                         bool singular_auto);

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

   /// @return True if the singular (projected) path is active.
   bool Singular() const { return singular_; }

   /// @return The resolved nu_pc scaling of the mass term.
   double NuPc() const { return nu_pc_; }

private:
   CahouetChabardConfig cfg_;   ///< Configuration (copied).
   bool singular_;              ///< Resolved nullspace flag.
   double sigma_;               ///< gamma0/dt (Reset updates).
   double nu_pc_;               ///< Resolved mass-term scaling (Reset updates).

   mfem::ParBilinearForm mp_form_;      ///< Pressure mass form (owned).
   mfem::OperatorPtr Mp_;               ///< Pressure mass on true dofs.
   mfem::Vector mp_diag_;               ///< Its assembled diagonal.
   std::unique_ptr<MassInverse> mp_inv_; ///< M_p^-1 (fixed op).
   std::unique_ptr<MassInverse> mv_inv_; ///< M_v^-1 (fixed op, inside BMB).
   std::unique_ptr<MixedPoissonOperator> bmb_; ///< B M_v^-1 B^T (matrix-free).
   std::unique_ptr<ConstantPressureProjector> proj_; ///< l2 projector.
   std::unique_ptr<ProjectedOperator> bmb_proj_; ///< P.S (singular CG operator).
   std::unique_ptr<LpSurrogate> lp_;    ///< Inner Poisson PC (LOR-AMG).
   std::unique_ptr<mfem::CGSolver> inner_cg_; ///< The inner CG (fixed iters).

   mutable mfem::Vector z_mass_, rhs_, w_; ///< Pressure-sized scratch (device).
};

} // namespace incns

#endif // INCNS_PRECOND_CAHOUET_CHABARD_HPP
