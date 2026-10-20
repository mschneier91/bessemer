/**
 * @file oseen_pcd.hpp
 * @brief The pressure Schur preconditioner of the Oseen schemes: PCD
 *        (pressure convection-diffusion, Elman-Tuminaro form) with
 *        A_p = B M_v^-1 B^T (docs/design/oseen_bdf2.md).
 */
#pragma once

#include "oseen/oseen_convection.hpp"
#include "oseen/oseen_wind.hpp"
#include "precond/lp_surrogate.hpp"
#include "precond/mass_inverse.hpp"
#include "precond/mixed_poisson_op.hpp"
#include "precond/nullspace.hpp"
#include "quadrature/rule_book.hpp"
#include "mfem.hpp"

#include <memory>

namespace incns
{

/**
 * @brief @f$ \tilde S^{-1} r = A_p^{-1} (\sigma r + N_p M_p^{-1} r) +
 *        \tilde\nu M_p^{-1} r @f$, i.e. @f$ A_p^{-1} F_p M_p^{-1} @f$ with
 *        @f$ F_p = \sigma M_p + \tilde\nu A_p + N_p @f$, @f$ \tilde\nu = \nu +
 *        \gamma @f$ (grad-div).
 *
 * - A_p = B M_v^-1 B^T with the collocated (diagonal) velocity mass and the
 *   Dirichlet-eliminated B: no artificial pressure boundary conditions. Applied
 *   by a fixed number of CG iterations, matrix-free, preconditioned by one
 *   LOR-AMG V-cycle on a pressure Laplacian (the Cahouet-Chabard machinery:
 *   MixedPoissonOperator, LpSurrogate). The recipe assembles A_p by SpGEMM;
 *   at Q5 in 3D that matrix would be far too large.
 * - N_p: the pressure-space Oseen operator (ScalarOseenOperator: convection,
 *   1/2 div w, OSS with the velocity order's tau), plus the inflow term
 *   -((w.n)_- p, q) on the boundary where w.n < 0 (host assembly). Without
 *   the stabilization the recipe's outer counts rose up to 5.4x.
 * - M_p^-1: Chebyshev on the consistent pressure mass, Jacobi-preconditioned.
 *
 * Positive (BlockStokesPC owns the minus of the pressure row). Nonlinear in
 * its input (inner CG): the outer Krylov must be flexible.
 */
class OseenPcdSchurPC : public mfem::Solver
{
public:
   /// Options.
   struct Options
   {
      double nu = 1.0;       ///< Kinematic viscosity (tau).
      double nu_tilde = 1.0; ///< nu + gamma, the mass term's coefficient.
      int n_inner = 2;       ///< CG iterations on A_p.
      int mp_chebyshev = 3;  ///< Chebyshev order of M_p^-1.
      bool oss = true;       ///< Stabilize N_p.
      bool inflow_term = true; ///< The -((w.n)_- p, q) boundary term.
   };

   /**
    * @param vfes          Velocity space (borrowed).
    * @param pfes          Pressure space (borrowed).
    * @param rules         Quadrature source (borrowed).
    * @param B             The Dirichlet-eliminated divergence (borrowed).
    * @param outflow_attrs Outflow boundary attributes (the L_p surrogate).
    * @param singular      The pressure has a constant null space.
    * @param opts          Options.
    */
   OseenPcdSchurPC(mfem::ParFiniteElementSpace& vfes,
                   mfem::ParFiniteElementSpace& pfes, const RuleBook& rules,
                   const mfem::Operator& B,
                   const mfem::Array<int>& outflow_attrs, bool singular,
                   const Options& opts);
   ~OseenPcdSchurPC() override;

   /// New wind (N_p) and mass factor sigma.
   void Update(const mfem::ParGridFunction& wind, double sigma);

   void Mult(const mfem::Vector& r, mfem::Vector& z) const override;
   void SetOperator(const mfem::Operator&) override {}

private:
   mfem::ParFiniteElementSpace& vfes_;
   mfem::ParFiniteElementSpace& pfes_;
   Options opts_;
   bool singular_;
   double sigma_ = 0.0;
   std::unique_ptr<mfem::ParBilinearForm> mp_form_, mv_form_;
   mfem::OperatorHandle Mp_, Mv_;
   mfem::Vector mp_diag_, mv_diag_;
   std::unique_ptr<MassInverse> mp_inv_, mv_inv_;
   std::unique_ptr<MixedPoissonOperator> bmb_;
   std::unique_ptr<ConstantPressureProjector> proj_;
   std::unique_ptr<ProjectedOperator> bmb_proj_;
   std::unique_ptr<LpSurrogate> lp_;
   std::unique_ptr<mfem::CGSolver> cg_;
   std::unique_ptr<OseenWindData> wind_p_;
   std::unique_ptr<ScalarOseenOperator> np_;
   std::unique_ptr<mfem::HypreParMatrix> inflow_;
   mfem::Array<int> dirichlet_bdr_; ///< Boundary attributes not outflow.
   mutable mfem::Vector t_, s_, w_;
};

} // namespace incns
