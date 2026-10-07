/**
 * @file lp_surrogate.hpp
 * @brief Inner Poisson preconditioner for the Cahouet-Chabard Schur block:
 *        fixed LOR-AMG V-cycle(s) on the pressure Laplacian L_p
 *        (SPEC_cahouet_chabard_mfem.md par.6.2; docs/precond_cc.md).
 *
 * L_p appears ONLY as the preconditioner of the inner CG on B M_v^-1 B^T --
 * never as the Schur surrogate itself (that mistake is exactly the
 * LaplacianLegacy comparison mode). The operator is kept singular in the
 * enclosed/periodic case: no pinning, no shift, ever (repo law); singularity
 * is handled by the Ortho wrapper + projected RHS/iterates.
 */
#pragma once

#include "mfem.hpp"

#include <memory>
#include <stdexcept>

namespace incns
{

/// Boundary conditions of the inner-PC pressure Laplacian.
enum class LpBC
{
   Auto,            ///< Dirichlet on outflow attributes if any, else all-Neumann.
   AllNeumann,      ///< Pure Neumann (singular; enclosed/periodic domains).
   DirichletOnAttrs ///< Homogeneous Dirichlet on a caller-given attribute list.
};

/**
 * @brief Fixed V-cycle(s) of BoomerAMG on the low-order-refined pressure
 *        Laplacian -- a fixed, SYMMETRIC linear operator (CG-legal).
 *
 * The high-order Diffusion form on the pressure space is handed to
 * mfem::LORSolver<HypreBoomerAMG> (Q1-on-GLL-nodes rediscretization, the same
 * machinery as the H5 velocity block). CG legality of the inner solve demands
 * a symmetric application, so the relaxation is SET explicitly (l1 hybrid
 * symmetric Gauss-Seidel, matched single pre/post sweeps) rather than trusting
 * library defaults. On singular (all-Neumann) operators the application is
 * wrapped in mfem::OrthoSolver so iterates never accumulate the constant mode.
 *
 * MEASURED DECISION (deviation from the spec's par.4 suggestion): the spec
 * proposes forcing a smoother-only coarsest solve on the singular operator.
 * Any HYPRE_BoomerAMGSetCycleRelaxType override (types 8 and 18 both tried)
 * makes hypre's setup fail with an argument error on degenerate tiny 3D
 * hierarchies (e.g. a 27-dof Q1 LOR L_p, where coarsening collapses), while
 * hypre's DEFAULT coarse solve is harmless in practice: as a preconditioner
 * its output rides inside the Ortho wrap with projected RHS/iterates, which
 * is what actually protects the solve (T2b demonstrates projections are the
 * load-bearing part -- removing THEM breaks the solve outright). So the
 * coarse solve is left at hypre's default, deliberately.
 *
 * The application is @c lp_vcycles V-cycles from a zero initial guess -- a
 * fixed linear operator per application, never a tolerance-based solve.
 */
class LpSurrogate : public mfem::Solver
{
public:
   /**
    * @brief Assemble the LOR L_p and set up the AMG.
    * @param pfes          Pressure space (borrowed; must outlive this).
    * @param outflow_attrs Boundary attributes carrying outflow/traction BCs
    *                      (used by LpBC::Auto; empty means none).
    * @param singular      Constant pressure mode present (enclosed/periodic)?
    *                      Drives the Ortho wrap + coarse-solve choice.
    * @param bc_mode       Boundary-condition policy (see LpBC).
    * @param dirichlet_attrs Attribute list for LpBC::DirichletOnAttrs.
    * @param vcycles       V-cycles per application (fixed count, default 1).
    * @throws std::invalid_argument on inconsistent configuration (e.g.
    *         DirichletOnAttrs with an empty list, or a singular flag combined
    *         with Dirichlet BCs -- Dirichlet removes the constant mode).
    */
   LpSurrogate(mfem::ParFiniteElementSpace& pfes,
               const mfem::Array<int>& outflow_attrs, bool singular,
               LpBC bc_mode = LpBC::Auto,
               const mfem::Array<int>& dirichlet_attrs = mfem::Array<int>(),
               int vcycles = 1)
      : mfem::Solver(pfes.GetTrueVSize()), form_(&pfes),
        ortho_(pfes.GetComm())
   {
      iterative_mode = false;
      if (vcycles < 1)
      {
         throw std::invalid_argument("lp_surrogate: vcycles must be >= 1");
      }

      // --- resolve the Dirichlet attribute set per the BC policy ------------
      mfem::ParMesh& mesh = *pfes.GetParMesh();
      const int max_attr =
         mesh.bdr_attributes.Size() ? mesh.bdr_attributes.Max() : 0;
      mfem::Array<int> marker(max_attr);
      marker = 0;
      auto mark = [&](const mfem::Array<int>& attrs)
      {
         for (int a : attrs)
         {
            if (a >= 1 && a <= max_attr) { marker[a - 1] = 1; }
         }
      };
      switch (bc_mode)
      {
         case LpBC::Auto:        mark(outflow_attrs); break;
         case LpBC::AllNeumann:  break;
         case LpBC::DirichletOnAttrs:
            if (dirichlet_attrs.Size() == 0)
            {
               throw std::invalid_argument(
                  "lp_surrogate: DirichletOnAttrs needs a non-empty list");
            }
            mark(dirichlet_attrs);
            break;
      }
      mfem::Array<int>& ess_tdofs = ess_tdofs_;
      if (max_attr > 0) { pfes.GetEssentialTrueDofs(marker, ess_tdofs); }

      // Consistency: Dirichlet dofs remove the constant mode -- a singular
      // wrap around a nonsingular operator (or vice versa) is a config bug.
      int has_dirichlet = ess_tdofs.Size() > 0 ? 1 : 0;
      MPI_Allreduce(MPI_IN_PLACE, &has_dirichlet, 1, MPI_INT, MPI_MAX,
                    pfes.GetComm());
      if (singular && has_dirichlet)
      {
         throw std::invalid_argument(
            "lp_surrogate: singular flag with Dirichlet pressure BCs -- "
            "Dirichlet removes the constant mode; the flags disagree");
      }

      // --- LOR L_p: DiffusionIntegrator on the HO pressure space ------------
      form_.AddDomainIntegrator(new mfem::DiffusionIntegrator);
      lor_ = std::make_unique<mfem::LORSolver<mfem::HypreBoomerAMG>>(
                form_, ess_tdofs);

      // --- CG legality + singular-safety knobs, set EXPLICITLY --------------
      mfem::HypreBoomerAMG& amg = lor_->GetSolver();
      amg.SetPrintLevel(0);
      amg.iterative_mode = false;
      amg.SetMaxIter(vcycles); // n V-cycles from zero guess: still a fixed op
      HYPRE_Solver h = static_cast<HYPRE_Solver>(amg);
      // l1 hybrid symmetric Gauss-Seidel relaxation, one matched sweep on the
      // way down and up: a SYMMETRIC V-cycle (required under the inner CG).
      HYPRE_BoomerAMGSetRelaxType(h, 8);
      HYPRE_BoomerAMGSetNumSweeps(h, 1);
      // Strength-of-connection starting points per the spec (2D/3D).
      HYPRE_BoomerAMGSetStrongThreshold(h, mesh.Dimension() == 3 ? 0.25 : 0.1);

      singular_ = singular;
      if (singular_) { ortho_.SetSolver(*lor_); }
   }

   /**
    * @brief Apply the fixed V-cycle(s); Ortho-wrapped when singular.
    * @param x Pressure-space residual (true dofs).
    * @param y Preconditioned output (true dofs).
    */
   void Mult(const mfem::Vector& x, mfem::Vector& y) const override
   {
      if (singular_) { ortho_.Mult(x, y); }
      else { lor_->Mult(x, y); }
   }

   /// Required by mfem::Solver; fixed at construction.
   void SetOperator(const mfem::Operator&) override {}

   /// @return True if the application is Ortho-wrapped (singular operator).
   bool Singular() const { return singular_; }

   /// @return The Dirichlet (essential) true dofs of L_p -- the outflow
   ///         pressure dofs under LpBC::Auto; empty when all-Neumann.
   const mfem::Array<int>& EssentialTrueDofs() const { return ess_tdofs_; }

private:
   mfem::Array<int> ess_tdofs_;  ///< Dirichlet true dofs of L_p.
   mfem::ParBilinearForm form_;  ///< HO Diffusion form (LOR source; owned).
   /// LOR-AMG on the rediscretized L_p (owned; must outlive applications).
   std::unique_ptr<mfem::LORSolver<mfem::HypreBoomerAMG>> lor_;
   mutable mfem::OrthoSolver ortho_; ///< Constant-mode wrap (singular only).
   bool singular_ = false;           ///< Ortho-wrapped?
};

} // namespace incns
