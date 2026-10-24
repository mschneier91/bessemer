/**
 * @file lp_surrogate.hpp
 * @brief Inner Poisson preconditioner for the Cahouet-Chabard Schur block:
 *        fixed LOR-AMG V-cycle(s) on the pressure Laplacian L_p
 *        (docs/design/SPEC_cahouet_chabard_mfem.md par.6.2; docs/precond_cc.md).
 *
 * L_p appears ONLY as the preconditioner of the inner CG on B M_v^-1 B^T --
 * never as the Schur surrogate itself (that mistake is exactly the
 * LaplacianLegacy comparison mode). The operator is kept singular in the
 * enclosed/periodic case: no pinning, no shift, ever (repo law); singularity
 * is handled by the Ortho wrapper + projected RHS/iterates.
 */
#pragma once

#include "precond/hypre_device.hpp"
#include "precond/nullspace.hpp"
#include "util/profiler.hpp"
#include "mfem.hpp"
#include "mfem/fem/lor/lor_batched.hpp"

#include <algorithm>
#include <cmath>
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

/// Index of LOR vertex (i, j, k) of element e in the batched LOR vertex
/// coordinates (byVDIM: sdim x n1^dim x ne, lexicographic, x fastest).
MFEM_HOST_DEVICE inline int LorVertex(int sdim, int n1, int nv, int e, int i,
                                      int j, int k)
{
   return sdim * (nv * e + i + n1 * (j + n1 * k));
}

/**
 * @brief The number of LOR sub-elements, over all ranks, of @p fes's
 *        GLL-node refinement whose Q1 map has a non-positive Jacobian at a
 *        vertex. The batched LOR assembly integrates there, so such a
 *        sub-element gives a negative (semi)definite element matrix.
 *
 * A moved curved mesh can fold sub-elements that its high-order elements
 * survive: FSI2's flag tip (2026-10-23), where L_p's diagonal went
 * non-positive and the V-cycles stopped preconditioning. Collective. 0 for
 * spaces the check does not cover (non-tensor elements, DG, a surface).
 */
inline long long CountFoldedLorElements(mfem::ParFiniteElementSpace& fes)
{
   mfem::ParMesh& mesh = *fes.GetParMesh();
   const int dim = mesh.Dimension();
   int covered = (mfem::UsesTensorBasis(fes) && !fes.IsDGSpace() &&
                  mesh.SpaceDimension() == dim && (dim == 2 || dim == 3))
                 ? 1 : 0;
   MPI_Allreduce(MPI_IN_PLACE, &covered, 1, MPI_INT, MPI_MIN, fes.GetComm());
   if (!covered) { return 0; }
   mfem::Vector x_vert;
   mfem::BatchedLORAssembly::FormLORVertexCoordinates(fes, x_vert);
   const int p = fes.GetMaxElementOrder(), n1 = p + 1;
   const int nv = dim == 2 ? n1 * n1 : n1 * n1 * n1;
   const int nsub = dim == 2 ? p * p : p * p * p;
   const int ne = fes.GetNE(), sdim = dim;
   mfem::Vector folded(ne * nsub);
   folded.UseDevice(true);
   const double* X = x_vert.Read();
   double* F = folded.Write();
   if (dim == 2)
   {
      mfem::forall(ne * nsub, [ = ] MFEM_HOST_DEVICE(int t)
      {
         const int e = t / nsub, s = t % nsub, ix = s % p, iy = s / p;
         double fold = 0.0;
         for (int b = 0; b < 2; ++b)
         {
            for (int a = 0; a < 2; ++a)
            {
               // The edges through vertex (a, b) along xi and eta.
               const int x0 = LorVertex(sdim, n1, nv, e, ix, iy + b, 0);
               const int x1 = LorVertex(sdim, n1, nv, e, ix + 1, iy + b, 0);
               const int y0 = LorVertex(sdim, n1, nv, e, ix + a, iy, 0);
               const int y1 = LorVertex(sdim, n1, nv, e, ix + a, iy + 1, 0);
               const double det = (X[x1] - X[x0]) * (X[y1 + 1] - X[y0 + 1]) -
                                  (X[x1 + 1] - X[x0 + 1]) * (X[y1] - X[y0]);
               if (!(det > 0.0)) { fold = 1.0; }
            }
         }
         F[t] = fold;
      });
   }
   else
   {
      mfem::forall(ne * nsub, [ = ] MFEM_HOST_DEVICE(int t)
      {
         const int e = t / nsub, s = t % nsub;
         const int ix = s % p, iy = (s / p) % p, iz = s / (p * p);
         double fold = 0.0;
         for (int c = 0; c < 2; ++c)
         {
            for (int b = 0; b < 2; ++b)
            {
               for (int a = 0; a < 2; ++a)
               {
                  // The edges through vertex (a, b, c) along xi, eta, zeta.
                  double u[3], v[3], w[3];
                  const int x0 = LorVertex(sdim, n1, nv, e, ix, iy + b, iz + c);
                  const int x1 = LorVertex(sdim, n1, nv, e, ix + 1, iy + b, iz + c);
                  const int y0 = LorVertex(sdim, n1, nv, e, ix + a, iy, iz + c);
                  const int y1 = LorVertex(sdim, n1, nv, e, ix + a, iy + 1, iz + c);
                  const int z0 = LorVertex(sdim, n1, nv, e, ix + a, iy + b, iz);
                  const int z1 = LorVertex(sdim, n1, nv, e, ix + a, iy + b, iz + 1);
                  for (int d = 0; d < 3; ++d)
                  {
                     u[d] = X[x1 + d] - X[x0 + d];
                     v[d] = X[y1 + d] - X[y0 + d];
                     w[d] = X[z1 + d] - X[z0 + d];
                  }
                  const double det = u[0] * (v[1] * w[2] - v[2] * w[1]) -
                                     u[1] * (v[0] * w[2] - v[2] * w[0]) +
                                     u[2] * (v[0] * w[1] - v[1] * w[0]);
                  if (!(det > 0.0)) { fold = 1.0; }
               }
            }
         }
         F[t] = fold;
      });
   }
   long long n = static_cast<long long>(folded.Sum() + 0.5);
   MPI_Allreduce(MPI_IN_PLACE, &n, 1, MPI_LONG_LONG, MPI_SUM, fes.GetComm());
   return n;
}

/**
 * @brief The Laplacian's element matrix with |det J| as the volume: on a
 *        folded element it stays positive semidefinite (MFEM's
 *        DiffusionIntegrator divides by the signed det J). For LpSurrogate's
 *        legacy LOR assembly, which sets the vertex rule as IntRule; |det J|
 *        is floored at 1e-6 of the element's largest (a degenerate vertex).
 */
class AbsDetDiffusionIntegrator : public mfem::BilinearFormIntegrator
{
public:
   void AssembleElementMatrix(const mfem::FiniteElement& el,
                              mfem::ElementTransformation& T,
                              mfem::DenseMatrix& elmat) override
   {
      const int nd = el.GetDof(), dim = el.GetDim();
      dshape_.SetSize(nd, dim);
      gshape_.SetSize(nd, dim);
      elmat.SetSize(nd);
      elmat = 0.0;
      const mfem::IntegrationRule& ir =
         IntRule ? *IntRule : mfem::IntRules.Get(el.GetGeomType(), 2 * el.GetOrder());
      double dmax = 0.0;
      for (int q = 0; q < ir.GetNPoints(); ++q)
      {
         T.SetIntPoint(&ir.IntPoint(q));
         dmax = std::max(dmax, std::fabs(T.Jacobian().Det()));
      }
      if (!(dmax > 0.0)) { return; }
      for (int q = 0; q < ir.GetNPoints(); ++q)
      {
         const mfem::IntegrationPoint& ip = ir.IntPoint(q);
         el.CalcDShape(ip, dshape_);
         T.SetIntPoint(&ip);
         const double det = std::max(std::fabs(T.Jacobian().Det()), 1e-6 * dmax);
         mfem::Mult(dshape_, T.AdjugateJacobian(), gshape_);
         mfem::AddMult_a_AAt(ip.weight / det, gshape_, elmat);
      }
   }

private:
   mfem::DenseMatrix dshape_,
        gshape_; ///< Shape gradients (reference, physical * det).
};

/**
 * @brief Fixed V-cycle(s) of BoomerAMG on the low-order-refined pressure
 *        Laplacian -- a fixed, SYMMETRIC linear operator (CG-legal).
 *
 * The high-order Diffusion form on the pressure space is rediscretized by
 * mfem::ParLORDiscretization (Q1 on the GLL nodes, the same machinery as the
 * H5 velocity block) and BoomerAMG is set up on the assembled matrix. Only the
 * matrix and the hierarchy are kept: the discretization's LOR mesh, space and
 * batched-assembly scratch (27 doubles per element dof) are freed once the
 * matrix is assembled, where mfem::LORSolver would keep them for its lifetime.
 * L_p does not depend on dt or nu, so it is never reassembled.
 *
 * On a moved mesh whose LOR sub-elements fold (CountFoldedLorElements), L_p
 * is assembled instead with |det J| (AbsDetDiffusionIntegrator, MFEM's legacy
 * LOR path), so it stays SPD; without folds the batched path is unchanged.
 *
 * CG legality of the inner solve demands a symmetric application, so the
 * relaxation is SET explicitly (l1 hybrid symmetric Gauss-Seidel on a host,
 * l1-Jacobi on a GPU: SymmetricAmgSmoother; matched single pre/post sweeps)
 * rather than trusting library defaults. On singular (all-Neumann) operators
 * the application is wrapped in DeviceOrthoSolver (precond/nullspace.hpp) so
 * iterates never accumulate the constant mode.
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
               int vcycles = 1, double strong_threshold = -1.0,
               int agg_levels = 0, int print_level = 0)
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
      {
         INCNS_PROFILE("lp::lor_assemble");
         // Folded LOR sub-elements made L_p indefinite (non-positive
         // diagonals) and the V-cycles diverge: FSI2's flow solve stalled
         // at 2e-5 of its start at every dt down to 1e-7 (2026-10-23).
         folded_ = CountFoldedLorElements(pfes);
         mfem::ParBilinearForm abs_form(&pfes); // outlives lor's use of it
         auto lor = std::make_unique<mfem::ParLORDiscretization>(pfes);
         if (folded_ > 0)
         {
            abs_form.AddDomainIntegrator(new AbsDetDiffusionIntegrator);
            lor->AssembleSystem(abs_form, ess_tdofs);
            static bool reported = false;
            if (!reported && pfes.GetMyRank() == 0)
            {
               reported = true;
               mfem::out << "[incns] pressure LOR-AMG: " << folded_
                         << " LOR sub-elements fold on the moved mesh; L_p "
                         "takes |det J| there (reported once)" << std::endl;
            }
         }
         else { lor->AssembleSystem(form_, ess_tdofs); }
         mfem::OperatorHandle& a = lor->GetAssembledSystem();
         if (folded_ > 0)
         {
            // The legacy assembly's matrix belongs to its form: a copy.
            A_ = std::make_unique<mfem::HypreParMatrix>(*a.As<mfem::HypreParMatrix>());
         }
         else
         {
            a.SetOperatorOwner(false); // the matrix outlives the discretization
            A_.reset(a.As<mfem::HypreParMatrix>());
         }
         {
            INCNS_PROFILE("lp::lor_release");
            lor.reset();
         }
      }
      amg_ = std::make_unique<mfem::HypreBoomerAMG>(*A_);

      // --- CG legality + singular-safety knobs, set EXPLICITLY --------------
      mfem::HypreBoomerAMG& amg = *amg_;
      amg.SetPrintLevel(print_level);
      amg.iterative_mode = false;
      amg.SetMaxIter(vcycles); // n V-cycles from zero guess: still a fixed op
      HYPRE_Solver h = static_cast<HYPRE_Solver>(amg);
      // A symmetric smoother, one matched sweep on the way down and up: a
      // SYMMETRIC V-cycle (required under the inner CG). Host: l1 hybrid
      // symmetric Gauss-Seidel (8). GPU: l1-Jacobi (18), because 8 runs there
      // as triangular solves, ~100x slower (precond/hypre_device.hpp).
      HYPRE_BoomerAMGSetRelaxType(h, SymmetricAmgSmoother());
      HYPRE_BoomerAMGSetNumSweeps(h, 1);
      // Strength-of-connection starting points per the spec (2D/3D).
      HYPRE_BoomerAMGSetStrongThreshold(
         h, strong_threshold >= 0.0 ? strong_threshold
         : (mesh.Dimension() == 3 ? 0.25 : 0.1));
      if (agg_levels > 0)
      {
         // Aggressive coarsening on the first levels; second-stage
         // interpolation 5 is one hypre runs on a GPU (hypre_device.hpp).
         HYPRE_BoomerAMGSetAggNumLevels(h, agg_levels);
         HYPRE_BoomerAMGSetAggInterpType(h, 5);
      }
      // On a GPU every option above must be one hypre runs there, fast.
      VerifyHypreAmgOnDevice(amg, "lp_surrogate: the pressure LOR-AMG");
      {
         // Build the hierarchy now rather than in the first application
         // (MFEM sets hypre up lazily), so setup is charged its memory and
         // time. The right-hand side does not enter the setup.
         INCNS_PROFILE("lp::amg_setup");
         mfem::Vector b(A_->Height()), x(A_->Height());
         b.UseDevice(true);
         x.UseDevice(true);
         b = 0.0;
         x = 0.0;
         amg.Setup(b, x);
      }

      singular_ = singular;
      if (singular_) { ortho_.SetSolver(amg); }
   }

   /**
    * @brief Apply the fixed V-cycle(s); Ortho-wrapped when singular.
    * @param x Pressure-space residual (true dofs).
    * @param y Preconditioned output (true dofs).
    */
   void Mult(const mfem::Vector& x, mfem::Vector& y) const override
   {
      if (singular_) { ortho_.Mult(x, y); }
      else { amg_->Mult(x, y); }
   }

   /// Required by mfem::Solver; fixed at construction.
   void SetOperator(const mfem::Operator&) override {}

   /// @return True if the application is Ortho-wrapped (singular operator).
   bool Singular() const { return singular_; }

   /// @return The folded LOR sub-elements found at construction (all ranks;
   ///         > 0: L_p was assembled with |det J|).
   long long FoldedLorElements() const { return folded_; }

   /// @return The assembled LOR L_p.
   const mfem::HypreParMatrix& Matrix() const { return *A_; }

   /// @return The Dirichlet (essential) true dofs of L_p -- the outflow
   ///         pressure dofs under LpBC::Auto; empty when all-Neumann.
   const mfem::Array<int>& EssentialTrueDofs() const { return ess_tdofs_; }

private:
   mfem::Array<int> ess_tdofs_;  ///< Dirichlet true dofs of L_p.
   mfem::ParBilinearForm form_;  ///< HO Diffusion form (LOR source; owned).
   /// The LOR L_p (owned). Declared before amg_, which references it.
   std::unique_ptr<mfem::HypreParMatrix> A_;
   /// BoomerAMG on A_ (owned; must outlive applications).
   std::unique_ptr<mfem::HypreBoomerAMG> amg_;
   mutable DeviceOrthoSolver ortho_; ///< Constant-mode wrap (singular only).
   bool singular_ = false;           ///< Ortho-wrapped?
   long long folded_ = 0;            ///< Folded LOR sub-elements (all ranks).
};

} // namespace incns
