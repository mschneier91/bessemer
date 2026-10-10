/**
 * @file oifs.hpp
 * @brief Operator-integration-factor splitting (OIFS) of the convective term
 *        (Maday, Patera & Ronquist, J. Sci. Comput. 5 (1990) 263-292; as in
 *        Nek5000): the BDF history is advected to the new time level by
 *        sub-stepped RK4, so the Stokes step is not bound by the convective
 *        CFL limit.
 */
#pragma once

#include "bc/boundary_conditions.hpp"
#include "quadrature/rule_book.hpp"
#include "spaces/mixed_spaces.hpp"
#include "time/cfl.hpp"
#include "mfem.hpp"

#include <deque>
#include <memory>
#include <vector>

namespace incns
{

/**
 * @brief Advects the BDF history for OIFS.
 *
 * BDF-k with OIFS replaces
 * @f$ \sum_j c_j u^{n+1-j} + \text{EXT}[(u\cdot\nabla)u] @f$ by
 * @f$ \sum_{j\ge1} c_j \tilde u_j(t^{n+1}) @f$, where @f$ \tilde u_j @f$
 * solves the pure advection problem
 * @f$ \partial_s \tilde u + (w(s)\cdot\nabla)\tilde u = 0 @f$ on
 * @f$ [t^{n+1-j}, t^{n+1}] @f$ from @f$ \tilde u_j(t^{n+1-j}) = u^{n+1-j} @f$,
 * and the advecting velocity @f$ w(s) @f$ is the Lagrange interpolant /
 * extrapolant in time of the velocity history (the EXT order). Advection is
 * linear in @f$ \tilde u @f$, so the weighted sum is advanced as ONE field
 * (Nek's way): start from @f$ c_k u^{n+1-k} @f$ at @f$ t^{n+1-k} @f$, advect
 * to the next history time, add the next level, and so on to
 * @f$ t^{n+1} @f$ -- the sub-integration covers k*dt once.
 *
 * Space: the semi-discrete advection @f$ M\,d\tilde u/ds = -C(w)\tilde u @f$
 * with MFEM's ConvectionIntegrator applied per velocity component (partial
 * assembly, the 3k dealiasing rule of the explicit convection), one operator
 * per history velocity combined with the interpolation weights at each RK
 * stage (C is linear in w).
 *
 * Mass: @f$ M @f$ IS the BDF step's velocity mass, passed in, never rebuilt
 * here. OIFS's dt -> 0 limit is @f$ M M_{sub}^{-1} N(u) @f$, the right
 * equation only when the substeps invert exactly the mass of the BDF step
 * (a mismatch cost C_D +3% / C_L,rms +12% on the Re 200 square cylinder at
 * every dt). A diagonal @f$ M @f$ (the collocated GLL mass on a conforming
 * mesh, OIFS's default) is inverted pointwise; any other (the consistent
 * mass, or the collocated mass on a nonconforming mesh, where
 * @f$ P^T D P @f$ is not diagonal) by a Jacobi-preconditioned CG solve per
 * RK stage (the full inverse, imposed rows included: see Rhs()). The
 * constructor verifies the inverse against @f$ M @f$ and
 * aborts on a mismatch (human 2026-10-10: "error out if the two aren't
 * consistent").
 *
 * Time: classical RK4, substeps chosen so that the convective CFL number of
 * each substep (time/cfl.hpp, the largest rate over the advecting history)
 * does not exceed @c sub_cfl.
 *
 * Boundaries (characteristic rule): a Dirichlet true dof is imposed only
 * where the wind enters the domain or runs along the boundary (w.n <= 0,
 * walls included); there it follows the boundary data at each stage time
 * (each advected level carries u_D(s), so the combined field carries
 * (accumulated weight) * u_D(s)). Where the wind LEAVES (w.n > 0) the
 * advected history is not the physical velocity, so it is left free --
 * imposing u_D there over-constrains the advection and costs an O(h^2)
 * boundary layer (measured: nse_mms_test OifsTemporalOrder2D). The sign uses
 * the newest wind, per Advect(). Do-nothing outflow is natural (pure
 * advection needs no outflow condition). The directional do-nothing term is
 * NOT in the substeps: the integrator applies it explicitly in the BDF step
 * (in the substeps it is stiff at backflow nodes and biased the steady state
 * at CFL 2; measured 2026-10-09).
 *
 * Accuracy: the BDF order, but the time error is the BDF's ALONG
 * TRAJECTORIES (@f$ D^3u/Dt^3 @f$ for BDF2), where IMEX's is that of the
 * Eulerian field (plus the extrapolation's); hence BDF3 for OIFS. Where fluid crosses a nearly fixed pattern quickly
 * (attached shear layers, corners, a steady vortex) OIFS's is far larger at
 * the same step: wall-bounded MMS at dt = 0.005, OIFS 8.6e-3 vs IMEX's
 * all-spatial 5.45e-3 (nse_mms_test OifsWallBounded2D).
 */
class OifsAdvector
{
public:
   /**
    * @param spaces  Mixed spaces (velocity byNODES, vdim = dim); borrowed.
    * @param rules   Quadrature source; must outlive this object.
    * @param bc      Boundary conditions (Dirichlet data, essential dofs);
    *                borrowed. Advect() moves its time and restores it.
    * @param sub_cfl CFL number of each RK4 substep (> 0).
    * @param mass    The BDF step's velocity mass @f$ M @f$ on true dofs
    *                (StokesOperator::Mass()); borrowed, must outlive this.
    * @param mass_diag Its assembled diagonal (StokesOperator::MassDiagonal()).
    * @pre Aborts if the substeps' inverse does not reproduce @p mass.
    */
   OifsAdvector(MixedSpaces& spaces, const RuleBook& rules,
                BoundaryConditions& bc, double sub_cfl, mfem::Operator& mass,
                const mfem::Vector& mass_diag);

   /**
    * @brief The advected BDF history combination.
    * @param c     BDF weights (c[0] leading, unused); k = c.size() - 1 levels.
    * @param hist  Velocity true-dof history, newest first (>= k entries).
    * @param times Times of @p hist.
    * @param t_new Time the history is advected to.
    * @param ext   Number of history velocities the advecting velocity is
    *              interpolated / extrapolated from (capped by @p hist).
    * @param phi   Output: @f$ \sum_{j\ge1} c_j \tilde u_j(t_{new}) @f$ (true dofs).
    */
   void Advect(const std::vector<double>& c, const std::deque<mfem::Vector>& hist,
               const std::deque<double>& times, double t_new, int ext,
               mfem::Vector& phi);

   /**
    * @brief The general form: advect the weighted @p fields by the velocity
    *        interpolated / extrapolated in time from @p wind (Advect() is
    *        this with wind = fields = the velocity history).
    * @param wind        Advecting velocity history, newest first (>= 1).
    * @param wind_times  Its times.
    * @param c           Weights (c[0] unused); k = c.size() - 1 fields.
    * @param fields      Fields to advect, newest first (>= k).
    * @param field_times Their times (increasing toward the front).
    * @param t_new       Target time.
    * @param phi         Output: @f$ \sum_{j\ge1} c_j \tilde f_j(t_{new}) @f$.
    */
   void AdvectWith(const std::deque<mfem::Vector>& wind,
                   const std::vector<double>& wind_times,
                   const std::vector<double>& c,
                   const std::deque<mfem::Vector>& fields,
                   const std::deque<double>& field_times, double t_new,
                   mfem::Vector& phi);

   /// @return RK4 substeps taken by the last Advect().
   int LastSubsteps() const { return last_substeps_; }

   /// @return The Dirichlet true dofs imposed in the last Advect() (where
   ///         the wind enters or is tangential).
   const mfem::Array<int>& ImposedDofs() const { return imposed_; }

   /**
    * @brief Nodal values from a weak-form vector by the BDF step's mass:
    *        @f$ M^{-1} g @f$ (e.g. the convective acceleration (u.grad)u at
    *        the nodes from N(u)).
    * @param g     Weak-form velocity vector (true dofs).
    * @param nodal Output (may alias @p g).
    */
   void MassInverse(const mfem::Vector& g, mfem::Vector& nodal);

   /// @return Whether the mass is diagonal (inverted pointwise, no solves).
   bool MassIsDiagonal() const { return mass_diagonal_; }

   /// @return CG iterations of the substeps' mass solves so far (0 when the
   ///         mass is diagonal).
   long MassSolveIterations() const { return mass_iterations_; }

private:
   /**
    * @brief The advection right-hand side, zero on the imposed dofs.
    * @param s    Time (selects the wind's interpolation weights).
    * @param phi  Field (true dofs).
    * @param dphi Output: @f$ -M^{-1} C(w(s))\,\phi @f$ (the full inverse),
    *             zero on the imposed dofs.
    */
   void Rhs(double s, const mfem::Vector& phi, mfem::Vector& dphi);
   /**
    * @brief Diagonal or not, the inverses, and the consistency check
    *        (aborts when @f$ M^{-1}(Mz) \ne z @f$).
    * @param mass_diag The assembled diagonal of mass_.
    */
   void SetupMass(const mfem::Vector& mass_diag);
   /**
    * @brief phi on the imposed Dirichlet dofs := weight * u_D(s).
    * @param s      Time of the boundary data.
    * @param weight Accumulated weight of the combined field.
    * @param phi    Field (true dofs), modified on the imposed dofs.
    */
   void SetDirichlet(double s, double weight, mfem::Vector& phi);
   /**
    * @brief imposed_ := the Dirichlet true dofs where the wind enters or is
    *        tangential (outward normal from the adjacent element;
    *        collective).
    * @param w The wind (the newest velocity).
    */
   void ClassifyBoundary(const mfem::ParGridFunction& w);

   MixedSpaces& spaces_;        ///< Mixed spaces (borrowed).
   const RuleBook& rules_;      ///< Quadrature source (borrowed).
   BoundaryConditions& bc_;     ///< Boundary conditions (borrowed).
   double sub_cfl_;             ///< Substep CFL number.
   int dim_ = 0;                ///< Spatial dimension.
   int n_s_ = 0;                ///< Scalar true dofs per component.
   int last_substeps_ = 0;      ///< Substeps of the last Advect().
   /// Scalar companion of the velocity space (same mesh and collection).
   std::unique_ptr<mfem::ParFiniteElementSpace> sfes_;
   mfem::Operator& mass_;       ///< The BDF step's velocity mass (borrowed).
   bool mass_diagonal_ = false; ///< mass_ is exactly diagonal.
   mfem::Vector mass_diag_;     ///< Diagonal of mass_ (true dofs).
   mfem::Vector inv_mass_diag_; ///< 1 / mass_diag_ (the diagonal inverse).
   long mass_iterations_ = 0;   ///< CG iterations of the mass solves.
   /// CG on mass_ (non-diagonal mass only).
   std::unique_ptr<mfem::CGSolver> mass_cg_;
   std::unique_ptr<mfem::OperatorJacobiSmoother>
   mass_jacobi_; ///< Its preconditioner.
   mfem::Vector mass_rhs_;      ///< Mass-solve right-hand side scratch.
   mfem::Array<int> no_dofs_;   ///< Empty list (the unconstrained Jacobi).
   std::unique_ptr<ConvectiveCfl> cfl_; ///< CFL rate of the advecting velocity.
   std::vector<std::unique_ptr<mfem::ParGridFunction>>
   vel_; ///< History velocities.
   /// Coefficients reading vel_ (one per history velocity).
   std::vector<std::unique_ptr<mfem::VectorGridFunctionCoefficient>> vel_coef_;
   /// Convection forms C_l (one per history velocity).
   std::vector<std::unique_ptr<mfem::ParBilinearForm>> conv_;
   std::vector<mfem::OperatorPtr> conv_op_; ///< Their true-dof operators.
   std::vector<double> times_ext_;  ///< Times of the advecting velocities.
   int n_ext_ = 0;                  ///< Advecting velocities in use.
   std::unique_ptr<mfem::ParGridFunction> bd_; ///< Dirichlet projection scratch.
   std::unique_ptr<mfem::ParGridFunction> flag_; ///< Inflow flags (scratch).
   mfem::Array<int> imposed_;       ///< Dirichlet tdofs imposed in the substeps.
   mfem::Vector bd_true_;           ///< Its true dofs.
   mfem::Vector bd_vals_;           ///< Values on the Dirichlet dofs.
   mfem::Vector tmp_s_;             ///< Scalar scratch.
   mfem::Vector k1_; ///< RK4 stage 1.
   mfem::Vector k2_; ///< RK4 stage 2.
   mfem::Vector k3_; ///< RK4 stage 3.
   mfem::Vector k4_; ///< RK4 stage 4.
   mfem::Vector y_;  ///< RK4 stage argument.
};

} // namespace incns
