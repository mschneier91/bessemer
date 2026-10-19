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
#include "time/oifs_convection.hpp"
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
 * Boundary data per stage: constant data (no-slip walls) is projected once
 * per space; data separable in time (the deck's velocity conditions,
 * u_D = s(t) g) has its shapes projected once per space and scaled by s(t)
 * on the device at each stage; any other data is projected at each stage on
 * the host, in a host-only vector, and only its values on the imposed dofs
 * travel to the device. Nothing of the boundary data lives on the device
 * beyond the imposed and Dirichlet dofs' values.
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
    * @param scratch   Element-vector buffers for the convection kernel
    *                  (borrowed; must outlive this); null: its own.
    * @param mass_diag_inv 1 / @p mass_diag when the BDF step's operator keeps
    *                  it (borrowed, like @p mass), or null: computed here.
    * @param on_the_fly The convection's on-the-fly path where eligible
    *                  (OifsConvection; deck time.oifs_on_the_fly).
    * @pre Aborts if the substeps' inverse does not reproduce @p mass.
    */
   OifsAdvector(MixedSpaces& spaces, const RuleBook& rules,
                BoundaryConditions& bc, double sub_cfl, mfem::Operator& mass,
                const mfem::Vector& mass_diag,
                ElementScratch* scratch = nullptr,
                const mfem::Vector* mass_diag_inv = nullptr,
                bool on_the_fly = false);

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

   /**
    * @brief The CFL rate of @p u, the solution at time @p t, through the
    *        advector's per-level cache (ConvectiveCfl::MaxRateAt): the step
    *        size is chosen from it, and the substeps then find the newest
    *        level's rate there (and the older levels' from earlier steps).
    * @param u Velocity on the velocity space.
    * @param t Its time.
    * @return The rate (the same value as ConvectiveCfl::Rate); collective.
    */
   double RateAt(const mfem::ParGridFunction& u, double t) const
   {
      return cfl_->MaxRateAt({&u}, {t});
   }

private:
   /// AdvectWith() on borrowed winds (Advect() passes the history itself:
   /// no copies of it per step). @p history_winds: the winds are the
   /// solution at @p wind_times, so their rates may come from the cache.
   void AdvectImpl(const std::vector<const mfem::Vector*>& wind,
                   const std::vector<double>& wind_times,
                   const std::vector<double>& c,
                   const std::deque<mfem::Vector>& fields,
                   const std::deque<double>& field_times, double t_new,
                   mfem::Vector& phi, bool history_winds);

public:
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
   void SetupMass(const mfem::Vector& mass_diag,
                  const mfem::Vector* mass_diag_inv);
   /**
    * @brief phi on the imposed Dirichlet dofs := weight * u_D(s).
    * @param s      Time of the boundary data.
    * @param weight Accumulated weight of the combined field.
    * @param phi    Field (true dofs), modified on the imposed dofs.
    */
   void SetDirichlet(double s, double weight, mfem::Vector& phi);
   /**
    * @brief The Dirichlet data u_D(s) on the imposed dofs, for data neither
    *        constant nor separable: projected on the host into the host-only
    *        bd_; only the imposed values travel to the device. Collective.
    * @param s    Time of the data.
    * @param vals Output: the values, in the order of imposed_.
    */
   void DirichletValues(double s, mfem::Vector& vals);
   /**
    * @brief Whether the Dirichlet data on the imposed dofs is the same at
    *        three times spanning [s0, s1] (exactly; collective).
    * @param s0   Window start.
    * @param s1   Window end.
    * @param vals Output: the data at s0.
    * @return True when it is: the substeps can then reuse @p vals instead of
    *         projecting at every stage.
    */
   bool StaticDirichlet(double s0, double s1, mfem::Vector& vals);
   /**
    * @brief The true-dof values of a host-only local vector (bd_ or a scratch
    *        like it) on the Dirichlet true dofs, as GetTrueDofs gives them: the
    *        value of each true dof's local dof (ess_ldof_).
    * @param x    Local (L-) vector, valid on the host.
    * @param vals Output, in the order of BoundaryConditions::EssentialTrueDofs().
    */
   void EssentialValues(const mfem::Vector& x, mfem::Vector& vals) const;
   /// Caches ess_ldof_: the local dof whose value each Dirichlet true dof
   /// takes (TrueDofLocalDofs).
   void CacheEssentialLocalDofs();
   /**
    * @brief imposed_ := the Dirichlet true dofs where the wind enters or is
    *        tangential (outward normal from the adjacent element;
    *        collective). Deterministic: a node is free (outflow) only when
    *        its outward normal velocity exceeds 1e-10 of the largest wind on
    *        the boundary, so a no-slip wall (zero wind) is imposed everywhere.
    *        Nodal boundary elements: the test runs on the device and only
    *        the Dirichlet dofs' flags come to the host. Also sets
    *        imposed_pos_.
    * @param w The wind (the newest velocity).
    */
   void ClassifyBoundary(const mfem::ParGridFunction& w);
   /**
    * @brief ClassifyBoundary's flags for nodal boundary elements, on the
    *        device: flag = 1 at the local vdofs of every boundary node where
    *        the wind does not leave (collective: the largest wind).
    * @param w    The wind (an L-vector).
    * @param flag Local vector, zero on entry.
    */
   void FlagBoundaryNodesDevice(const mfem::ParGridFunction& w,
                                mfem::Vector& flag);
   /**
    * @brief ClassifyBoundary's flags for other boundary elements: the wind
    *        evaluated through each boundary element on the host (collective).
    * @param w    The wind (an L-vector).
    * @param flag Local vector, zero on entry.
    */
   void FlagBoundaryNodesHost(const mfem::ParGridFunction& w,
                              mfem::Vector& flag);
   /// Caches the boundary nodes of ClassifyBoundary (bnode_*).
   void CacheBoundaryNodes();

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
   /// 1 / (the diagonal of mass_), true dofs: the diagonal inverse.
   mfem::Vector inv_mass_diag_;
   /// The reciprocal applied: inv_mass_diag_, or the operator's (borrowed).
   const mfem::Vector* inv_ = nullptr;
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
   /// 3D at the solver's sizes: all winds and components in one kernel per
   /// RK stage (OifsConvection), in place of conv_ / conv_op_.
   std::unique_ptr<OifsConvection> combo_;
   std::vector<double> times_ext_;  ///< Times of the advecting velocities.
   int n_ext_ = 0;                  ///< Advecting velocities in use.
   /// Dirichlet projection scratch, host-only (never on the device).
   std::unique_ptr<mfem::ParGridFunction> bd_;
   mfem::Array<int> imposed_;       ///< Dirichlet tdofs imposed in the substeps.
   /// Their positions in BoundaryConditions::EssentialTrueDofs()
   /// (device-usable: the gathers of the per-space data run there).
   mfem::Array<int> imposed_pos_;
   /// Per Dirichlet true dof, the local dof GetTrueDofs reads (host).
   std::vector<int> ess_ldof_;
   long ess_seq_ = -1;        ///< Velocity-space sequence of ess_ldof_.
   /// The nodes of the boundary elements, for ClassifyBoundary: per node its
   /// dim_ local vdofs (component-major), outward normal, boundary element
   /// and node index. Cached per velocity-space sequence (the geometry does
   /// not change between steps).
   std::vector<int> bnode_vdofs_;
   std::vector<double> bnode_normals_; ///< Outward normals (unnormalised).
   std::vector<int> bnode_be_;         ///< Boundary element of each node.
   std::vector<int> bnode_k_;          ///< Node index within it.
   long bnode_seq_ = -1;      ///< Velocity-space sequence of the cache.
   bool bnode_nodal_ = false; ///< Nodal boundary elements: w(node) = dofs.
   std::vector<double> bnode_w_; ///< The wind at the nodes (scratch).
   /// bnode_vdofs_ as a device-usable index list (the wind's gather).
   mfem::Array<int> bnode_gather_;
   mfem::Vector bnode_nor_;   ///< bnode_normals_ on the device (nodal test).
   /// The device maximum's workspace (mfem::reduce; pinned host memory).
   mfem::Array<double> max_workspace_;
   mfem::Array<int> flag_idx_; ///< Local dofs flagged imposed (scratch).
   /// Constant Dirichlet data (BoundaryConditions::DirichletIsConstant) on
   /// the Dirichlet true dofs, projected once per velocity-space sequence
   /// (bd_const_seq_).
   mfem::Vector bd_const_ess_;
   long bd_const_seq_ = -1;   ///< Velocity-space sequence of bd_const_ess_.
   bool bd_const_zero_ = false; ///< ... and zero on every Dirichlet dof.
   /// Separable data (BoundaryConditions::DirichletIsSeparable), whose
   /// shapes BoundaryConditions keeps on the Dirichlet true dofs: this window
   /// scales them per stage.
   bool bd_sep_ = false;      ///< This window scales the shapes per stage.
   mfem::Vector bd_shape_;    ///< The shapes on imposed_ (device).
   mfem::Vector bd_entry_;    ///< Their entries on imposed_ (device).
   mfem::Vector bd_fac_;      ///< The entries' time factors (scratch).
   /// The fallback's local dofs of imposed_ (DirichletValues; host).
   std::vector<int> imposed_ldof_;
   mfem::Vector bd_vals_;           ///< Values on the Dirichlet dofs.
   /// The Dirichlet data on imposed_, when it does not change over the
   /// current Advect() window (bd_static_valid_): reused at every stage.
   mfem::Vector bd_static_;
   bool bd_static_valid_ = false;   ///< bd_static_ holds this window's data.
   bool bd_zero_ = false;           ///< ... and it is zero (no-slip walls).
   mfem::Vector ess_scratch_;       ///< Dirichlet-dof values (scratch).
   mfem::Vector tmp_s_;             ///< Scalar scratch.
   mfem::Vector k1_; ///< RK4 stage 1.
   mfem::Vector k2_; ///< RK4 stage 2.
   mfem::Vector k3_; ///< RK4 stage 3.
   mfem::Vector k4_; ///< RK4 stage 4.
   mfem::Vector y_;  ///< RK4 stage argument.
};

} // namespace incns
