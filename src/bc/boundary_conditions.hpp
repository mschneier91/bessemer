/**
 * @file boundary_conditions.hpp
 * @brief Per-attribute velocity boundary conditions (Dirichlet / outflow).
 */
#pragma once

#include "mfem.hpp"

#include <functional>
#include <memory>
#include <vector>

namespace incns
{

/**
 * @brief Velocity data separable in time, @f$ u_D(x, t) = s(t)\,g(x) @f$: the
 *        deck's velocity conditions (a constant or parabolic profile g,
 *        optionally ramped or modulated by s).
 *
 * Eval() is the data itself, exactly as the given function computes it.
 * TimeFactor() and Shape() expose s and g, so that the OIFS substeps project
 * g once and scale it at every RK stage instead of projecting the data
 * (OifsAdvector). The function must equal s(t) g(x) to roundoff, and exactly
 * where s(t) = 1.
 */
class SeparableVectorCoefficient : public mfem::VectorFunctionCoefficient
{
public:
   /// The data u_D(x, t).
   using Function =
      std::function<void(const mfem::Vector&, double, mfem::Vector&)>;
   /// The shape g(x).
   using ShapeFunction = std::function<void(const mfem::Vector&, mfem::Vector&)>;
   /// The time factor s(t).
   using Factor = std::function<double(double)>;

   /**
    * @param dim   Vector dimension.
    * @param f     The data.
    * @param g     Its shape.
    * @param s     Its time factor.
    */
   SeparableVectorCoefficient(int dim, Function f, ShapeFunction g, Factor s)
      : mfem::VectorFunctionCoefficient(dim, std::move(f)),
        shape_(dim, std::move(g)), factor_(std::move(s)) {}

   /// @return s(t).
   double TimeFactor(double t) const { return factor_(t); }

   /// @return The shape g as a coefficient (for projections).
   mfem::VectorCoefficient& Shape() { return shape_; }

private:
   mfem::VectorFunctionCoefficient shape_; ///< g.
   Factor factor_;                         ///< s.
};

/**
 * @brief The local dof whose value each listed true dof takes in GetTrueDofs
 *        (R has one unit entry per row): on a conforming space the owned local
 *        dof of the true-dof number, found without building R (and P);
 *        otherwise read from R.
 * @param fes   Space.
 * @param tdofs True dofs (this rank's numbering).
 * @param ldofs Output: their local dofs, in order.
 */
void TrueDofLocalDofs(const mfem::ParFiniteElementSpace& fes,
                      const mfem::Array<int>& tdofs, std::vector<int>& ldofs);

/**
 * @brief Velocity boundary conditions, assigned per boundary attribute.
 *
 * Two kinds are supported at the library level:
 *  - @b Dirichlet (no-slip / prescribed): the attribute's velocity dofs are
 *    eliminated from the block system; the data may be time-dependent, so the
 *    coefficients are advanced with SetTime() and re-projected every step.
 *  - @b Outflow (do-nothing traction): natural -- nothing is assembled or
 *    eliminated; registering the attribute only records the physics intent
 *    (it drives pressure null-space detection).
 *
 * Periodicity is mesh-level (see MakeBoxMesh), not a condition here; an empty
 * BC set on a periodic mesh means fully periodic.
 *
 * Dirichlet coefficients are borrowed: the caller keeps them alive for this
 * object's lifetime.
 */
class BoundaryConditions
{
public:
   /**
    * @brief Create an empty BC set over a velocity space.
    * @param vfes Velocity finite element space (vector H1).
    */
   explicit BoundaryConditions(mfem::ParFiniteElementSpace& vfes);

   /**
    * @brief Prescribe velocity Dirichlet data on a boundary attribute.
    * @param attr  Boundary attribute (1-based, must exist in the mesh).
    * @param coeff Velocity data; may be time-dependent (advanced by SetTime).
    */
   void AddVelocityDirichlet(int attr, mfem::VectorCoefficient& coeff);

   /**
    * @brief No-slip wall: prescribe zero velocity on a boundary attribute.
    *
    * A convenience for the ubiquitous wall condition -- equivalent to
    * AddVelocityDirichlet with a zero field, but the zero coefficient is owned
    * internally so the caller supplies nothing.
    *
    * @param attr Boundary attribute (1-based, must exist in the mesh).
    */
   void AddNoSlip(int attr);

   /**
    * @brief Mark a boundary attribute as outflow (do-nothing traction).
    * @param attr Boundary attribute (1-based, must exist in the mesh).
    */
   void AddOutflow(int attr);

   /**
    * @brief Mark a boundary attribute as a Robin boundary: the traction is
    *        sigma(u, p) n = g + alpha (w - u), w the given velocity and g a
    *        load the caller sets (StokesTimeIntegrator::SetBoundaryLoad).
    *        The momentum block gets alpha (u, v) on these faces
    *        (StokesOperator), the right-hand side alpha (w, v) + g. The
    *        interface of a Robin-Neumann FSI coupling (docs/fsi.md).
    * @param attr  Boundary attribute (1-based, must exist in the mesh).
    * @param alpha Robin coefficient (kinematic: per unit density), > 0;
    *              the same on every Robin attribute.
    * @param w     The Robin data velocity (borrowed; must outlive this).
    */
   void AddRobin(int attr, double alpha, mfem::VectorCoefficient& w);

   /**
    * @brief AddRobin() with a coefficient that varies along the boundary.
    * @param attr      Boundary attribute.
    * @param alpha     Robin coefficient, > 0 on the faces (borrowed; the same
    *                  object on every Robin attribute).
    * @param alpha_max An upper bound of alpha, > 0 (the LOR preconditioner's
    *                  constant stand-in: a coefficient of the high-order
    *                  mesh cannot be evaluated on the LOR mesh).
    * @param w         The Robin data velocity (borrowed).
    */
   void AddRobin(int attr, mfem::Coefficient& alpha, double alpha_max,
                 mfem::VectorCoefficient& w);

   /// @return Whether any attribute is a Robin boundary.
   bool HasRobin() const { return robin_alpha_ != nullptr; }

   /// @return The Robin coefficient (null without Robin boundaries).
   mfem::Coefficient* RobinAlpha() const { return robin_alpha_; }

   /// @return An upper bound of RobinAlpha() (0 without Robin boundaries).
   double RobinAlphaMax() const { return robin_alpha_max_; }

   /// @return Per-attribute Robin flags.
   const mfem::Array<int>& RobinMarker() const { return robin_marker_; }

   /// @return The Robin data velocity (null without Robin boundaries).
   mfem::VectorCoefficient* RobinVelocity() const { return robin_w_; }

   /**
    * @brief Whether every Dirichlet coefficient is a constant
    *        (mfem::VectorConstantCoefficient, e.g. no-slip): data that does not
    *        depend on time or position. Deck `velocity` conditions are
    *        functions (they may ramp in time), so they do not count.
    * @return True when they all are (also when there are none).
    */
   bool DirichletIsConstant() const;

   /**
    * @brief Whether every Dirichlet coefficient is separable in time: a
    *        constant (factor 1) or a SeparableVectorCoefficient (the deck's
    *        velocity conditions).
    * @return True when they all are (also when there are none).
    */
   bool DirichletIsSeparable() const;

   /// @return The Dirichlet entries: one per attribute, in the order
   ///         ProjectDirichlet() applies them (the last one wins at boundary
   ///         dofs that attributes share).
   int NumDirichlet() const { return static_cast<int>(dirichlet_.size()); }

   /**
    * @brief The time factor of Dirichlet entry @p e (separable data).
    * @param e Entry, 0 <= e < NumDirichlet().
    * @param t Time.
    * @return s(t) of its SeparableVectorCoefficient; 1 for a constant.
    */
   double DirichletFactor(int e, double t) const;

   /**
    * @brief Separable data (DirichletIsSeparable()): the shapes g projected
    *        as ProjectDirichlet() projects the data -- attribute by attribute,
    *        averaged over the boundary elements (and ranks) that share a dof,
    *        the last entry winning -- and that entry's index at each dof. The
    *        data at time t is then s_e(t) g there: exactly where s_e(t) = 1,
    *        to roundoff otherwise (the projection averages s_e(t) g).
    *        Collective.
    * @param g     Receives the shapes on the Dirichlet boundary.
    * @param entry Receives the entry index (a whole number) there.
    */
   void ProjectDirichletShapes(mfem::ParGridFunction& g,
                               mfem::ParGridFunction& entry) const;

   /**
    * @brief Separable data: the shapes on EssentialTrueDofs() (true-dof
    *        values, as GetTrueDofs gives them), projected on first use and
    *        again after the space changes (collective then). Device-usable.
    */
   const mfem::Vector& EssentialShapes() const;
   /// Their entries (whole numbers), as EssentialShapes().
   const mfem::Vector& EssentialEntries() const;

   /**
    * @brief Separable data (DirichletIsSeparable()) on EssentialTrueDofs() at
    *        each coefficient's current time: s_e(t) g, from the shapes
    *        projected once, on the device. The projected data exactly where
    *        s_e = 1, to roundoff otherwise (ProjectDirichletShapes()).
    *        Collective on first use.
    * @param vals Output, in the order of EssentialTrueDofs().
    */
   void SeparableEssentialValues(mfem::Vector& vals) const;

   /**
    * @brief Advance every Dirichlet coefficient to time @p t.
    * @param t New time; a subsequent ProjectDirichlet() uses data at this time.
    */
   void SetTime(double t);

   /**
    * @brief Essential (Dirichlet-eliminated) velocity true dofs.
    * @return Sorted list of true-dof indices on this rank.
    */
   const mfem::Array<int>& EssentialTrueDofs() const { return ess_tdofs_; }

   /**
    * @brief Project the Dirichlet data (at the current time) onto the boundary
    *        of a velocity field. Interior values are left untouched.
    * @param u Velocity grid function to receive the boundary values.
    */
   void ProjectDirichlet(mfem::ParGridFunction& u) const;

   /// @return True if any attribute is marked outflow.
   bool HasOutflow() const { return has_outflow_; }

   /**
    * @brief The boundary attributes registered as outflow (1-based).
    * @return Attribute list (empty when none) -- e.g. for the CC L_p BC policy.
    */
   mfem::Array<int> OutflowAttributes() const
   {
      mfem::Array<int> attrs;
      for (int a = 1; a <= outflow_marker_.Size(); ++a)
      {
         if (outflow_marker_[a - 1]) { attrs.Append(a); }
      }
      return attrs;
   }

   /**
    * @brief Does the pressure constant null space exist for this BC set?
    *
    * It exists exactly when no boundary carries an outflow/natural condition:
    * fully periodic (no real boundary faces), or every attribute with real
    * boundary faces is velocity Dirichlet. An attribute with no assigned
    * condition is natural (do-nothing), so full Dirichlet coverage of the real
    * boundary is required; periodic pseudo-boundary attributes are ignored.
    *
    * @return True if the constant pressure mode is in the null space (and must
    *         be removed by orthogonalization -- never by pinning).
    */
   bool PressureNullspaceExists() const;

   /**
    * @brief Rebuild the dof-dependent data after the velocity space changed
    *        (adaptive refinement): the essential true-dof list. The attribute
    *        classification is topological and survives refinement unchanged.
    */
   void Update() { UpdateEssentialTrueDofs(); }

private:
   /// Recompute the essential true-dof list from the Dirichlet markers.
   void UpdateEssentialTrueDofs();
   /// The shapes and entries on the essential true dofs (ess_shape_,
   /// ess_entry_), unless current for the space's sequence.
   void CacheShapes() const;

   mfem::ParFiniteElementSpace& vfes_; ///< Velocity space (borrowed).
   int max_attr_;                      ///< Largest boundary attribute in the mesh.
   mfem::Array<int> dirichlet_marker_; ///< Per-attribute Dirichlet flags.
   mfem::Array<int> outflow_marker_;   ///< Per-attribute outflow flags.
   mfem::Array<int> robin_marker_;     ///< Per-attribute Robin flags.
   mfem::Coefficient* robin_alpha_ = nullptr; ///< Robin coefficient.
   double robin_alpha_max_ = 0.0; ///< Its upper bound.
   /// A constant coefficient given as a number (AddRobin(int, double, ...)).
   std::unique_ptr<mfem::ConstantCoefficient> robin_constant_;
   mfem::VectorCoefficient* robin_w_ = nullptr; ///< Robin data velocity.
   /// Per-attribute "owns real boundary faces" flags. MakePeriodic keeps the
   /// base mesh's boundary elements but their faces are topologically interior,
   /// so attributes are classified by face topology (globally reduced); BCs on
   /// a periodic pseudo-boundary attribute are rejected.
   mfem::Array<int> real_bdr_marker_;
   bool has_outflow_ = false;          ///< Any outflow attribute registered.
   /// (attribute, coefficient) pairs; coefficients are borrowed.
   std::vector<std::pair<int, mfem::VectorCoefficient*>> dirichlet_;
   mfem::Array<int> ess_tdofs_;        ///< Essential velocity true dofs.
   /// Zero velocity coefficient owned for no-slip walls (built on first use).
   std::unique_ptr<mfem::VectorConstantCoefficient> zero_coeff_;
   /// Separable data on ess_tdofs_ (CacheShapes): shapes, entries; and the
   /// entries' factors (scratch).
   mutable mfem::Vector ess_shape_, ess_entry_, fac_;
   mutable long shape_seq_ = -1; ///< Velocity-space sequence of the shapes.
};

} // namespace incns
