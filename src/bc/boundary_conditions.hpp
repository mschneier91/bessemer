/**
 * @file boundary_conditions.hpp
 * @brief Per-attribute velocity boundary conditions (Dirichlet / outflow).
 */
#pragma once

#include "mfem.hpp"

#include <memory>
#include <vector>

namespace incns
{

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

private:
   /// Recompute the essential true-dof list from the Dirichlet markers.
   void UpdateEssentialTrueDofs();

   mfem::ParFiniteElementSpace& vfes_; ///< Velocity space (borrowed).
   int max_attr_;                      ///< Largest boundary attribute in the mesh.
   mfem::Array<int> dirichlet_marker_; ///< Per-attribute Dirichlet flags.
   mfem::Array<int> outflow_marker_;   ///< Per-attribute outflow flags.
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
};

} // namespace incns
