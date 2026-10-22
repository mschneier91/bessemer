/**
 * @file mesh_convection.hpp
 * @brief The mesh-velocity part of the ALE convection, C(w) u = ((w . grad) u, v),
 *        and the mesh velocity w as a velocity field (docs/ale_fluid.md).
 */
#pragma once

#include "ale/mesh_motion.hpp"
#include "quadrature/rule_book.hpp"
#include "spaces/mixed_spaces.hpp"

#include "mfem.hpp"

#include <memory>

namespace incns
{

/**
 * @brief The mesh velocity w on the fluid's velocity space, and a coefficient
 *        that reads it (the Dirichlet data u = w of a moving wall).
 *
 * Update() takes the BDF derivative of the node positions from a MeshMotion
 * (MeshMotion::MeshVelocity) and interpolates it from the mesh's nodal space
 * onto the velocity space: exact when the velocity space contains the nodal
 * space (the same or a higher order). The coefficient stays the same object
 * for the life of this one, so a BoundaryConditions set can hold it.
 */
class MeshVelocityField
{
public:
   /// @param vfes The fluid's velocity space (borrowed; must outlive this).
   explicit MeshVelocityField(mfem::ParFiniteElementSpace& vfes);

   /**
    * @brief w := the BDF-@p order derivative of the node positions at the
    *        motion's newest state.
    * @param motion The mesh motion (its mesh is @p vfes's mesh).
    * @param order  BDF order, 1 to 3 (MeshMotion::States() > order).
    */
   void Update(const MeshMotion& motion, int order);

   /// w := 0 (no motion yet).
   void Zero();

   /// @return w on the velocity space (local dofs valid).
   const mfem::ParGridFunction& Field() const { return w_; }

   /// @return w as a vector coefficient (reads Field()).
   mfem::VectorCoefficient& Coefficient() { return coeff_; }

   /// @return The BDF weight c_0 of the newest node positions in the last
   ///         Update() (0 after Zero()): dw / dx^{n+1}.
   double Lead() const { return lead_; }

private:
   double lead_ = 0.0;                       ///< See Lead().
   mfem::ParGridFunction w_;                 ///< w on the velocity space.
   mfem::VectorGridFunctionCoefficient coeff_; ///< Reads w_.
};

/**
 * @brief y := C(w) u = ((w . grad) u, v) for a given wind w, component by
 *        component (MFEM's ConvectionIntegrator on the scalar companion
 *        space, partial assembly), at the dealiased 3k rule of the fluid's
 *        convection (Convection::DealiasedOrder).
 *
 * In the non-conservative ALE form the convective velocity is u - w, so the
 * mesh velocity adds -C(w) u to the left-hand side; bessemer treats it
 * explicitly, extrapolated like the convection (time_integrator.cpp). The
 * form holds the wind's values and the mesh's geometry at its quadrature
 * points: SetWind() rebuilds it, and must be called after every mesh move.
 * The velocity space must be ordered byNODES (component blocks).
 */
class MeshConvection
{
public:
   /**
    * @param spaces The fluid's spaces (borrowed; must outlive this).
    * @param rules  Quadrature source (borrowed; must outlive this).
    */
   MeshConvection(MixedSpaces& spaces, const RuleBook& rules);

   /**
    * @brief Rebuild the form for wind @p w on the current mesh.
    * @param w The wind on the velocity space (borrowed: it is read at
    *          assembly only).
    */
   void SetWind(const mfem::ParGridFunction& w);

   /**
    * @brief y := C(w) u.
    * @param u Velocity true dofs.
    * @param y Velocity true dofs (overwritten).
    * @pre SetWind() since the last mesh move.
    */
   void Mult(const mfem::Vector& u, mfem::Vector& y) const;

private:
   MixedSpaces& spaces_;                ///< Fluid spaces (borrowed).
   const mfem::IntegrationRule* rule_;  ///< The dealiased rule (RuleBook's).
   /// The scalar space of one velocity component.
   std::unique_ptr<mfem::ParFiniteElementSpace> sfes_;
   int dim_ = 0; ///< Velocity components.
   int n_s_ = 0; ///< True dofs of one component.
   /// The wind as a coefficient (its own copy of the grid function's values).
   std::unique_ptr<mfem::ParGridFunction> wind_;
   /// Coefficient that reads wind_.
   std::unique_ptr<mfem::VectorGridFunctionCoefficient> wind_coeff_;
   /// The scalar convection form (partial assembly).
   std::unique_ptr<mfem::ParBilinearForm> form_;
   /// Its operator on the scalar true dofs.
   mfem::OperatorPtr op_;
};

} // namespace incns
