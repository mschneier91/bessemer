/**
 * @file subgrid_stress.hpp
 * @brief The explicit subgrid stress term of large-eddy simulation,
 *        @f$ (2\nu_t(u)\,S(u), S(v)) @f$, as a device operator on true dofs.
 */
#pragma once

#include "operators/subgrid_model.hpp"
#include "quadrature/rule_book.hpp"
#include "spaces/mixed_spaces.hpp"
#include "mfem.hpp"

namespace incns
{

/**
 * @brief The subgrid stress @f$ \tau = 2\nu_t(\nabla u)\,S(u) @f$,
 *        @f$ S = \tfrac12(\nabla u + \nabla u^T) @f$ the symmetric gradient,
 *        in weak form: @c Mult returns @f$ y_i = \int_\Omega \tau(u) :
 *        \nabla\phi_i @f$ on the velocity true dofs, LHS-signed like
 *        Convection::Mult (the term @f$ -\nabla\cdot\tau @f$ stands on the
 *        left of the momentum equation; the caller subtracts it from an
 *        explicit right-hand side). Since @f$ \tau @f$ is symmetric,
 *        @f$ \tau : \nabla\phi = \tau : S(\phi) @f$: the term IS
 *        @f$ (2\nu_t S(u), S(v)) @f$.
 *
 * Why explicit, and why the stress form:
 * - Explicit and EXT-extrapolated together with the convection (IMEX) or
 *   with the directional do-nothing term (OIFS): the implicit block, its
 *   preconditioners and every solver test stay untouched, and nothing is
 *   assembled or stored per step. The model's time scale is the one the
 *   convective CFL already resolves (nu_t ~ |grad u| Delta^2, so
 *   nu_t dt / h^2 ~ CFL Delta^2 / h^2 <= CFL at the node spacing).
 * - The STRESS form, with the symmetric gradient (human 2026-10-16: "don't
 *   forget the symmetric gradient"): a variable viscosity in the Laplacian
 *   form, @f$ -\nabla\cdot(\nu_t\nabla u) @f$, is not the divergence of a
 *   symmetric stress and is wrong wherever @f$ \nabla\nu_t \ne 0 @f$ (the
 *   difference is @f$ \nabla u^T \nabla\nu_t @f$); the bulk viscous term keeps
 *   its Laplacian form because its @f$ \nu @f$ is constant.
 *
 * Quadrature: the RuleBook's dealiasing rule (order 3k, Convection's): nu_t
 * is a non-polynomial function of grad u, and the point count is the one
 * every other nonlinear term here uses.
 *
 * Implementation (docs/les.md): two device kernels per apply. The first, one
 * thread per quadrature point, interpolates grad u from the element dofs,
 * evaluates the model and writes the point's weighted stress pulled back to
 * reference coordinates; the second, one thread per element dof, contracts
 * it with the reference gradients of that dof's basis function. Direct
 * (non sum-factorized) interpolation in both: O(k^6) per element in 3D, a
 * few percent of a step at k = 6 (the sum-factorized version is the
 * follow-up if it ever shows in a profile). No per-step state: the scratch
 * (element vectors in and out, the stress at the points: about 150 B per
 * velocity dof at Q3 and Q6) is allocated and freed in each apply, so the
 * model holds 8 B per element between applies (the filter widths).
 *
 * Quads/hexes only (tensor-product bases), 2D and 3D, hanging nodes through
 * the space's prolongation like every other operator here.
 */
class SubgridStress
{
public:
   /// Volume statistics of the model on one velocity field (Statistics()).
   struct Values
   {
      double nu_t_max = 0.0;     ///< max over the quadrature points of nu_t.
      double nu_t_mean = 0.0;    ///< Volume average of nu_t.
      /// @f$ \int 2\nu_t S(u):S(u) @f$, the subgrid dissipation.
      double dissipation = 0.0;
      double volume = 0.0;       ///< @f$ \int 1 @f$ with the same rule.
   };

   /**
    * @brief Build the operator on the velocity space.
    * @param spaces Mixed spaces (borrowed; must outlive this).
    * @param rules  Quadrature source (borrowed; must outlive this).
    * @param opts   The model and its constants; Enabled() must hold.
    */
   SubgridStress(MixedSpaces& spaces, const RuleBook& rules,
                 const SubgridOptions& opts);

   /**
    * @brief @f$ y_i = \int 2\nu_t(\nabla u) S(u) : \nabla\phi_i @f$ on true
    *        dofs (LHS-signed; see the class note).
    * @param u True-dof velocity (input).
    * @param y True-dof result (output, resized as needed).
    */
   void Mult(const mfem::Vector& u, mfem::Vector& y) const;

   /**
    * @brief The model's volume statistics on a velocity field (collective:
    *        reduced over the communicator).
    * @param u True-dof velocity.
    * @return The maximum and mean eddy viscosity, the subgrid dissipation,
    *         the volume.
    */
   Values Statistics(const mfem::Vector& u) const;

   /// @return The options the operator was built with.
   const SubgridOptions& Options() const { return opts_; }

   /// @return The quadrature rule (the RuleBook's 3k rule).
   const mfem::IntegrationRule& Rule() const { return *rule_; }

   /**
    * @brief The filter width of a local element (host read; for tests).
    * @param e Local element index.
    * @return Delta_K = filter_width * h_K / k_u.
    */
   double FilterWidth(int e) const;

private:
   /**
    * @brief Interpolate u to the element vector and run the two kernels,
    *        leaving the result in the local (L-vector) layout.
    * @param u  True-dof velocity.
    * @param yl Local-dof result (sized to the space's VSize).
    */
   void LocalMult(const mfem::Vector& u, mfem::Vector& yl) const;

   MixedSpaces& spaces_;    ///< Mixed spaces (borrowed).
   const RuleBook& rules_;  ///< Quadrature source (borrowed).
   SubgridOptions opts_;    ///< The model.
   /// Non-owning; owned by the RuleBook, which outlives this object.
   const mfem::IntegrationRule* rule_ = nullptr;
   /// Lexicographic element restriction (owned by the velocity space).
   const mfem::Operator* restrict_ = nullptr;
   /// @f$ (C\Delta_K)^2 @f$ per local element (device).
   mfem::Vector cdelta2_;
   /// @f$ \Delta_K @f$ per local element (host; FilterWidth).
   mfem::Vector delta_;
};

} // namespace incns
