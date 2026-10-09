/**
 * @file convection.hpp
 * @brief Dealiased nonlinear convection operator N(u) = (u . grad)u (Sprint 2.1).
 */
#pragma once

#include "quadrature/rule_book.hpp"
#include "spaces/mixed_spaces.hpp"
#include "mfem.hpp"

namespace incns
{

/**
 * @brief How the nonlinear term of the momentum equation is treated
 *        (deck `physics.convective_form: convective|rotational`).
 *
 * Applies to Equation::NavierStokes only. The skew-symmetric form (2.2b) is
 * still planned, gated on upstream MFEM.
 */
enum class ConvectiveForm
{
   /// (u.grad)u, dealiased, AB/EXT-extrapolated on the right-hand side (IMEX);
   /// the implicit block is the symmetric Stokes one. The default.
   Convective,
   /// Semi-implicit rotational form (docs/design/rotational_convection_pa_spec.md):
   /// (curl w*) x u with the vorticity lagged on the EXT-extrapolated w*, in
   /// the IMPLICIT velocity block (nonsymmetric, skew, energy neutral for any
   /// w*); the gradient part 1/2|u|^2 is absorbed into the pressure, which the
   /// integrator converts back to static pressure for output.
   Rotational
};

/**
 * @brief How the convective term of the convective form is advanced in time
 *        (deck `time.convection: imex|oifs`).
 */
enum class ConvectionTreatment
{
   /// EXT-extrapolated N(u) on the right-hand side: bound by the convective
   /// CFL limit of the BDF step. The default.
   Imex,
   /// Operator-integration-factor splitting (time/oifs.hpp): the BDF history
   /// is advected to the new time by RK4 substeps, so the BDF step can run at
   /// a CFL number of several.
   Oifs
};

/**
 * @brief The condition on do-nothing (natural) outflow boundaries with the IMEX
 *        convective form (deck `physics.outflow: directional|classical`).
 *
 * The rotational form keeps the classical condition (on its Bernoulli head).
 */
enum class OutflowCondition
{
   /// T(u,p) n = 0: nothing assembled. Backflow through the boundary feeds
   /// the convective energy flux 1/2 (u.n)|u|^2 into the domain unchecked.
   Classical,
   /// Braack & Mucha's directional do-nothing condition (J. Comput. Math. 32
   /// (2014) 507-521): T(u,p) n - 1/2 (u.n)_- u = 0. Identical to Classical
   /// where u.n >= 0; cancels the backflow energy flux where u.n < 0. The
   /// default. See DirectionalDoNothingIntegrator.
   Directional
};

/**
 * @brief The nonlinear convective term @c N(u) = (u . grad)u, evaluated with a
 *        DEALIASED (over-integrated) quadrature rule.
 *
 * Sprint 2.1 is deliberately OPERATOR LEVEL ONLY: this class evaluates N(u) on
 * true dofs and nothing more. The AB/EXT extrapolation, the BDF right-hand side
 * and @c NavierStokesSolver are 2.2 -- do not add them here.
 *
 * ### Why over-integration (the whole point of this class)
 *
 * The integrand @c (u.grad)u . v is a product of THREE degree-@c k factors (two
 * from the quadratic nonlinearity, one from the test function), so on an affine
 * element it is a polynomial of degree ~@c 3k-1. The default rule used by the
 * linear blocks is exact only to ~@c 2k, so it commits an aliasing error: energy
 * from unresolved high modes folds back onto resolved ones, which in DNS drives
 * the classic aliasing blow-up. Integrating exactly to degree @c 3k removes it.
 *
 * The rule comes from the RuleBook at order @c 3*k_u (see @ref DealiasedOrder),
 * i.e. ~@c ceil(3k/2) points per direction, per CLAUDE.md's dealiasing rule.
 * Linear terms keep their standard rule -- only this operator is elevated.
 *
 * @warning Do NOT "simplify" this back to the default (or a GLL-collocated)
 * rule. MFEM's own `miniapps/fluids/navier` solver collocates the nonlinear form
 * on a GLL rule at @c 2k-1, which is the aliasing-prone choice this project
 * explicitly rejects; the flagship exactness test asserts that the collocation
 * rule FAILS where the over-integrated rule succeeds.
 *
 * ### Ownership
 * Quadrature comes from the RuleBook, which must outlive this object (MFEM
 * integrators hold non-owning rule pointers) -- same contract as StokesOperator.
 */
class Convection
{
public:
   /**
    * @brief Build the dealiased convection form on the velocity space.
    * @param spaces Mixed velocity/pressure spaces (borrowed, must outlive this).
    * @param rules  Quadrature source (borrowed, must outlive this).
    */
   Convection(MixedSpaces& spaces, const RuleBook& rules);

   /**
    * @brief Apply @c y = N(u) = (u . grad)u on true dofs.
    *
    * Sign convention: this returns the convective term as it appears on the
    * LEFT-hand side of the momentum equation (@c du/dt + N(u) - nu*lap(u) +
    * grad p = f). The caller subtracts it when building an explicit right-hand
    * side. MFEM's navier miniapp instead folds a @c -1 into the integrator's
    * coefficient; keeping the sign OUT of the operator makes this testable
    * against the analytic @c (u.grad)u without a sign convention to remember.
    *
    * @param u True-dof velocity (input).
    * @param y True-dof result (output, resized as needed).
    */
   void Mult(const mfem::Vector& u, mfem::Vector& y) const;

   /**
    * @brief Polynomial degree the dealiasing rule is exact for, given order @p k.
    *
    * @c 3k: the (u.grad)u . v integrand is degree @c 3k-1 on an affine element,
    * and one extra degree costs nothing while covering the mildly-curved case.
    * @param k Velocity polynomial order.
    * @return The exactness order, 3k.
    */
   static int DealiasedOrder(int k) { return 3 * k; }

   /// @return The quadrature rule this operator integrates with (for tests).
   const mfem::IntegrationRule& Rule() const { return *rule_; }

   /**
    * @brief Add the directional do-nothing term on the given outflow
    *        boundaries: Mult then returns
    *        @f$ N(u) - \tfrac12\int_{S_1}(u\cdot n)_-\,u\cdot\phi @f$ --
    *        the boundary part of the convective flux, so the IMEX scheme
    *        extrapolates it exactly like N (explicitly, with the EXT weights).
    *        Face rule: the RuleBook's dealiasing order 3k on the face
    *        geometry. Call once, before the first Mult.
    * @param outflow_attrs Boundary attributes (1-based) of the outflow S_1.
    */
   void EnableDirectionalDoNothing(const mfem::Array<int>& outflow_attrs);

   /**
    * @brief The directional do-nothing term alone, LHS-signed:
    *        @f$ y = -\tfrac12\int_{S_1}(u\cdot n)_-\,u\cdot\phi @f$ (OIFS
    *        keeps N in its substeps and extrapolates only this term).
    * @pre EnableDirectionalDoNothing() was called.
    * @param u True-dof velocity (input).
    * @param y True-dof result (output, resized as needed).
    */
   void MultDirectionalDoNothing(const mfem::Vector& u, mfem::Vector& y) const;

   /// @return True when Mult includes the directional do-nothing term.
   bool DirectionalDoNothing() const { return ddn_form_ != nullptr; }

private:
   MixedSpaces& spaces_;   ///< Mixed spaces (borrowed).
   const RuleBook& rules_; ///< Quadrature source (borrowed).
   /// Non-owning; owned by the RuleBook, which outlives this object.
   const mfem::IntegrationRule* rule_ = nullptr;
   /// The nonlinear form carrying VectorConvectionNLFIntegrator.
   std::unique_ptr<mfem::ParNonlinearForm> form_;
   /// Outflow marker of the directional do-nothing term (per attribute).
   mfem::Array<int> ddn_marker_;
   /// The velocity the boundary term is evaluated at (set in Mult).
   std::unique_ptr<mfem::ParGridFunction> ddn_u_;
   /// Boundary-face linear form of the term (null = classical do-nothing).
   std::unique_ptr<mfem::ParLinearForm> ddn_form_;
   /// The term on true dofs (Mult scratch).
   mutable mfem::Vector ddn_true_;
};

} // namespace incns
