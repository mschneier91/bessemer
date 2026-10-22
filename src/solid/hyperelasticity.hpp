/**
 * @file hyperelasticity.hpp
 * @brief Large-deformation elasticity (St. Venant-Kirchhoff), total
 *        Lagrangian, static: Newton with load steps on HyperelasticOperator,
 *        CG with the reference configuration's LOR-AMG
 *        (docs/design/hyperelasticity_spec.md sections 1 and 4, milestone 2).
 */
#pragma once

#include "quadrature/rule_book.hpp"
#include "solid/elasticity_parameters.hpp"
#include "solid/hyperelastic_operator.hpp"
#include "mfem.hpp"

#include <memory>
#include <vector>

namespace incns
{

/**
 * @brief Finds u in [H1 Q_k]^dim with u = g on the Dirichlet boundary and
 *        int P(F(u)) : grad v = (f, v) + (t, v)_N for every v vanishing there:
 *        dead loads, per reference volume and area.
 *
 * Newton on HyperelasticOperator's residual and tangent. Each Newton step
 * solves the tangent with CG, preconditioned by one BoomerAMG V-cycle on the
 * LOR linear-elasticity operator of the reference configuration (the tangent
 * at u = 0), built once (spec D5). Load steps ramp the loads and the Dirichlet
 * data together, from zero; a step's first Newton step takes its Dirichlet
 * increment (prescribed on the correction, the free dofs following through the
 * tangent). A step converges when its residual (true dofs, Dirichlet entries
 * out) falls below max(newton_atol, newton_rtol r_ref), r_ref the largest of its
 * first residual, its load's norm and its increment's right-hand side (the
 * tangent applied to it, free rows). The API mirrors
 * LinearElasticity's (static part). Every quadrature rule is owned here, so the
 * solver must not outlive its mesh.
 */
class Hyperelasticity
{
public:
   /// Construction-time settings.
   struct Options
   {
      int order = 2;       ///< Displacement polynomial order k.
      double lambda = 1.0; ///< First Lame parameter (plane strain in 2D).
      double mu = 1.0;     ///< Shear modulus.
      HyperelasticMaterial material = HyperelasticMaterial::StVenantKirchhoff;
      /// Newton: residual reduction per load step. Not much below 1e-8: the
      /// residual's roundoff floor was 3e-11 of the load on a small cantilever
      /// (H6), and grows with slenderness and refinement.
      double newton_rtol = 1e-8;
      double newton_atol = 0.0;   ///< Newton: absolute residual tolerance.
      int newton_max_iter = 25;   ///< Newton iterations per load step.
      int load_steps = 1;         ///< Load steps of a Solve().
      bool line_search = false;   ///< Backtracking on the residual norm.
      double rtol = 1e-8;  ///< CG relative tolerance of each Newton step.
      double atol = 0.0;   ///< CG absolute tolerance.
      int max_iter = 2000; ///< CG iteration cap.
      int print_level = -1; ///< MFEM CG print level (-1 silent).
      /// What the tangent keeps from its linearization point.
      TangentStorage storage = TangentStorage::Deformation;
      /// Per-element stiffness factors (one per local element; borrowed, must
      /// outlive the solver), multiplying lambda and mu (the mesh motion's
      /// Jacobian stiffening); null: none.
      const mfem::Vector* element_stiffness = nullptr;
   };

   /**
    * @brief Build the displacement space and the operators. Collective.
    * @param mesh Quad/hex mesh (borrowed; must outlive this object).
    * @param opts Order, material and solver settings.
    */
   Hyperelasticity(mfem::ParMesh& mesh, const Options& opts);

   /// Clears the mesh's geometric factors, which are keyed by rules owned here.
   ~Hyperelasticity();

   Hyperelasticity(const Hyperelasticity&) = delete;
   Hyperelasticity& operator=(const Hyperelasticity&) = delete;

   /// @param attr Boundary attribute held fixed (u = 0).
   void AddFixed(int attr);
   /**
    * @param attr Boundary attribute with a prescribed displacement.
    * @param g    The displacement (borrowed; must outlive the solves).
    */
   void AddDisplacement(int attr, mfem::VectorCoefficient& g);
   /**
    * @param attr Boundary attribute with a prescribed traction P N (per
    *             reference area: a dead load).
    * @param t    The traction (borrowed; must outlive the solves).
    */
   void AddTraction(int attr, mfem::VectorCoefficient& t);
   /// @param f Body force per reference volume (borrowed).
   void SetBodyForce(mfem::VectorCoefficient& f);

   /// @brief Solve() at time 0. Collective. @return Whether Newton converged.
   bool Solve() { return Solve(0.0); }

   /**
    * @brief The static solve with the data at time @p t (their coefficients'
    *        SetTime), in load_steps steps, from the current displacement.
    *        Collective.
    * @param t Time of the data.
    * @return Whether every load step converged.
    */
   bool Solve(double t);

   /// @return The displacement (after Solve()).
   mfem::ParGridFunction& Displacement() { return *u_; }
   /// @return The displacement (after Solve()).
   const mfem::ParGridFunction& Displacement() const { return *u_; }
   /// @return The displacement space (vector H1, ordered by nodes).
   mfem::ParFiniteElementSpace& Space() { return *fes_; }
   /// @return The residual and tangent operator (no boundary conditions).
   HyperelasticOperator& NonlinearOperator() { return *op_; }
   /// @return The quadrature rule of the operator and the load.
   const mfem::IntegrationRule& DomainRule() const;

   /// @return Newton iterations of the last Solve(), summed over its load steps.
   int NewtonIterations() const { return newton_iterations_; }
   /// @return The most Newton iterations of one load step in the last Solve().
   int MaxNewtonIterations() const { return max_newton_; }
   /// @return CG iterations of the last Solve().
   int LinearIterations() const { return linear_iterations_; }
   /// @return CG iterations of every Solve() since construction.
   long long TotalLinearIterations() const { return total_linear_; }
   /// @return The last Solve()'s residual norms, every Newton iterate of every
   ///         load step in order.
   const std::vector<double>& ResidualHistory() const { return history_; }
   /// @return Whether the last Solve() converged.
   bool Converged() const { return converged_; }
   /// @return Global number of true displacement dofs.
   HYPRE_BigInt GlobalTrueDofs() const { return fes_->GlobalTrueVSize(); }
   /// @return The options in effect.
   const Options& GetOptions() const { return opts_; }

   /// @return The strain energy int W(F(u_h)) dX. Collective.
   double StrainEnergy() const;
   /// @return The work of the last Solve()'s full load on u_h. Collective.
   double ExternalWork() const;
   /// @return max over the nodes of |u_h|. Collective.
   double MaxDisplacement() const;

   /**
    * @brief Take @p x (true dofs) as the displacement, the next Solve()'s
    *        starting point.
    * @param x The displacement's true dofs.
    */
   void SetSolution(const mfem::Vector& x);

private:
   /// @param attr Boundary attribute. @return A marker selecting only it.
   mfem::Array<int> Marker(int attr) const;
   /// Fix the Dirichlet set, the load form, the preconditioner, the CG.
   void Finalize();
   /// The LOR-AMG of the reference linear elasticity, once.
   void SetupPreconditioner();
   /// The full load at time @p t into load_ (true dofs).
   void AssembleLoad(double t);
   /// The full Dirichlet data at time @p t into ess_full_.
   void DirichletValues(double t);
   /**
    * @brief r = R(x) - s load_, its Dirichlet entries zeroed.
    * @return ||r|| (global l2).
    */
   double Residual(const mfem::Vector& x, double s, mfem::Vector& r);

   mfem::ParMesh& mesh_; ///< The mesh (borrowed).
   Options opts_;        ///< Settings.
   RuleBook rules_;      ///< Owns every quadrature rule used here.
   std::unique_ptr<mfem::H1_FECollection> fec_;       ///< GLL nodal H1.
   std::unique_ptr<mfem::ParFiniteElementSpace> fes_; ///< Vector H1, byNODES.
   mfem::Vector lambda_e_, mu_e_;  ///< Lame parameters per local element.
   std::unique_ptr<HyperelasticOperator> op_; ///< R(u) and K_T(u).
   std::unique_ptr<mfem::ParGridFunction> u_;     ///< Displacement.
   std::unique_ptr<mfem::ParGridFunction> bc_gf_; ///< Dirichlet data work field.
   std::unique_ptr<mfem::ParLinearForm> b_;       ///< The load form.

   std::vector<int> fixed_; ///< Fixed attributes.
   /// Prescribed-displacement attributes with their data (borrowed).
   std::vector<std::pair<int, mfem::VectorCoefficient*>> displaced_;
   /// Traction attributes with their data (borrowed).
   std::vector<std::pair<int, mfem::VectorCoefficient*>> tractions_;
   /// Markers of the traction integrators (MFEM keeps pointers to them).
   std::vector<std::unique_ptr<mfem::Array<int>>> traction_markers_;
   mfem::VectorCoefficient* body_ = nullptr; ///< Body force (borrowed).

   mfem::Array<int> ess_tdofs_; ///< Essential true dofs.
   bool finalized_ = false;     ///< Whether the Dirichlet set is fixed.
   std::unique_ptr<mfem::Coefficient> lor_lambda_, lor_mu_; ///< On the LOR mesh.
   std::unique_ptr<mfem::HypreParMatrix> lor_matrix_; ///< The LOR operator.
   std::unique_ptr<mfem::Solver> pc_;                 ///< CG preconditioner.
   std::unique_ptr<mfem::CGSolver> cg_;               ///< Tangent solves.
   std::unique_ptr<mfem::ConstrainedOperator> kc_; ///< The tangent, constrained.
   const mfem::Operator* k_ = nullptr;             ///< The tangent kc_ wraps.

   mfem::Vector load_;     ///< Full load of the last Solve() (true dofs).
   mfem::Vector ess_full_; ///< Full Dirichlet data at the essential dofs.
   mfem::Vector ess_step_; ///< A load step's Dirichlet data.
   mfem::Vector ess_jump_; ///< Its increment over the current displacement.
   mfem::Vector x_, r_, dx_, rhs_, trial_; ///< Newton work vectors.

   std::vector<double> history_; ///< Residual norms of the last Solve().
   int newton_iterations_ = 0, max_newton_ = 0, linear_iterations_ = 0;
   long long total_linear_ = 0;
   bool converged_ = false;
};

} // namespace incns
