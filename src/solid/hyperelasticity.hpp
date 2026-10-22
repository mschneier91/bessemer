/**
 * @file hyperelasticity.hpp
 * @brief Large-deformation elasticity (St. Venant-Kirchhoff), total
 *        Lagrangian, static or dynamic: Newton with load steps on
 *        HyperelasticOperator, generalized-alpha or BDF2 in time, CG with the
 *        reference configuration's LOR-AMG (docs/design/hyperelasticity_spec.md
 *        sections 1 and 4, milestones 2 and 3).
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
 * tangent applied to it, free rows), or when a correction falls below
 * newton_dx_rtol times the displacement.
 *
 * Dynamics: rho u_tt + R(u) = f(t), the consistent mass at rho. Each step is
 * one Newton solve of c M u^{n+1} + R(theta u^{n+1} + (1 - theta) u^n) = b:
 * generalized-alpha (Chung-Hulbert, the internal force and the load at
 * t_{n+1-alpha_f}: theta = 1 - alpha_f), or BDF2 on (u, v) after one
 * trapezoidal step (theta = 1), from u^n. The preconditioner is then the
 * LOR-AMG of K_0 + (c / theta) rho M, rebuilt when dt changes. The API mirrors
 * LinearElasticity's. Every quadrature rule is owned here, so the solver must
 * not outlive its mesh.
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
      /// Newton: residual reduction per load step. The residual has a
      /// roundoff floor that grows with slenderness and refinement (3e-11 of
      /// the load on H6's small cantilever, 1e-7 on CSM1's flag at 4e4 dofs),
      /// so newton_dx_rtol usually decides on fine meshes.
      double newton_rtol = 1e-8;
      double newton_atol = 0.0;   ///< Newton: absolute residual tolerance.
      /// Newton: a load step also converges when a correction's norm falls
      /// below this times the displacement's (0: off). Corrections keep
      /// falling quadratically to ~1e-15, below the residual's floor.
      double newton_dx_rtol = 1e-10;
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
      double density = 1.0; ///< Mass density per reference volume (dynamics).
      /// Time integrator (dynamics): generalized-alpha, or BDF2 (Bdf; BDF3 is
      /// not offered: it is not A-stable, docs/elasticity.md section 1).
      SolidTimeScheme scheme = SolidTimeScheme::GeneralizedAlpha;
      double rho_inf = 0.8; ///< Generalized-alpha high-frequency radius, [0, 1].
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

   /**
    * @brief Start a dynamic march: the time, the initial displacement and
    *        velocity (projected; zero when null; the Dirichlet data at @p t0
    *        overrides them on the boundary), the load at @p t0 and, for
    *        generalized-alpha, the initial acceleration from M a0 = f(t0) -
    *        R(u0) (zero on the Dirichlet boundary). Collective. Step() calls
    *        it with t0 = 0 and a zero state if it was not called.
    * @param t0 Initial time.
    * @param u0 Initial displacement, or null.
    * @param v0 Initial velocity, or null.
    */
   void Initialize(double t0, mfem::VectorCoefficient* u0 = nullptr,
                   mfem::VectorCoefficient* v0 = nullptr);

   /**
    * @brief Advance by @p dt: one Newton solve. Collective. The first BDF step
    *        is a trapezoidal step; the BDF2 weights come from the actual step
    *        sizes.
    * @param dt Step size (> 0).
    * @return Whether Newton converged.
    */
   bool Step(double dt);

   /// @return The time of the current dynamic state.
   double Time() const { return time_; }
   /// @return Steps taken since Initialize().
   int StepCount() const { return steps_; }
   /// @return The velocity (valid after Initialize()).
   mfem::ParGridFunction& Velocity() { return *v_gf_; }
   /// @return 1/2 v.M v of the current state. Collective.
   double KineticEnergy() const;
   /// @return Newton iterations of every Solve() and Step() since construction.
   long long TotalNewtonIterations() const { return total_newton_; }

   /// @return The displacement (after Solve(), Initialize() or Step()).
   mfem::ParGridFunction& Displacement() { return *u_; }
   /// @return The displacement (after Solve(), Initialize() or Step()).
   const mfem::ParGridFunction& Displacement() const { return *u_; }
   /// @return The displacement space (vector H1, ordered by nodes).
   mfem::ParFiniteElementSpace& Space() { return *fes_; }
   /// @return The residual and tangent operator (no boundary conditions).
   HyperelasticOperator& NonlinearOperator() { return *op_; }
   /// @return The quadrature rule of the operator and the load.
   const mfem::IntegrationRule& DomainRule() const;

   /// @return Newton iterations of the last Solve() (summed over its load
   ///         steps) or Step().
   int NewtonIterations() const { return newton_iterations_; }
   /// @return The most Newton iterations of one load step in the last Solve()
   ///         (of the last Step()).
   int MaxNewtonIterations() const { return max_newton_; }
   /// @return CG iterations of the last Solve() or Step().
   int LinearIterations() const { return linear_iterations_; }
   /// @return CG iterations of every Solve() and Step() since construction.
   long long TotalLinearIterations() const { return total_linear_; }
   /// @return The last Solve()'s (Step()'s) residual norms, every Newton
   ///         iterate of every load step in order.
   const std::vector<double>& ResidualHistory() const { return history_; }
   /// @return The last Solve()'s (Step()'s) Newton corrections, each ||dx|| /
   ///         ||u|| after it, every load step in order.
   const std::vector<double>& CorrectionHistory() const { return corrections_; }
   /// @return Whether the last Solve() or Step() converged.
   bool Converged() const { return converged_; }
   /// @return Global number of true displacement dofs.
   HYPRE_BigInt GlobalTrueDofs() const { return fes_->GlobalTrueVSize(); }
   /// @return The options in effect.
   const Options& GetOptions() const { return opts_; }

   /// @return The strain energy int W(F(u_h)) dX. Collective.
   double StrainEnergy() const;
   /// @return The work of the last full load (Solve()) or the current one
   ///         (dynamics) on u_h. Collective.
   double ExternalWork() const;
   /// @return max over the nodes of |u_h|. Collective.
   double MaxDisplacement() const;

   /**
    * @brief The displacement at a physical point of the reference
    *        configuration. Collective: every rank passes the same point. The
    *        point search is cached for the last point asked for.
    * @param x Point (dim coordinates).
    * @param u The displacement there (dim components).
    * @return False when no element contains the point (u is then zero).
    */
   bool DisplacementAt(const mfem::Vector& x, mfem::Vector& u) const;

   /**
    * @brief Take @p x (true dofs) as the displacement, the next Solve()'s
    *        starting point.
    * @param x The displacement's true dofs.
    */
   void SetSolution(const mfem::Vector& x);

private:
   class ShiftedTangent; // theta K_T + c M, the Newton operator (the .cpp)

   /// @param attr Boundary attribute. @return A marker selecting only it.
   mfem::Array<int> Marker(int attr) const;
   /// Fix the Dirichlet set, the load form, the preconditioner, the CG.
   void Finalize();
   /// The LOR-AMG of the reference linear elasticity plus @p shift rho M.
   void SetupPreconditioner(double shift);
   /// The consistent mass at rho (partial assembly, true dofs), once.
   void BuildMass();
   /// The displacement and velocity fields from the newest state.
   void SyncFields();
   /// The full load at time @p t into load_ (true dofs).
   void AssembleLoad(double t);
   /// The full Dirichlet data at time @p t into ess_full_.
   void DirichletValues(double t);
   /**
    * @brief Newton on G(x) = c M x + R(theta x + (1 - theta) y) - b, the
    *        Dirichlet dofs of x moved to @p g by the first correction (the
    *        free dofs following through the tangent). Appends to the
    *        histories and counters. Collective.
    * @param c     Mass weight (0: static).
    * @param theta Weight of x in the internal force's argument.
    * @param y     The argument's other part (u^n), or null when theta is 1.
    * @param b     Right-hand side (true dofs; Dirichlet entries ignored).
    * @param g     Dirichlet values at the essential dofs.
    * @param x     Initial guess in, the solution out.
    * @return Whether it converged.
    */
   bool Newton(double c, double theta, const mfem::Vector* y,
               const mfem::Vector& b, const mfem::Vector& g, mfem::Vector& x);
   /**
    * @brief r = G(x) of the running Newton(), its Dirichlet entries zeroed.
    * @return ||r|| (global l2).
    */
   double Residual(const mfem::Vector& x, mfem::Vector& r);
   /// Reset the per-solve histories and counters.
   void ResetCounters();

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
   std::unique_ptr<ShiftedTangent> jac_;            ///< theta K_T + c M.
   std::unique_ptr<mfem::ConstrainedOperator> kc_; ///< jac_, constrained.
   const mfem::Operator* k_ = nullptr;             ///< The tangent jac_ wraps.
   double pc_shift_ = 0.0; ///< The mass shift the preconditioner was built for.
   mfem::ConstantCoefficient lor_mass_; ///< shift rho, on the LOR mesh.

   mfem::ConstantCoefficient rho_;          ///< Density.
   std::unique_ptr<mfem::ParBilinearForm> m_form_; ///< The mass (PA).
   mfem::OperatorHandle m_;                 ///< The mass on true dofs.
   mfem::Vector m_diag_;                    ///< Its diagonal (true dofs).
   /// The running Newton()'s G: mass weight, theta, u^n and right-hand side.
   double g_c_ = 0.0, g_theta_ = 1.0;
   const mfem::Vector* g_y_ = nullptr;
   const mfem::Vector* g_b_ = nullptr;

   std::vector<mfem::Vector> u_hist_; ///< u^n, u^{n-1}, ...; then a spare level.
   std::vector<mfem::Vector> v_hist_; ///< The velocities alike.
   std::vector<double> t_hist_;       ///< Their times.
   mfem::Vector a_n_;  ///< Generalized-alpha acceleration a^n.
   mfem::Vector f_n_;  ///< Load at t_n.
   mfem::Vector step_rhs_; ///< A step's or load step's right-hand side.
   mfem::Vector w_, z_, mx_; ///< Work vectors (predictor, argument, mass apply).
   std::unique_ptr<mfem::ParGridFunction> v_gf_; ///< Velocity.
   double time_ = 0.0;        ///< Time of the dynamic state.
   int steps_ = 0;            ///< Steps since Initialize().
   bool initialized_ = false; ///< Whether Initialize() ran.

   mfem::Vector load_;     ///< Full load of the last Solve() or Step().
   mfem::Vector ess_full_; ///< Full Dirichlet data at the essential dofs.
   mfem::Vector ess_step_; ///< A load step's Dirichlet data.
   mfem::Vector ess_jump_; ///< Its increment over the current displacement.
   mfem::Vector x_, r_, dx_, rhs_, trial_; ///< Newton work vectors.

   /// Point of the cached DisplacementAt() search.
   mutable mfem::Vector probe_x_;
   mutable int probe_elem_ = -1;             ///< Its element on this rank, or -1.
   mutable mfem::IntegrationPoint probe_ip_; ///< Its reference point.
   mutable bool probe_cached_ = false;       ///< Whether probe_x_ is searched.

   std::vector<double> history_; ///< Residual norms of the last Solve().
   std::vector<double> corrections_; ///< Its relative correction norms.
   int newton_iterations_ = 0, max_newton_ = 0, linear_iterations_ = 0;
   long long total_linear_ = 0, total_newton_ = 0;
   bool converged_ = false;
};

} // namespace incns
