/**
 * @file linear_elasticity.hpp
 * @brief Small-strain linear elasticity, displacement-only, on the reference
 *        (Lagrangian) configuration, static or dynamic: partially assembled
 *        operators, CG with a LOR-BoomerAMG or Jacobi preconditioner,
 *        generalized-alpha or BDF2/BDF3 in time (docs/elasticity.md).
 */
#pragma once

#include "quadrature/rule_book.hpp"
#include "solid/elasticity_parameters.hpp"
#include "mfem.hpp"

#include <memory>
#include <vector>

namespace incns
{

/**
 * @brief Finds u in [H1 Q_k]^dim with
 *        (2 mu eps(u), eps(v)) + (lambda div u, div v) = (f, v) + (t, v)_N
 *        for every v vanishing on the Dirichlet boundary.
 *
 * The operator is MFEM's ElasticityIntegrator in partial assembly (it runs on
 * the device) at the RuleBook's Gauss-Legendre rule of order 2k + 1 (k + 1
 * points per direction). Boundaries are fixed (u = 0), prescribed displacement,
 * traction (natural) or, by omission, traction-free. Every quadrature rule is
 * owned here, so the solver must not outlive its mesh (the destructor clears
 * the mesh's rule-keyed geometric factors, developing.md section 6).
 */
class LinearElasticity
{
public:
   /// Construction-time settings.
   struct Options
   {
      int order = 2;       ///< Displacement polynomial order k.
      double lambda = 1.0; ///< First Lame parameter (the plane model applied).
      double mu = 1.0;     ///< Shear modulus.
      double rtol = 1e-10; ///< CG relative tolerance.
      double atol = 0.0;   ///< CG absolute tolerance.
      int max_iter = 2000; ///< CG iteration cap.
      ElasticityPC preconditioner = ElasticityPC::LorAmg; ///< CG preconditioner.
      int print_level = -1; ///< MFEM CG print level (-1 silent).
      double density = 1.0; ///< Mass density (dynamics).
      /// Time integrator (dynamics).
      SolidTimeScheme scheme = SolidTimeScheme::GeneralizedAlpha;
      int bdf_order = 2;    ///< BDF order, 2 or 3 (scheme Bdf).
      double rho_inf = 0.8; ///< Generalized-alpha high-frequency radius.
      /// Per-element stiffness factors (one per local element; borrowed,
      /// must outlive the solver), multiplying lambda and mu on each element
      /// (the Jacobian stiffening of mesh motion); null: none.
      const mfem::Vector* element_stiffness = nullptr;
   };

   /**
    * @brief Build the displacement space and the operator. Collective.
    * @param mesh Quad/hex mesh (borrowed; must outlive this object).
    * @param opts Order, material and solver settings.
    */
   LinearElasticity(mfem::ParMesh& mesh, const Options& opts);

   /// Clears the mesh's geometric factors, which are keyed by rules owned here.
   ~LinearElasticity();

   LinearElasticity(const LinearElasticity&) = delete;
   LinearElasticity& operator=(const LinearElasticity&) = delete;

   /// @param attr Boundary attribute held fixed (u = 0).
   void AddFixed(int attr);

   /**
    * @param attr Boundary attribute with a prescribed displacement.
    * @param g    The displacement (borrowed; must outlive the solves).
    */
   void AddDisplacement(int attr, mfem::VectorCoefficient& g);

   /**
    * @param attr Boundary attribute with a prescribed traction sigma n.
    * @param t    The traction (borrowed; must outlive the solves).
    */
   void AddTraction(int attr, mfem::VectorCoefficient& t);

   /// @param f Body force per unit volume (borrowed; must outlive the solves).
   void SetBodyForce(mfem::VectorCoefficient& f);

   /**
    * @brief Assemble the load, build the preconditioner on first use (the
    *        Dirichlet set is final then) and solve. Collective.
    * @return Whether CG reached its tolerance.
    */
   bool Solve();

   /**
    * @brief A static solve with the Dirichlet data and the load at time
    *        @p t (their coefficients' SetTime), from a zero initial guess.
    *        Collective. Does not change Time(), which is the dynamic state's.
    * @param t Time of the data.
    * @return Whether CG reached its tolerance.
    */
   bool Solve(double t);

   /**
    * @brief The static solve of Solve(double), from the initial guess
    *        @p guess (true dofs; its Dirichlet values are replaced), to the
    *        tolerance a zero guess would have had: rtol relative to the
    *        preconditioned norm of the zero guess's residual (one extra
    *        preconditioner apply). A close guess then saves iterations
    *        instead of buying digits. Collective.
    * @param t     Time of the data.
    * @param guess Initial guess (true dofs).
    * @return Whether CG reached its tolerance.
    */
   bool Solve(double t, const mfem::Vector& guess);

   /**
    * @brief Start a dynamic march: the time, the initial displacement and
    *        velocity (projected; zero when null; the Dirichlet data at @p t0
    *        overrides them on the boundary), the load at @p t0 and, for
    *        generalized-alpha, the initial acceleration from M a0 = f(t0) - K
    *        u0 (zero on the Dirichlet boundary). Collective. Step() calls it
    *        with t0 = 0 and a zero state if it was not called.
    * @param t0 Initial time.
    * @param u0 Initial displacement, or null.
    * @param v0 Initial velocity, or null.
    */
   void Initialize(double t0, mfem::VectorCoefficient* u0 = nullptr,
                   mfem::VectorCoefficient* v0 = nullptr);

   /**
    * @brief Advance by @p dt. Collective. The first BDF step is a trapezoidal
    *        step, the second BDF2, the third BDF3 (bdf_order 3); the BDF
    *        weights come from the actual step sizes.
    * @param dt Step size (> 0).
    * @return Whether the step's CG reached its tolerance.
    */
   bool Step(double dt);

   /// @return The time of the current state.
   double Time() const { return time_; }
   /// @return Steps taken since Initialize().
   int StepCount() const { return steps_; }
   /// @return The velocity (valid after Initialize()).
   mfem::ParGridFunction& Velocity() { return *v_gf_; }
   /// @return 1/2 v.M v of the current state (device). Collective.
   double KineticEnergy() const;
   /// @return Total CG iterations of every solve since construction.
   long long TotalIterations() const { return total_iterations_; }
   /// @return The largest CG iteration count of one solve.
   int MaxIterations() const { return max_iterations_; }

   /// @return The displacement (after Solve(), Initialize() or Step()).
   mfem::ParGridFunction& Displacement() { return *u_; }
   /// @return The displacement (after Solve(), Initialize() or Step()).
   const mfem::ParGridFunction& Displacement() const { return *u_; }
   /// @return The displacement space (vector H1, ordered by nodes).
   mfem::ParFiniteElementSpace& Space() { return *fes_; }
   /// @return The partially assembled bilinear form (no boundary conditions).
   mfem::ParBilinearForm& Form() { return *a_; }
   /// @return The rules every integrator here uses.
   const RuleBook& Rules() const { return rules_; }
   /// @return The quadrature rule of the operator and the load.
   const mfem::IntegrationRule& DomainRule() const;

   /// @return CG iterations of the last solve.
   int Iterations() const { return iterations_; }
   /// @return Final CG residual norm of the last solve.
   double FinalNorm() const { return final_norm_; }
   /// @return Whether the last solve converged.
   bool Converged() const { return converged_; }
   /// @return Global number of true displacement dofs.
   HYPRE_BigInt GlobalTrueDofs() const { return fes_->GlobalTrueVSize(); }
   /// @return The options in effect.
   const Options& GetOptions() const { return opts_; }

   /**
    * @return The strain energy 1/2 u.K u = 1/2 int sigma(u_h) : eps(u_h), from
    *         the partially assembled operator (device). Collective.
    */
   double StrainEnergy() const;

   /**
    * @return The work of the last assembled load on the solution, (f, u_h) +
    *         (t, u_h)_N. With zero Dirichlet data a static solve makes it
    *         twice the strain energy (to the solver tolerance). Collective.
    */
   double ExternalWork() const;

   /**
    * @brief The Dirichlet data at time @p t on the essential true dofs, in
    *        their order (the fixed groups' zeros included).
    * @param t Time of the data.
    * @param g Receives the values.
    */
   void DirichletValues(double t, mfem::Vector& g);

   /**
    * @brief Take @p x (true dofs) as the solution, as if a solve had made it
    *        (no iterations counted).
    * @param x The displacement's true dofs.
    */
   void SetSolution(const mfem::Vector& x);

   /// @return max over the nodes of |u_h|. Collective.
   double MaxDisplacement() const;

   /**
    * @brief The displacement at a physical point. Collective: every rank
    *        passes the same point. The point search is cached for the last
    *        point asked for.
    * @param x Point (dim coordinates).
    * @param u The displacement there (dim components).
    * @return False when no element contains the point (u is then zero).
    */
   bool DisplacementAt(const mfem::Vector& x, mfem::Vector& u) const;

private:
   /// (c_K K + c_M M) on true dofs from the two PA forms (defined in the .cpp).
   class TrueOperator;

   /// @param attr Boundary attribute. @return A marker selecting only it.
   mfem::Array<int> Marker(int attr) const;
   /// Fix the Dirichlet set: the essential true dofs, the operators.
   void Finalize();
   /// Build the mass form (dynamics), once.
   void BuildMass();
   /**
    * @brief Build the CG preconditioner of K + c M (rebuilt when c changes).
    * @param c Mass shift.
    */
   void SetupPreconditioner(double c);
   /**
    * @brief Assemble the load at time @p t into @p f (true dofs).
    * @param t Time.
    * @param f Load vector.
    */
   void AssembleLoad(double t, mfem::Vector& f);
   /**
    * @brief Overwrite the essential entries of @p x with the Dirichlet data
    *        at time @p t.
    * @param t Time.
    * @param x True-dof vector.
    */
   void ApplyDirichlet(double t, mfem::Vector& x);
   /**
    * @brief Solve (K + c M) x = b with x's essential entries as the data
    *        and its other entries as the initial guess. Collective.
    * @param c Mass shift (0: static).
    * @param b Right-hand side (true dofs; overwritten by the elimination).
    * @param x Solution (true dofs).
    * @return Whether CG converged.
    */
   bool SolveShifted(double c, mfem::Vector& b, mfem::Vector& x);
   /// Copy the current true-dof state into the grid functions.
   void SyncFields();

   mfem::ParMesh& mesh_; ///< The mesh (borrowed).
   Options opts_;        ///< Settings.
   RuleBook rules_;      ///< Owns every quadrature rule used here.
   std::unique_ptr<mfem::H1_FECollection> fec_;          ///< GLL nodal H1.
   std::unique_ptr<mfem::ParFiniteElementSpace> fes_;    ///< Vector H1, byNODES.
   /**
    * @param value A Lame parameter.
    * @param children Sub-elements per element of the mesh it is evaluated on
    *        (1, or k^dim on the LOR mesh).
    * @return The constant, or value times Options::element_stiffness.
    */
   std::unique_ptr<mfem::Coefficient> Lame(double value, int children) const;

   std::unique_ptr<mfem::Coefficient> lambda_; ///< First Lame parameter.
   std::unique_ptr<mfem::Coefficient> mu_;     ///< Shear modulus.
   std::unique_ptr<mfem::Coefficient> lor_lambda_; ///< lambda on the LOR mesh.
   std::unique_ptr<mfem::Coefficient> lor_mu_;     ///< mu on the LOR mesh.
   std::unique_ptr<mfem::ParBilinearForm> a_;     ///< PA elasticity form.
   mfem::ConstantCoefficient rho_;                ///< Density.
   std::unique_ptr<mfem::ParBilinearForm> m_;     ///< PA mass form (dynamics).
   std::unique_ptr<TrueOperator> op_;             ///< K + c M on true dofs.
   std::unique_ptr<mfem::ConstrainedOperator> constrained_; ///< op_ constrained.
   std::unique_ptr<mfem::ParGridFunction> u_;     ///< Displacement.
   std::unique_ptr<mfem::ParGridFunction> v_gf_;  ///< Velocity (dynamics).
   std::unique_ptr<mfem::ParGridFunction> bc_gf_; ///< Dirichlet data work field.
   std::unique_ptr<mfem::ParLinearForm> b_;       ///< The load form.
   mfem::Vector load_; ///< Load of the last assembly (true dofs).

   std::vector<int> fixed_;                         ///< Fixed attributes.
   /// Prescribed-displacement attributes with their data (borrowed).
   std::vector<std::pair<int, mfem::VectorCoefficient*>> displaced_;
   /// Traction attributes with their data (borrowed).
   std::vector<std::pair<int, mfem::VectorCoefficient*>> tractions_;
   /// Markers of the traction integrators (MFEM keeps pointers to them).
   std::vector<std::unique_ptr<mfem::Array<int>>> traction_markers_;
   mfem::VectorCoefficient* body_ = nullptr;        ///< Body force (borrowed).

   /// Essential true dofs (outlives the preconditioner, which borrows it).
   mfem::Array<int> ess_tdofs_;
   bool finalized_ = false; ///< Whether the Dirichlet set is fixed.
   bool pc_ready_ = false;  ///< Whether pc_ matches pc_shift_.
   double pc_shift_ = 0.0;  ///< The mass shift c pc_ was built for.
   mfem::ConstantCoefficient lor_mass_; ///< c rho in the LOR form.
   std::unique_ptr<mfem::HypreParMatrix> lor_matrix_; ///< The LOR operator.
   std::unique_ptr<mfem::Solver> pc_;                 ///< CG preconditioner.
   mfem::Vector diag_;   ///< Operator diagonal (Jacobi; borrowed by pc_).
   mfem::Vector m_diag_; ///< Mass diagonal (Jacobi, dynamics).

   // Dynamic state (true dofs). Generalized-alpha keeps u, v, a and the load
   // at t_n; BDF keeps up to three back levels of u and v.
   bool initialized_ = false; ///< Whether Initialize() ran.
   double time_ = 0.0;        ///< Time of the current state.
   int steps_ = 0;            ///< Steps since Initialize().
   std::vector<mfem::Vector> u_hist_; ///< u^n, u^{n-1}, ... (newest first).
   std::vector<mfem::Vector> v_hist_; ///< v^n, v^{n-1}, ...
   std::vector<double> t_hist_;       ///< Their times.
   mfem::Vector a_n_;    ///< Generalized-alpha acceleration a^n.
   mfem::Vector f_n_;    ///< Load at t_n (generalized-alpha, trapezoidal).
   mfem::Vector rhs_;    ///< Right-hand side work vector.
   mfem::Vector work_;   ///< Work vector.
   mfem::Vector work2_;  ///< Work vector.
   mfem::Vector work3_;  ///< Work vector.
   mfem::Vector work4_;  ///< Work vector (Solve with a guess).
   mfem::Vector ess_vals_; ///< Dirichlet values at the essential true dofs.
   mutable mfem::Vector ework_; ///< Work vector of the energies.
   std::unique_ptr<mfem::CGSolver> cg_; ///< The CG of every K + c M solve.

   /// Point of the cached DisplacementAt() search.
   mutable mfem::Vector probe_x_;
   mutable int probe_elem_ = -1;           ///< Its element on this rank, or -1.
   mutable mfem::IntegrationPoint probe_ip_; ///< Its reference point.
   mutable bool probe_cached_ = false;     ///< Whether the cache is valid.

   int iterations_ = 0;      ///< CG iterations of the last solve.
   double final_norm_ = 0.0; ///< CG final residual norm.
   bool converged_ = false;  ///< Last solve converged.
   long long total_iterations_ = 0; ///< CG iterations of every solve.
   int max_iterations_ = 0;  ///< Largest CG count of one solve.
};

} // namespace incns
