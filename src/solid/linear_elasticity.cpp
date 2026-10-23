#include "solid/linear_elasticity.hpp"

#include "solid/element_scaled_coefficient.hpp"
#include "solid/hyperelastic_operator.hpp"

#include "precond/hypre_device.hpp"
#include "time/multistep_coeffs.hpp"
#include "util/profiler.hpp"

#include <algorithm>
#include <cmath>

namespace incns
{

using namespace mfem;

/**
 * @brief y = (c_K K + c_M M) x on true dofs: K on true dofs itself, or P^T K
 *        P from MFEM's partially assembled form; P^T M P from the mass form
 *        (no assembled matrix, no boundary conditions).
 */
class LinearElasticity::TrueOperator : public Operator
{
public:
   /**
    * @param fes The displacement space.
    * @param k   The stiffness on true dofs (borrowed), or null.
    * @param kl  The stiffness form on local dofs where @p k is null
    *            (borrowed).
    */
   TrueOperator(const ParFiniteElementSpace& fes, const Operator* k,
                const ParBilinearForm* kl)
      : Operator(fes.GetTrueVSize()), P_(fes.GetProlongationMatrix()), k_(k),
        kl_(kl)
   {
      for (Vector* v : {&xl_, &yl_, &tl_})
      {
         v->SetSize(fes.GetVSize());
         v->UseDevice(true);
      }
      if (k_)
      {
         tt_.SetSize(Height());
         tt_.UseDevice(true);
      }
   }

   /// @param m The mass form (borrowed), or null.
   void SetMass(const ParBilinearForm* m) { m_ = m; }

   /// @param s An elastic support's true-dof matrix, part of the stiffness
   ///          (borrowed), or null.
   void SetSupport(const HypreParMatrix* s) { s_ = s; }

   /**
    * @param ck The stiffness weight Mult() applies.
    * @param cm The mass weight Mult() applies.
    */
   void SetWeights(double ck, double cm)
   {
      ck_ = ck;
      cm_ = cm;
   }

   /// y = (c_K K + c_M M) x with the weights of SetWeights().
   void Mult(const Vector& x, Vector& y) const override { Apply(ck_, cm_, x, y); }

   /**
    * @brief y = (ck K + cm M) x.
    * @param ck Stiffness weight.
    * @param cm Mass weight (needs the mass form unless 0).
    * @param x  True-dof input.
    * @param y  True-dof output.
    */
   void Apply(double ck, double cm, const Vector& x, Vector& y) const
   {
      y.SetSize(Height());
      const bool own = ck != 0.0 && k_;  // K on true dofs
      const bool form = ck != 0.0 && !k_; // MFEM's form, on local dofs
      if (form || cm != 0.0)
      {
         P_->Mult(x, xl_);
         if (form)
         {
            kl_->Mult(xl_, yl_);
            if (ck != 1.0) { yl_ *= ck; }
         }
         else
         {
            yl_ = 0.0;
         }
         if (cm != 0.0)
         {
            MFEM_VERIFY(m_, "linear_elasticity: the mass form is not built");
            m_->Mult(xl_, tl_); // BilinearForm::AddMult needs an assembled matrix
            yl_.Add(cm, tl_);
         }
         P_->MultTranspose(yl_, own ? tt_ : y);
      }
      if (own)
      {
         k_->Mult(x, y);
         if (ck != 1.0) { y *= ck; }
         if (cm != 0.0) { y += tt_; }
      }
      else if (!form && cm == 0.0)
      {
         y = 0.0;
      }
      if (s_ && ck != 0.0) { s_->Mult(ck, x, 1.0, y); }
   }

private:
   const Operator* P_;              ///< Prolongation (true -> local dofs).
   const Operator* k_;              ///< Stiffness on true dofs, or null.
   const ParBilinearForm* kl_;      ///< Stiffness form (k_ null).
   const ParBilinearForm* m_ = nullptr; ///< Mass form, or null.
   const HypreParMatrix* s_ = nullptr;  ///< Elastic support, or null.
   double ck_ = 1.0;                ///< Stiffness weight of Mult().
   double cm_ = 0.0;                ///< Mass weight of Mult().
   mutable Vector xl_;              ///< P x.
   mutable Vector yl_;              ///< The local result.
   mutable Vector tl_;              ///< M P x.
   mutable Vector tt_;              ///< P^T M P x (with k_).
};

LinearElasticity::LinearElasticity(ParMesh& mesh, const Options& opts)
   : mesh_(mesh), opts_(opts), rho_(opts.density)
{
   INCNS_PROFILE("solid::setup");
   const int dim = mesh_.Dimension();
   MFEM_VERIFY(opts_.order >= 1, "linear_elasticity: order must be >= 1");
   MFEM_VERIFY(opts_.mu > 0.0 && opts_.lambda > -2.0 * opts_.mu / dim,
               "linear_elasticity: the material is not positive definite "
               "(mu > 0 and lambda > -2 mu / dim)");
   MFEM_VERIFY(opts_.density > 0.0, "linear_elasticity: density must be > 0");
   MFEM_VERIFY(opts_.bdf_order == 2 || opts_.bdf_order == 3,
               "linear_elasticity: bdf_order must be 2 or 3");
   MFEM_VERIFY(opts_.rho_inf >= 0.0 && opts_.rho_inf <= 1.0,
               "linear_elasticity: rho_inf must be in [0, 1]");
   if (opts_.element_stiffness)
   {
      MFEM_VERIFY(opts_.element_stiffness->Size() == mesh_.GetNE(),
                  "linear_elasticity: element_stiffness needs one factor per "
                  "local element");
      MFEM_VERIFY(mesh_.GetNE() == 0 || opts_.element_stiffness->Min() > 0.0,
                  "linear_elasticity: element_stiffness must be positive");
   }
   lambda_ = Lame(opts_.lambda, 1);
   mu_ = Lame(opts_.mu, 1);
   // GLL nodal basis: the LOR preconditioner refines on the GLL nodes.
   fec_ = std::make_unique<H1_FECollection>(opts_.order, dim);
   // byNODES: MFEM's elasticity partial assembly requires it.
   fes_ = std::make_unique<ParFiniteElementSpace>(&mesh_, fec_.get(), dim,
          Ordering::byNODES);

   if (opts_.order <= 7)
   {
      // lambda and mu per element, as the coefficients evaluate them.
      const int ne = mesh_.GetNE();
      const double* f = opts_.element_stiffness ?
                        opts_.element_stiffness->HostRead() : nullptr;
      Vector lam(ne), mu(ne);
      for (int e = 0; e < ne; ++e)
      {
         lam(e) = opts_.lambda * (f ? f[e] : 1.0);
         mu(e) = opts_.mu * (f ? f[e] : 1.0);
      }
      k_ = std::make_unique<HyperelasticOperator>(
              *fes_, DomainRule(), HyperelasticMaterial::LinearElastic, lam, mu);
   }
   else
   {
      a_ = std::make_unique<ParBilinearForm>(fes_.get());
      a_->SetAssemblyLevel(AssemblyLevel::PARTIAL);
      auto* integ = new ElasticityIntegrator(*lambda_, *mu_);
      integ->SetIntRule(&DomainRule());
      a_->AddDomainIntegrator(integ);
      a_->Assemble();
   }
   op_ = std::make_unique<TrueOperator>(*fes_, k_.get(), a_.get());

   u_ = std::make_unique<ParGridFunction>(fes_.get());
   *u_ = 0.0;
   const int n = fes_->GetTrueVSize();
   for (Vector* v : {&load_, &rhs_, &work_, &work2_, &work3_, &ework_})
   {
      v->SetSize(n);
      v->UseDevice(true);
      *v = 0.0;
   }
}

void LinearElasticity::ApplyStiffness(const Vector& x, Vector& y) const
{
   op_->Apply(1.0, 0.0, x, y);
}

void LinearElasticity::StiffnessDiagonal(Vector& d) const
{
   d.SetSize(fes_->GetTrueVSize());
   d.UseDevice(true);
   if (k_) { k_->AssembleDiagonal(d); }
   else { a_->AssembleDiagonal(d); }
}

LinearElasticity::~LinearElasticity()
{
   // The cached geometric factors are keyed by addresses of rules_' rules.
   mesh_.DeleteGeometricFactors();
}

std::unique_ptr<Coefficient>
LinearElasticity::Lame(double value, int children) const
{
   if (!opts_.element_stiffness)
   {
      return std::make_unique<ConstantCoefficient>(value);
   }
   return std::make_unique<ElementScaledCoefficient>(
             value, *opts_.element_stiffness, children);
}

const IntegrationRule& LinearElasticity::DomainRule() const
{
   const Geometry::Type g = mesh_.Dimension() == 2 ? Geometry::SQUARE :
                            Geometry::CUBE;
   return rules_.Get(g, 2 * opts_.order + 1);
}

Array<int> LinearElasticity::Marker(int attr) const
{
   const int n = mesh_.bdr_attributes.Size() ? mesh_.bdr_attributes.Max() : 0;
   MFEM_VERIFY(attr >= 1 && attr <= n, "linear_elasticity: boundary attribute "
               << attr << " is not on the mesh (1.." << n << ")");
   Array<int> m(n);
   m = 0;
   m[attr - 1] = 1;
   return m;
}

void LinearElasticity::AddFixed(int attr)
{
   MFEM_VERIFY(!finalized_, "linear_elasticity: boundary conditions are fixed "
               "at the first solve");
   Marker(attr); // checks the attribute
   fixed_.push_back(attr);
}

void LinearElasticity::AddDisplacement(int attr, VectorCoefficient& g)
{
   MFEM_VERIFY(!finalized_, "linear_elasticity: boundary conditions are fixed "
               "at the first solve");
   Marker(attr);
   displaced_.push_back({attr, &g});
}

void LinearElasticity::AddTraction(int attr, VectorCoefficient& t)
{
   MFEM_VERIFY(!finalized_, "linear_elasticity: boundary conditions are fixed "
               "at the first solve");
   Marker(attr);
   tractions_.push_back({attr, &t});
}

void LinearElasticity::AddSupport(int attr, double k)
{
   MFEM_VERIFY(!finalized_, "linear_elasticity: boundary conditions are fixed "
               "at the first solve");
   MFEM_VERIFY(k > 0.0, "linear_elasticity: the support stiffness must be "
               "positive");
   MFEM_VERIFY(support_k_.constant == 0.0 || support_k_.constant == k,
               "linear_elasticity: one support stiffness for all attributes");
   const Array<int> m = Marker(attr);
   if (support_marker_.Size() == 0)
   {
      support_marker_.SetSize(m.Size());
      support_marker_ = 0;
   }
   for (int i = 0; i < m.Size(); ++i) { if (m[i]) { support_marker_[i] = 1; } }
   support_k_.constant = k;
}

void LinearElasticity::SetBodyForce(VectorCoefficient& f)
{
   MFEM_VERIFY(!finalized_, "linear_elasticity: the load is fixed at the "
               "first solve");
   body_ = &f;
}

void LinearElasticity::Finalize()
{
   if (finalized_) { return; }
   const int n = mesh_.bdr_attributes.Size() ? mesh_.bdr_attributes.Max() : 0;
   Array<int> ess(n);
   ess = 0;
   for (int a : fixed_) { ess[a - 1] = 1; }
   for (const auto& d : displaced_) { ess[d.first - 1] = 1; }
   fes_->GetEssentialTrueDofs(ess, ess_tdofs_);
   // Read on the device by the constrained operator, the Jacobi smoother and
   // the Dirichlet copies (developing.md section 6).
   ess_tdofs_.GetMemory().UseDevice(true);
   ess_vals_.SetSize(ess_tdofs_.Size());
   ess_vals_.UseDevice(true);
   if (support_marker_.Size() > 0)
   {
      ParBilinearForm s(fes_.get());
      s.AddBoundaryIntegrator(new VectorMassIntegrator(support_k_),
                              support_marker_);
      s.Assemble();
      s.Finalize();
      support_.reset(s.ParallelAssemble());
      op_->SetSupport(support_.get());
   }
   constrained_ = std::make_unique<ConstrainedOperator>(op_.get(), ess_tdofs_);
   bc_gf_ = std::make_unique<ParGridFunction>(fes_.get());

   // The load form: its integrators are re-evaluated at every assembly.
   b_ = std::make_unique<ParLinearForm>(fes_.get());
   if (body_)
   {
      auto* lf = new VectorDomainLFIntegrator(*body_);
      lf->SetIntRule(&DomainRule());
      b_->AddDomainIntegrator(lf);
   }
   const Geometry::Type face = mesh_.Dimension() == 2 ? Geometry::SEGMENT :
                               Geometry::SQUARE;
   for (const auto& t : tractions_)
   {
      traction_markers_.push_back(std::make_unique<Array<int>>(Marker(t.first)));
      auto* lf = new VectorBoundaryLFIntegrator(*t.second);
      lf->SetIntRule(&rules_.Get(face, 2 * opts_.order + 1));
      b_->AddBoundaryIntegrator(lf, *traction_markers_.back());
   }

   cg_ = std::make_unique<CGSolver>(mesh_.GetComm());
   cg_->SetRelTol(opts_.rtol);
   cg_->SetAbsTol(opts_.atol);
   cg_->SetMaxIter(opts_.max_iter);
   cg_->SetPrintLevel(opts_.print_level);
   // The operator before the preconditioner: a Krylov solver forwards
   // SetOperator to its preconditioner, which BoomerAMG would refuse.
   cg_->SetOperator(*constrained_);
   cg_->iterative_mode = true; // x holds the Dirichlet values and the guess
   finalized_ = true;
}

void LinearElasticity::BuildMass()
{
   if (m_) { return; }
   INCNS_PROFILE("solid::mass_setup");
   m_ = std::make_unique<ParBilinearForm>(fes_.get());
   m_->SetAssemblyLevel(AssemblyLevel::PARTIAL);
   m_->AddDomainIntegrator(new VectorMassIntegrator(rho_, &DomainRule()));
   m_->Assemble();
   op_->SetMass(m_.get());
   m_diag_.SetSize(fes_->GetTrueVSize());
   m_diag_.UseDevice(true);
   m_->AssembleDiagonal(m_diag_);
}

void LinearElasticity::SetupPreconditioner(double c)
{
   INCNS_PROFILE("solid::pc_setup");
   pc_.reset();
   lor_matrix_.reset();
   if (opts_.preconditioner == ElasticityPC::Jacobi)
   {
      StiffnessDiagonal(diag_);
      if (c != 0.0) { diag_.Add(c, m_diag_); }
      if (support_)
      {
         Vector sd; // a boundary-sized matrix: rows without a diagonal entry
         HypreDiagonal(*support_, sd);
         diag_ += sd;
      }
      pc_ = std::make_unique<OperatorJacobiSmoother>(diag_, ess_tdofs_);
   }
   else
   {
      // The LOR matrix comes from a legacy form with its own integrators: MFEM
      // reuses a form's integrators on the LOR mesh with a GLL order-1 rule,
      // which must not touch the partially assembled ones.
      ParBilinearForm lor_form(fes_.get());
      if (!lor_lambda_)
      {
         // The LOR mesh splits each element into k^dim (MakeRefined order).
         const int k = opts_.order;
         int children = k;
         for (int d = 1; d < mesh_.Dimension(); ++d) { children *= k; }
         lor_lambda_ = Lame(opts_.lambda, children);
         lor_mu_ = Lame(opts_.mu, children);
      }
      auto* lor_integ = new ElasticityIntegrator(*lor_lambda_, *lor_mu_);
      lor_form.AddDomainIntegrator(lor_integ);
      if (c != 0.0)
      {
         lor_mass_.constant = c * opts_.density;
         lor_form.AddDomainIntegrator(new VectorMassIntegrator(lor_mass_));
      }
      if (support_)
      {
         lor_form.AddBoundaryIntegrator(new VectorMassIntegrator(support_k_),
                                        support_marker_);
      }
      {
         INCNS_PROFILE("solid::lor_assemble");
         // A copy: the matrix belongs to the discretization's own form, and
         // the LOR mesh and space are freed with it.
         ParLORDiscretization lor(lor_form, ess_tdofs_);
         lor_matrix_ = std::make_unique<HypreParMatrix>(
                          *lor.GetAssembledSystem().As<HypreParMatrix>());
      }
      auto amg = std::make_unique<HypreBoomerAMG>(*lor_matrix_);
      amg->SetSystemsOptions(mesh_.Dimension(), /*order_bynodes=*/true);
      amg->SetPrintLevel(0);
      amg->iterative_mode = false;
      // A symmetric V-cycle (CG): l1 symmetric Gauss-Seidel on a host,
      // l1-Jacobi on a GPU (precond/hypre_device.hpp).
      HYPRE_Solver h = static_cast<HYPRE_Solver>(*amg);
      HYPRE_BoomerAMGSetRelaxType(h, SymmetricAmgSmoother());
      HYPRE_BoomerAMGSetNumSweeps(h, 1);
      VerifyHypreAmgOnDevice(*amg, "linear_elasticity: the LOR-AMG");
      pc_ = std::move(amg);
   }
   cg_->SetPreconditioner(*pc_);
   pc_shift_ = c;
   pc_ready_ = true;
}

void LinearElasticity::AssembleLoad(double t, Vector& f)
{
   INCNS_PROFILE("solid::load");
   for (const auto& tr : tractions_) { tr.second->SetTime(t); }
   if (body_) { body_->SetTime(t); }
   b_->Assemble();
   b_->ParallelAssemble(f);
}

void LinearElasticity::ApplyDirichlet(double t, Vector& x)
{
   if (ess_tdofs_.Size() == 0 && displaced_.empty()) { return; }
   if (displaced_.empty())
   {
      x.SetSubVector(ess_tdofs_, 0.0);
      return;
   }
   // Zero (the fixed groups), then each prescribed group projected on its
   // faces (it wins at a node shared with a fixed face).
   ParGridFunction& g = *bc_gf_;
   g = 0.0;
   for (const auto& d : displaced_)
   {
      d.second->SetTime(t);
      Array<int> m = Marker(d.first);
      g.ProjectBdrCoefficient(*d.second, m);
   }
   g.GetTrueDofs(work3_);
   work3_.GetSubVector(ess_tdofs_, ess_vals_);
   x.SetSubVector(ess_tdofs_, ess_vals_);
}

bool LinearElasticity::SolveShifted(double c, Vector& b, Vector& x,
                                    double gamma)
{
   if (!pc_ready_ || c != pc_shift_) { SetupPreconditioner(c); }
   op_->SetWeights(1.0, c);
   if (track_reaction_) { reaction_b_ = b; }
   constrained_->EliminateRHS(x, b);
   {
      INCNS_PROFILE("solid::cg");
      cg_->Mult(b, x);
   }
   iterations_ = cg_->GetNumIterations();
   final_norm_ = cg_->GetFinalNorm();
   converged_ = cg_->GetConverged();
   total_iterations_ += iterations_;
   max_iterations_ = std::max(max_iterations_, iterations_);
   if (track_reaction_)
   {
      // R = (A x - b) at the essential dofs, less gamma R_previous.
      INCNS_PROFILE("solid::reaction");
      Vector& r = reaction_b_;
      work4_.SetSize(x.Size());
      work4_.UseDevice(true);
      op_->Mult(x, work4_);
      r.Neg();
      r += work4_;
      const double* rr = r.HostRead();
      double* out = reaction_.HostReadWrite();
      const int* ess = ess_tdofs_.HostRead();
      std::vector<double> vals(ess_tdofs_.Size());
      for (int i = 0; i < ess_tdofs_.Size(); ++i)
      {
         vals[i] = rr[ess[i]] - gamma * out[ess[i]];
      }
      reaction_.HostWrite();
      reaction_ = 0.0;
      out = reaction_.HostReadWrite();
      for (int i = 0; i < ess_tdofs_.Size(); ++i) { out[ess[i]] = vals[i]; }
   }
   return converged_;
}

void LinearElasticity::TrackReaction()
{
   track_reaction_ = true;
   reaction_.SetSize(fes_->GetTrueVSize());
   reaction_.UseDevice(true);
   reaction_ = 0.0;
}

void LinearElasticity::SyncFields()
{
   if (u_hist_.empty()) { return; }
   u_->SetFromTrueDofs(u_hist_[0]);
   v_gf_->SetFromTrueDofs(v_hist_[0]);
}

bool LinearElasticity::Solve() { return Solve(time_); }

bool LinearElasticity::Solve(double t)
{
   INCNS_PROFILE("solid::solve");
   Finalize();
   Vector& x = work2_;
   x = 0.0;
   ApplyDirichlet(t, x);
   AssembleLoad(t, load_);
   rhs_ = load_;
   const bool ok = SolveShifted(0.0, rhs_, x);
   u_->SetFromTrueDofs(x);
   return ok;
}

void LinearElasticity::DirichletValues(double t, Vector& g)
{
   Finalize();
   Vector& x = work4_;
   x.SetSize(fes_->GetTrueVSize());
   x = 0.0;
   ApplyDirichlet(t, x);
   x.GetSubVector(ess_tdofs_, g);
}

void LinearElasticity::SetSolution(const Vector& x)
{
   iterations_ = 0;
   converged_ = true;
   u_->SetFromTrueDofs(x);
}

bool LinearElasticity::Solve(double t, const Vector& guess)
{
   INCNS_PROFILE("solid::solve");
   Finalize();
   if (!pc_ready_ || pc_shift_ != 0.0) { SetupPreconditioner(0.0); }
   op_->SetWeights(1.0, 0.0);
   AssembleLoad(t, load_);
   // The zero guess's residual, in CG's (preconditioned) norm: the
   // reference of the tolerance.
   Vector& x = work2_;
   Vector& z = work4_;
   x.SetSize(guess.Size());
   x = 0.0;
   ApplyDirichlet(t, x);
   rhs_ = load_;
   constrained_->EliminateRHS(x, rhs_);
   z.SetSize(rhs_.Size());
   pc_->Mult(rhs_, z);
   const double nom0 = InnerProduct(mesh_.GetComm(), z, rhs_);
   if (nom0 <= opts_.atol * opts_.atol)
   {
      // The zero guess already meets the tolerance (zero data, for one):
      // its answer is the Dirichlet values alone, as CG would leave it.
      iterations_ = 0;
      converged_ = true;
      final_norm_ = std::sqrt(std::max(nom0, 0.0));
      u_->SetFromTrueDofs(x);
      return true;
   }
   // The solve from the guess.
   x = guess;
   ApplyDirichlet(t, x);
   rhs_ = load_;
   cg_->SetRelTol(0.0);
   cg_->SetAbsTol(std::max(opts_.atol, opts_.rtol * std::sqrt(std::max(nom0,
                           0.0))));
   const bool ok = SolveShifted(0.0, rhs_, x);
   cg_->SetRelTol(opts_.rtol);
   cg_->SetAbsTol(opts_.atol);
   u_->SetFromTrueDofs(x);
   return ok;
}

void LinearElasticity::Initialize(double t0, VectorCoefficient* u0,
                                  VectorCoefficient* v0)
{
   INCNS_PROFILE("solid::initialize");
   Finalize();
   BuildMass();
   const int n = fes_->GetTrueVSize();
   const bool bdf = opts_.scheme == SolidTimeScheme::Bdf;
   // BDF: the levels the order uses plus one for the new state;
   // generalized-alpha: u^n and u^{n+1}, v in place.
   const std::size_t levels = bdf ? opts_.bdf_order + 1 : 2;
   u_hist_.assign(levels, Vector());
   v_hist_.assign(bdf ? levels : 1, Vector());
   for (auto* h : {&u_hist_, &v_hist_})
   {
      for (Vector& v : *h)
      {
         v.SetSize(n);
         v.UseDevice(true);
         v = 0.0;
      }
   }
   t_hist_.assign(levels, t0);
   if (!v_gf_) { v_gf_ = std::make_unique<ParGridFunction>(fes_.get()); }

   ParGridFunction& g = *bc_gf_;
   if (u0)
   {
      u0->SetTime(t0);
      g.ProjectCoefficient(*u0);
      g.GetTrueDofs(u_hist_[0]);
   }
   ApplyDirichlet(t0, u_hist_[0]);
   if (v0)
   {
      v0->SetTime(t0);
      g.ProjectCoefficient(*v0);
      g.GetTrueDofs(v_hist_[0]);
   }
   f_n_.SetSize(n);
   f_n_.UseDevice(true);
   AssembleLoad(t0, f_n_);

   if (!bdf)
   {
      // M a0 = f(t0) - K u0, a0 = 0 on the Dirichlet boundary.
      INCNS_PROFILE("solid::initial_acceleration");
      a_n_.SetSize(n);
      a_n_.UseDevice(true);
      a_n_ = 0.0;
      op_->Apply(1.0, 0.0, u_hist_[0], work_);
      rhs_ = f_n_;
      rhs_ -= work_;
      op_->SetWeights(0.0, 1.0); // the mass alone (SolveShifted resets it)
      ConstrainedOperator cm(op_.get(), ess_tdofs_);
      cm.EliminateRHS(a_n_, rhs_);
      OperatorJacobiSmoother jac(m_diag_, ess_tdofs_);
      CGSolver cg(mesh_.GetComm());
      cg.SetRelTol(opts_.rtol);
      cg.SetAbsTol(opts_.atol);
      cg.SetMaxIter(opts_.max_iter);
      cg.SetPrintLevel(opts_.print_level);
      cg.SetOperator(cm);
      cg.SetPreconditioner(jac);
      cg.iterative_mode = true;
      cg.Mult(rhs_, a_n_);
      MFEM_VERIFY(cg.GetConverged(), "linear_elasticity: the initial "
                  "acceleration's mass solve did not converge");
   }
   time_ = t0;
   steps_ = 0;
   if (track_reaction_) { reaction_ = 0.0; }
   initialized_ = true;
   SyncFields();
}

bool LinearElasticity::Step(double dt)
{
   INCNS_PROFILE("solid::step");
   MFEM_VERIFY(dt > 0.0, "linear_elasticity: dt must be > 0");
   if (!initialized_) { Initialize(time_); }
   const double t1 = time_ + dt;
   bool ok = true;

   if (opts_.scheme == SolidTimeScheme::GeneralizedAlpha)
   {
      const GeneralizedAlphaCoefficients ga = GeneralizedAlpha(opts_.rho_inf);
      const double am = ga.alpha_m, af = ga.alpha_f, beta = ga.beta;
      const double bdt2 = beta * dt * dt;
      Vector& u = u_hist_[0];
      Vector& v = v_hist_[0];
      Vector& x = u_hist_[1];
      // w = u + dt v + dt^2 (1/2 - beta) a: the predictor of u^{n+1}.
      Vector& w = work_;
      w = u;
      w.Add(dt, v);
      w.Add(dt * dt * (0.5 - beta), a_n_);
      // (1 - af) [(1 - am)/(beta dt^2) M + (1 - af) K] u^{n+1} =
      //   (1 - af) f^{n+1} + af f^n - af K u^n
      //   + M((1 - am)/(beta dt^2) w - am a^n), divided by (1 - af).
      AssembleLoad(t1, load_);
      rhs_ = load_;
      op_->Apply(1.0, 0.0, u, work2_);
      work2_.Neg();
      work2_ += f_n_;
      rhs_.Add(af / (1.0 - af), work2_);
      work2_.Set((1.0 - am) / bdt2, w);
      work2_.Add(-am, a_n_);
      op_->Apply(0.0, 1.0, work2_, work3_);
      rhs_.Add(1.0 / (1.0 - af), work3_);
      x = u; // the initial guess
      ApplyDirichlet(t1, x);
      ok = SolveShifted((1.0 - am) / ((1.0 - af) * bdt2), rhs_, x,
                        af / (1.0 - af));
      // a^{n+1} = (u^{n+1} - w)/(beta dt^2); v^{n+1} by Newmark.
      work2_ = x;
      work2_ -= w;
      work2_ *= 1.0 / bdt2;
      v.Add(dt * (1.0 - ga.gamma), a_n_);
      v.Add(dt * ga.gamma, work2_);
      a_n_.Swap(work2_);
      f_n_ = load_; // load_ stays the current load (ExternalWork)
      u_hist_[0].Swap(u_hist_[1]);
      t_hist_[0] = t1;
   }
   else
   {
      const std::size_t L = u_hist_.size();
      Vector& x = u_hist_[L - 1]; // the spare level receives u^{n+1}
      Vector& vx = v_hist_[L - 1];
      const Vector& u = u_hist_[0];
      const Vector& v = v_hist_[0];
      AssembleLoad(t1, load_);
      if (steps_ == 0)
      {
         // Trapezoidal start: u1 - u0 = dt/2 (v0 + v1),
         // M (v1 - v0) = dt/2 (f0 + f1 - K u0 - K u1), so
         // (K + 4/dt^2 M) u1 = 4/dt^2 M (u0 + dt v0) + f0 + f1 - K u0.
         const double c = 4.0 / (dt * dt);
         work_ = u;
         work_.Add(dt, v);
         op_->Apply(0.0, c, work_, rhs_);
         rhs_ += load_;
         rhs_ += f_n_;
         op_->Apply(1.0, 0.0, u, work2_);
         rhs_ -= work2_;
         x = u;
         ApplyDirichlet(t1, x);
         ok = SolveShifted(c, rhs_, x, 1.0); // f0 + f1: R0 + R1
         vx = x;
         vx -= u;
         vx *= 2.0 / dt;
         vx -= v;
      }
      else
      {
         // v^{n+1} = c0 u^{n+1} + S_u, S_u = sum_{j>=1} c_j u^{n+1-j};
         // rho M (c0 v^{n+1} + S_v) + K u^{n+1} = f^{n+1}, so
         // (K + c0^2 M) u^{n+1} = f^{n+1} - M (c0 S_u + S_v).
         const int avail = std::min(steps_ + 1, static_cast<int>(L) - 1);
         const int q = std::min(opts_.bdf_order, avail);
         std::vector<double> times(q + 1);
         times[0] = t1;
         for (int j = 0; j < q; ++j) { times[j + 1] = t_hist_[j]; }
         const std::vector<double> c = BdfWeights(times);
         Vector& su = work_;
         Vector& sv = work2_;
         su.Set(c[1], u_hist_[0]);
         sv.Set(c[1], v_hist_[0]);
         for (int j = 2; j <= q; ++j)
         {
            su.Add(c[j], u_hist_[j - 1]);
            sv.Add(c[j], v_hist_[j - 1]);
         }
         sv.Add(c[0], su);
         op_->Apply(0.0, 1.0, sv, work3_);
         rhs_ = load_;
         rhs_ -= work3_;
         x = u;
         ApplyDirichlet(t1, x);
         ok = SolveShifted(c[0] * c[0], rhs_, x);
         vx.Set(c[0], x);
         vx += su;
      }
      // The new level to the front.
      for (std::size_t i = L - 1; i > 0; --i)
      {
         u_hist_[i].Swap(u_hist_[i - 1]);
         v_hist_[i].Swap(v_hist_[i - 1]);
         std::swap(t_hist_[i], t_hist_[i - 1]);
      }
      t_hist_[0] = t1;
      f_n_ = load_;
   }
   time_ = t1;
   ++steps_;
   SyncFields();
   return ok;
}

LinearElasticity::State LinearElasticity::ExportState() const
{
   State s;
   s.u_hist = u_hist_;
   s.v_hist = v_hist_;
   s.t_hist = t_hist_;
   s.a_n = a_n_;
   s.f_n = f_n_;
   s.load = load_;
   s.time = time_;
   s.steps = steps_;
   s.initialized = initialized_;
   s.reaction = reaction_;
   return s;
}

void LinearElasticity::ImportState(const State& s)
{
   MFEM_VERIFY(s.u_hist.size() == u_hist_.size() || !initialized_,
               "linear_elasticity: the state is not this solver's");
   u_hist_ = s.u_hist;
   v_hist_ = s.v_hist;
   t_hist_ = s.t_hist;
   a_n_ = s.a_n;
   f_n_ = s.f_n;
   load_ = s.load;
   time_ = s.time;
   steps_ = s.steps;
   initialized_ = s.initialized;
   if (track_reaction_ && s.reaction.Size() == reaction_.Size())
   {
      reaction_ = s.reaction;
   }
   SyncFields();
}

double LinearElasticity::KineticEnergy() const
{
   if (v_hist_.empty()) { return 0.0; }
   op_->Apply(0.0, 1.0, v_hist_[0], ework_);
   return 0.5 * InnerProduct(mesh_.GetComm(), v_hist_[0], ework_);
}

double LinearElasticity::StrainEnergy() const
{
   Vector ut(fes_->GetTrueVSize());
   ut.UseDevice(true);
   u_->GetTrueDofs(ut);
   op_->Apply(1.0, 0.0, ut, ework_);
   return 0.5 * InnerProduct(mesh_.GetComm(), ut, ework_);
}

double LinearElasticity::ExternalWork() const
{
   if (!finalized_) { return 0.0; }
   Vector ut(fes_->GetTrueVSize());
   ut.UseDevice(true);
   u_->GetTrueDofs(ut);
   return InnerProduct(mesh_.GetComm(), load_, ut);
}

double LinearElasticity::MaxDisplacement() const
{
   const int dim = mesh_.Dimension();
   const int nd = fes_->GetNDofs();
   const double* u = u_->HostRead();
   double loc = 0.0;
   for (int i = 0; i < nd; ++i)
   {
      double s = 0.0;
      for (int c = 0; c < dim; ++c) { s += u[c * nd + i] * u[c * nd + i]; }
      loc = std::max(loc, s);
   }
   double glob = 0.0;
   MPI_Allreduce(&loc, &glob, 1, MPI_DOUBLE, MPI_MAX, mesh_.GetComm());
   return std::sqrt(glob);
}

bool LinearElasticity::DisplacementAt(const Vector& x, Vector& u) const
{
   const int dim = mesh_.Dimension();
   bool same = probe_cached_ && probe_x_.Size() == dim;
   for (int d = 0; same && d < dim; ++d) { same = probe_x_(d) == x(d); }
   if (!same)
   {
      DenseMatrix pts(dim, 1);
      for (int d = 0; d < dim; ++d) { pts(d, 0) = x(d); }
      Array<int> elems;
      Array<IntegrationPoint> ips;
      mesh_.FindPoints(pts, elems, ips, /*warn=*/false); // collective
      probe_elem_ = (elems.Size() > 0 && elems[0] >= 0) ? elems[0] : -1;
      if (probe_elem_ >= 0) { probe_ip_ = ips[0]; }
      probe_x_ = x;
      probe_cached_ = true;
   }
   u.SetSize(dim);
   u = 0.0;
   double found = 0.0;
   if (probe_elem_ >= 0)
   {
      u_->HostRead();
      u_->GetVectorValue(probe_elem_, probe_ip_, u);
      found = 1.0;
   }
   Vector sum(dim + 1);
   for (int d = 0; d < dim; ++d) { sum(d) = u(d); }
   sum(dim) = found;
   MPI_Allreduce(MPI_IN_PLACE, sum.GetData(), dim + 1, MPI_DOUBLE, MPI_SUM,
                 mesh_.GetComm());
   if (sum(dim) < 0.5) { u = 0.0; return false; }
   // ParMesh::FindPoints gives each point to one rank; divide in case not.
   for (int d = 0; d < dim; ++d) { u(d) = sum(d) / sum(dim); }
   return true;
}

} // namespace incns
