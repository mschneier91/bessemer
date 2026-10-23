#include "solid/hyperelasticity.hpp"

#include "solid/element_scaled_coefficient.hpp"

#include "precond/hypre_device.hpp"
#include "time/multistep_coeffs.hpp"
#include "util/profiler.hpp"

#include <algorithm>
#include <cmath>

namespace incns
{

using namespace mfem;

/// theta K_T + c M on true dofs: the Newton operator of a load step (theta 1,
/// c 0) or a time step.
class Hyperelasticity::ShiftedTangent : public Operator
{
public:
   /// @param K The tangent (HyperelasticOperator::GetGradient's object).
   explicit ShiftedTangent(const Operator& K) : Operator(K.Height()), k_(K)
   {
      w_.SetSize(height);
      w_.UseDevice(true);
   }

   /**
    * @param theta Weight of K (and S). @param c Weight of M.
    * @param M The mass (c != 0). @param S The support, or null.
    */
   void Set(double theta, double c, const Operator* M, const HypreParMatrix* S)
   {
      theta_ = theta;
      c_ = c;
      m_ = M;
      s_ = S;
   }

   void Mult(const Vector& x, Vector& y) const override
   {
      k_.Mult(x, y);
      if (s_) { s_->Mult(1.0, x, 1.0, y); }
      if (theta_ != 1.0) { y *= theta_; }
      if (c_ != 0.0)
      {
         m_->Mult(x, w_);
         y.Add(c_, w_);
      }
   }

private:
   const Operator& k_;           ///< The tangent.
   const Operator* m_ = nullptr; ///< The mass (c != 0).
   const HypreParMatrix* s_ = nullptr; ///< The support (AddSupport).
   double theta_ = 1.0, c_ = 0.0; ///< The weights.
   mutable Vector w_;            ///< M x.
};

Hyperelasticity::Hyperelasticity(ParMesh& mesh, const Options& opts)
   : mesh_(mesh), opts_(opts), rho_(opts.density)
{
   INCNS_PROFILE("hyperelastic::setup");
   const int dim = mesh_.Dimension();
   MFEM_VERIFY(opts_.order >= 1, "hyperelasticity: order must be >= 1");
   MFEM_VERIFY(opts_.mu > 0.0 && opts_.lambda > -2.0 * opts_.mu / dim,
               "hyperelasticity: the material is not positive definite "
               "(mu > 0 and lambda > -2 mu / dim)");
   MFEM_VERIFY(opts_.load_steps >= 1, "hyperelasticity: load_steps must be >= 1");
   MFEM_VERIFY(opts_.newton_max_iter >= 1,
               "hyperelasticity: newton_max_iter must be >= 1");
   MFEM_VERIFY(opts_.density > 0.0, "hyperelasticity: density must be > 0");
   MFEM_VERIFY(opts_.rho_inf >= 0.0 && opts_.rho_inf <= 1.0,
               "hyperelasticity: rho_inf must be in [0, 1]");
   const int ne = mesh_.GetNE();
   if (opts_.element_stiffness)
   {
      MFEM_VERIFY(opts_.element_stiffness->Size() == ne,
                  "hyperelasticity: element_stiffness needs one factor per "
                  "local element");
      MFEM_VERIFY(ne == 0 || opts_.element_stiffness->Min() > 0.0,
                  "hyperelasticity: element_stiffness must be positive");
   }
   // GLL nodal basis (the LOR preconditioner refines on the GLL nodes),
   // ordered by nodes (HyperelasticOperator's layout).
   fec_ = std::make_unique<H1_FECollection>(opts_.order, dim);
   fes_ = std::make_unique<ParFiniteElementSpace>(&mesh_, fec_.get(), dim,
          Ordering::byNODES);
   lambda_e_.SetSize(ne);
   mu_e_.SetSize(ne);
   for (int e = 0; e < ne; e++)
   {
      const double s = opts_.element_stiffness ? (*opts_.element_stiffness)(e) : 1.0;
      lambda_e_(e) = s * opts_.lambda;
      mu_e_(e) = s * opts_.mu;
   }
   op_ = std::make_unique<HyperelasticOperator>(*fes_, DomainRule(),
         opts_.material, lambda_e_, mu_e_,
         opts_.storage);
   u_ = std::make_unique<ParGridFunction>(fes_.get());
   *u_ = 0.0;
   const int n = fes_->GetTrueVSize();
   for (Vector* v :
        {
           &load_, &x_, &r_, &dx_, &rhs_, &trial_, &step_rhs_, &w_,
           &z_, &mx_
        })
   {
      v->SetSize(n);
      v->UseDevice(true);
      *v = 0.0;
   }
}

Hyperelasticity::~Hyperelasticity()
{
   // The operator's geometry came from factors keyed by rules_' rules.
   mesh_.DeleteGeometricFactors();
}

const IntegrationRule& Hyperelasticity::DomainRule() const
{
   const Geometry::Type g = mesh_.Dimension() == 2 ? Geometry::SQUARE :
                            Geometry::CUBE;
   return rules_.Get(g, 2 * opts_.order + 1);
}

Array<int> Hyperelasticity::Marker(int attr) const
{
   const int n = mesh_.bdr_attributes.Size() ? mesh_.bdr_attributes.Max() : 0;
   MFEM_VERIFY(attr >= 1 && attr <= n, "hyperelasticity: boundary attribute "
               << attr << " is not on the mesh (1.." << n << ")");
   Array<int> m(n);
   m = 0;
   m[attr - 1] = 1;
   return m;
}

void Hyperelasticity::AddFixed(int attr)
{
   MFEM_VERIFY(!finalized_, "hyperelasticity: boundary conditions are fixed "
               "at the first solve");
   Marker(attr); // checks the attribute
   fixed_.push_back(attr);
}

void Hyperelasticity::AddDisplacement(int attr, VectorCoefficient& g)
{
   MFEM_VERIFY(!finalized_, "hyperelasticity: boundary conditions are fixed "
               "at the first solve");
   Marker(attr);
   displaced_.push_back({attr, &g});
}

void Hyperelasticity::AddTraction(int attr, VectorCoefficient& t)
{
   MFEM_VERIFY(!finalized_, "hyperelasticity: boundary conditions are fixed "
               "at the first solve");
   Marker(attr);
   tractions_.push_back({attr, &t});
}

void Hyperelasticity::AddSupport(int attr, double k)
{
   MFEM_VERIFY(!finalized_, "hyperelasticity: boundary conditions are fixed "
               "at the first solve");
   MFEM_VERIFY(k > 0.0, "hyperelasticity: the support stiffness must be "
               "positive");
   MFEM_VERIFY(support_k_.constant == 0.0 || support_k_.constant == k,
               "hyperelasticity: one support stiffness for all attributes");
   const Array<int> m = Marker(attr);
   if (support_marker_.Size() == 0)
   {
      support_marker_.SetSize(m.Size());
      support_marker_ = 0;
   }
   for (int i = 0; i < m.Size(); ++i) { if (m[i]) { support_marker_[i] = 1; } }
   support_k_.constant = k;
}

void Hyperelasticity::SetBodyForce(VectorCoefficient& f)
{
   MFEM_VERIFY(!finalized_, "hyperelasticity: the load is fixed at the first "
               "solve");
   body_ = &f;
}

void Hyperelasticity::Finalize()
{
   if (finalized_) { return; }
   const int n = mesh_.bdr_attributes.Size() ? mesh_.bdr_attributes.Max() : 0;
   Array<int> ess(n);
   ess = 0;
   for (int a : fixed_) { ess[a - 1] = 1; }
   for (const auto& d : displaced_) { ess[d.first - 1] = 1; }
   fes_->GetEssentialTrueDofs(ess, ess_tdofs_);
   ess_tdofs_.GetMemory().UseDevice(true); // read on the device (SetSubVector)
   for (Vector* v : {&ess_full_, &ess_step_, &ess_jump_})
   {
      v->SetSize(ess_tdofs_.Size());
      v->UseDevice(true);
      *v = 0.0;
   }
   bc_gf_ = std::make_unique<ParGridFunction>(fes_.get());
   if (support_marker_.Size() > 0)
   {
      // Assembled, as LinearElasticity's: a boundary-sized matrix.
      ParBilinearForm s(fes_.get());
      s.AddBoundaryIntegrator(new VectorMassIntegrator(support_k_),
                              support_marker_);
      s.Assemble();
      s.Finalize();
      support_.reset(s.ParallelAssemble());
   }

   // The load form: dead loads, re-evaluated at every assembly.
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
   cg_->iterative_mode = false; // each Newton correction from zero
   SetupPreconditioner(0.0);
   finalized_ = true;
}

void Hyperelasticity::SetupPreconditioner(double shift)
{
   INCNS_PROFILE("hyperelastic::pc_setup");
   // The tangent at u = 0 is the linear elasticity operator: its LOR-AMG on
   // the reference configuration (plus shift rho M in dynamics), built once
   // per shift (LinearElasticity builds the same one; a shared helper would
   // need its bitwise check, spec section 5). A legacy form with its own
   // integrators: MFEM reuses them on the LOR mesh.
   pc_.reset();
   lor_matrix_.reset();
   const int dim = mesh_.Dimension(), k = opts_.order;
   int children = k;
   for (int d = 1; d < dim; ++d) { children *= k; }
   if (opts_.element_stiffness)
   {
      lor_lambda_ = std::make_unique<ElementScaledCoefficient>(
                       opts_.lambda, *opts_.element_stiffness, children);
      lor_mu_ = std::make_unique<ElementScaledCoefficient>(
                   opts_.mu, *opts_.element_stiffness, children);
   }
   else
   {
      lor_lambda_ = std::make_unique<ConstantCoefficient>(opts_.lambda);
      lor_mu_ = std::make_unique<ConstantCoefficient>(opts_.mu);
   }
   ParBilinearForm lor_form(fes_.get());
   lor_form.AddDomainIntegrator(new ElasticityIntegrator(*lor_lambda_, *lor_mu_));
   if (support_marker_.Size() > 0)
   {
      lor_form.AddBoundaryIntegrator(new VectorMassIntegrator(support_k_),
                                     support_marker_);
   }
   if (shift != 0.0)
   {
      lor_mass_.constant = shift * opts_.density;
      lor_form.AddDomainIntegrator(new VectorMassIntegrator(lor_mass_));
   }
   {
      ParLORDiscretization lor(lor_form, ess_tdofs_);
      lor_matrix_ = std::make_unique<HypreParMatrix>(
                       *lor.GetAssembledSystem().As<HypreParMatrix>());
   }
   auto amg = std::make_unique<HypreBoomerAMG>(*lor_matrix_);
   amg->SetSystemsOptions(dim, /*order_bynodes=*/true);
   amg->SetPrintLevel(0);
   amg->iterative_mode = false;
   HYPRE_Solver h = static_cast<HYPRE_Solver>(*amg);
   HYPRE_BoomerAMGSetRelaxType(h, SymmetricAmgSmoother());
   HYPRE_BoomerAMGSetNumSweeps(h, 1);
   VerifyHypreAmgOnDevice(*amg, "hyperelasticity: the LOR-AMG");
   pc_ = std::move(amg);
   pc_shift_ = shift;
}

void Hyperelasticity::AssembleLoad(double t)
{
   INCNS_PROFILE("hyperelastic::load");
   for (const auto& tr : tractions_) { tr.second->SetTime(t); }
   if (body_) { body_->SetTime(t); }
   b_->Assemble();
   b_->ParallelAssemble(load_);
}

void Hyperelasticity::DirichletValues(double t)
{
   if (displaced_.empty())
   {
      ess_full_ = 0.0;
      return;
   }
   // Zero (the fixed groups), then each prescribed group projected on its
   // faces (it wins at a node shared with a fixed face), as LinearElasticity.
   ParGridFunction& g = *bc_gf_;
   g = 0.0;
   for (const auto& d : displaced_)
   {
      d.second->SetTime(t);
      Array<int> m = Marker(d.first);
      g.ProjectBdrCoefficient(*d.second, m);
   }
   g.GetTrueDofs(trial_);
   trial_.GetSubVector(ess_tdofs_, ess_full_);
}

void Hyperelasticity::ResetCounters()
{
   history_.clear();
   corrections_.clear();
   newton_iterations_ = 0;
   max_newton_ = 0;
   linear_iterations_ = 0;
}

double Hyperelasticity::Residual(const Vector& x, Vector& r)
{
   // r = c M x + R(z) - b, z = theta x + (1 - theta) y.
   const Vector* z = &x;
   if (g_y_)
   {
      add(g_theta_, x, 1.0 - g_theta_, *g_y_, z_);
      z = &z_;
   }
   op_->Mult(*z, r);
   if (support_) { support_->Mult(1.0, *z, 1.0, r); }
   if (g_c_ != 0.0)
   {
      m_->Mult(x, mx_);
      r.Add(g_c_, mx_);
   }
   r -= *g_b_;
   r.SetSubVector(ess_tdofs_, 0.0);
   return std::sqrt(InnerProduct(mesh_.GetComm(), r, r));
}

bool Hyperelasticity::Newton(double c, double theta, const Vector* y,
                             const Vector& b, const Vector& g, Vector& x)
{
   const MPI_Comm comm = mesh_.GetComm();
   MFEM_VERIFY(theta > 0.0 && (y || theta == 1.0),
               "hyperelasticity: Newton needs u^n when theta < 1");
   g_c_ = c;
   g_theta_ = theta;
   g_y_ = y;
   g_b_ = &b;
   // The preconditioner approximates theta (K_0 + (c / theta) rho M).
   const double shift = c / theta;
   if (shift != pc_shift_)
   {
      SetupPreconditioner(shift);
      if (kc_) { cg_->SetPreconditioner(*pc_); }
   }
   // The Dirichlet increment, taken by the first Newton step (the increment
   // prescribed on dx, the rest from the tangent). Moving the Dirichlet dofs
   // alone would strain a layer of boundary elements and start Newton far
   // from the solution.
   x.GetSubVector(ess_tdofs_, ess_jump_);
   ess_jump_.Neg();
   ess_jump_ += g;
   bool lift = InnerProduct(comm, ess_jump_, ess_jump_) > 0.0;
   rhs_ = b;
   rhs_.SetSubVector(ess_tdofs_, 0.0);
   const double fnorm = std::sqrt(InnerProduct(comm, rhs_, rhs_));
   double rnorm = Residual(x, r_);
   history_.push_back(rnorm);
   double ref = std::max(rnorm, fnorm);
   double tol = std::max(opts_.newton_atol, opts_.newton_rtol * ref);
   int it = 0;
   bool small = false; // the last correction below newton_dx_rtol
   while ((lift || (rnorm > tol && !small)) && it < opts_.newton_max_iter)
   {
      // J dx = -r, dx the Dirichlet increment on the Dirichlet dofs (the
      // constrained J is the identity there; r vanishes there).
      const Vector* z = &x;
      if (y)
      {
         add(theta, x, 1.0 - theta, *y, z_);
         z = &z_;
      }
      Operator& K = op_->GetGradient(*z);
      if (!kc_)
      {
         // The tangent is one object (HyperelasticOperator::GetGradient),
         // so are J, its constrained form and CG's operator: set once, as CG
         // forwards SetOperator to BoomerAMG, which takes only matrices.
         jac_ = std::make_unique<ShiftedTangent>(K);
         kc_ = std::make_unique<ConstrainedOperator>(jac_.get(), ess_tdofs_);
         k_ = &K;
         cg_->SetOperator(*kc_);
         cg_->SetPreconditioner(*pc_);
      }
      MFEM_VERIFY(&K == k_, "hyperelasticity: the tangent moved");
      jac_->Set(theta, c, c != 0.0 ? m_.Ptr() : nullptr, support_.get());
      rhs_ = r_;
      rhs_.Neg();
      if (lift)
      {
         // The increment's pull on the free dofs joins the right-hand
         // side, and its norm the step's reference residual.
         dx_ = 0.0;
         dx_.SetSubVector(ess_tdofs_, ess_jump_);
         kc_->EliminateRHS(dx_, rhs_);
         trial_ = rhs_;
         trial_.SetSubVector(ess_tdofs_, 0.0);
         ref = std::max(ref, std::sqrt(InnerProduct(comm, trial_, trial_)));
         tol = std::max(opts_.newton_atol, opts_.newton_rtol * ref);
      }
      {
         INCNS_PROFILE("hyperelastic::cg");
         cg_->Mult(rhs_, dx_);
      }
      linear_iterations_ += cg_->GetNumIterations();
      const double dxnorm = std::sqrt(InnerProduct(comm, dx_, dx_));
      // Backtracking: the full step unless the residual fails to drop
      // (not on the increment's step, whose residual has new data).
      double alpha = 1.0;
      trial_ = x;
      trial_.Add(alpha, dx_);
      if (lift) { trial_.SetSubVector(ess_tdofs_, g); }
      double trial_norm = Residual(trial_, r_);
      for (int h = 0; opts_.line_search && !lift && h < 10 &&
           trial_norm > (1.0 - 1e-4 * alpha) * rnorm; h++)
      {
         alpha *= 0.5;
         trial_ = x;
         trial_.Add(alpha, dx_);
         trial_norm = Residual(trial_, r_);
      }
      x = trial_;
      rnorm = trial_norm;
      history_.push_back(rnorm);
      // The full correction against the displacement: near the residual's
      // roundoff floor it still falls quadratically (section 4 of
      // docs/hyperelasticity.md).
      const double xnorm = std::sqrt(InnerProduct(comm, x, x));
      const double rel = xnorm > 0.0 ? dxnorm / xnorm : 0.0;
      corrections_.push_back(rel);
      // Or against an absolute size (Options::newton_dx_atol).
      const double rms = dxnorm / std::sqrt(static_cast<double>
                                            (fes_->GlobalTrueVSize()));
      small = !lift && ((opts_.newton_dx_rtol > 0.0 &&
                         rel <= opts_.newton_dx_rtol) ||
                        (opts_.newton_dx_atol > 0.0 && rms <= opts_.newton_dx_atol));
      lift = false;
      it++;
   }
   newton_iterations_ += it;
   total_newton_ += it;
   max_newton_ = std::max(max_newton_, it);
   return rnorm <= tol || small;
}

bool Hyperelasticity::Solve(double t)
{
   INCNS_PROFILE("hyperelastic::solve");
   Finalize();
   AssembleLoad(t);
   DirichletValues(t);
   u_->GetTrueDofs(x_);
   ResetCounters();
   converged_ = true;
   const int steps = opts_.load_steps;
   for (int st = 1; st <= steps && converged_; st++)
   {
      // Loads and Dirichlet data ramped together, from zero.
      const double s = static_cast<double>(st) / steps;
      ess_step_ = ess_full_;
      ess_step_ *= s;
      step_rhs_ = load_;
      step_rhs_ *= s;
      converged_ = Newton(0.0, 1.0, nullptr, step_rhs_, ess_step_, x_);
      UpdateReaction(x_, 0.0, 1.0);
   }
   total_linear_ += linear_iterations_;
   u_->SetFromTrueDofs(x_);
   return converged_;
}

void Hyperelasticity::BuildMass()
{
   if (m_form_) { return; }
   INCNS_PROFILE("hyperelastic::mass_setup");
   m_form_ = std::make_unique<ParBilinearForm>(fes_.get());
   m_form_->SetAssemblyLevel(AssemblyLevel::PARTIAL);
   m_form_->AddDomainIntegrator(new VectorMassIntegrator(rho_, &DomainRule()));
   m_form_->Assemble();
   Array<int> none;
   m_form_->FormSystemMatrix(none, m_); // P^T M P, no constraints
   m_diag_.SetSize(fes_->GetTrueVSize());
   m_diag_.UseDevice(true);
   m_form_->AssembleDiagonal(m_diag_);
}

void Hyperelasticity::SyncFields()
{
   if (u_hist_.empty()) { return; }
   u_->SetFromTrueDofs(u_hist_[0]);
   v_gf_->SetFromTrueDofs(v_hist_[0]);
}

void Hyperelasticity::Initialize(double t0, VectorCoefficient* u0,
                                 VectorCoefficient* v0)
{
   INCNS_PROFILE("hyperelastic::initialize");
   Finalize();
   BuildMass();
   const int n = fes_->GetTrueVSize();
   const bool bdf = opts_.scheme == SolidTimeScheme::Bdf;
   // BDF2: two levels plus one for the new state; generalized-alpha: u^n and
   // u^{n+1}, v in place.
   const std::size_t levels = bdf ? 3 : 2;
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
   DirichletValues(t0);
   u_hist_[0].SetSubVector(ess_tdofs_, ess_full_);
   if (v0)
   {
      v0->SetTime(t0);
      g.ProjectCoefficient(*v0);
      g.GetTrueDofs(v_hist_[0]);
   }
   AssembleLoad(t0);
   f_n_ = load_;

   if (!bdf)
   {
      // M a0 = f(t0) - R(u0), a0 = 0 on the Dirichlet boundary.
      INCNS_PROFILE("hyperelastic::initial_acceleration");
      a_n_.SetSize(n);
      a_n_.UseDevice(true);
      a_n_ = 0.0;
      op_->Mult(u_hist_[0], rhs_);
      rhs_.Neg();
      rhs_ += f_n_;
      ConstrainedOperator cm(m_.Ptr(), ess_tdofs_);
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
      MFEM_VERIFY(cg.GetConverged(), "hyperelasticity: the initial "
                  "acceleration's mass solve did not converge");
   }
   if (track_reaction_) { reaction_ = 0.0; } // a dynamic run starts from rest
   time_ = t0;
   steps_ = 0;
   initialized_ = true;
   SyncFields();
}

bool Hyperelasticity::Step(double dt)
{
   INCNS_PROFILE("hyperelastic::step");
   MFEM_VERIFY(dt > 0.0, "hyperelasticity: dt must be > 0");
   if (!initialized_) { Initialize(time_); }
   const double t1 = time_ + dt;
   ResetCounters();
   AssembleLoad(t1);
   DirichletValues(t1);
   bool ok = true;

   if (opts_.scheme == SolidTimeScheme::GeneralizedAlpha)
   {
      const GeneralizedAlphaCoefficients ga = GeneralizedAlpha(opts_.rho_inf);
      const double am = ga.alpha_m, af = ga.alpha_f, beta = ga.beta;
      const double bdt2 = beta * dt * dt;
      Vector& u = u_hist_[0];
      Vector& v = v_hist_[0];
      Vector& x = u_hist_[1];
      // w = u + dt v + dt^2 (1/2 - beta) a: the predictor of u^{n+1}, and
      // a^{n+1} = (u^{n+1} - w) / (beta dt^2). The balance at the alpha
      // points, M a_{n+1-am} + R(u_{n+1-af}) = f_{n+1-af}, is G(u^{n+1}) = 0
      // with c = (1 - am) / (beta dt^2), theta = 1 - af and
      // b = (1 - af) f^{n+1} + af f^n + M (c w - am a^n).
      w_ = u;
      w_.Add(dt, v);
      w_.Add(dt * dt * (0.5 - beta), a_n_);
      const double c = (1.0 - am) / bdt2;
      add(1.0 - af, load_, af, f_n_, step_rhs_);
      add(c, w_, -am, a_n_, z_);
      m_->Mult(z_, mx_);
      step_rhs_ += mx_;
      x = u; // the initial guess
      ok = Newton(c, 1.0 - af, &u, step_rhs_, ess_full_, x);
      UpdateReaction(x, af, 1.0 / (1.0 - af));
      // a^{n+1}; v^{n+1} by Newmark.
      z_ = x;
      z_ -= w_;
      z_ *= 1.0 / bdt2;
      v.Add(dt * (1.0 - ga.gamma), a_n_);
      v.Add(dt * ga.gamma, z_);
      a_n_.Swap(z_);
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
      if (steps_ == 0)
      {
         // Trapezoidal start: u1 - u0 = dt/2 (v0 + v1),
         // M (v1 - v0) = dt/2 (f0 + f1 - R(u0) - R(u1)), so with c = 4/dt^2,
         // c M u1 + R(u1) = c M (u0 + dt v0) + f0 + f1 - R(u0).
         const double c = 4.0 / (dt * dt);
         w_ = u;
         w_.Add(dt, v);
         m_->Mult(w_, step_rhs_);
         step_rhs_ *= c;
         step_rhs_ += load_;
         step_rhs_ += f_n_;
         op_->Mult(u, mx_);
         step_rhs_ -= mx_;
         x = u;
         ok = Newton(c, 1.0, nullptr, step_rhs_, ess_full_, x);
         UpdateReaction(x, 1.0, 1.0); // R^0 + R^1
         vx = x;
         vx -= u;
         vx *= 2.0 / dt;
         vx -= v;
      }
      else
      {
         // v^{n+1} = c0 u^{n+1} + S_u, S_u = c1 u^n + c2 u^{n-1};
         // M (c0 v^{n+1} + S_v) + R(u^{n+1}) = f^{n+1}, so
         // c0^2 M u^{n+1} + R(u^{n+1}) = f^{n+1} - M (c0 S_u + S_v).
         const std::vector<double> c = BdfWeights({t1, t_hist_[0], t_hist_[1]});
         Vector& su = w_;
         Vector& sv = z_;
         su.Set(c[1], u_hist_[0]);
         su.Add(c[2], u_hist_[1]);
         sv.Set(c[1], v_hist_[0]);
         sv.Add(c[2], v_hist_[1]);
         sv.Add(c[0], su);
         m_->Mult(sv, mx_);
         step_rhs_ = load_;
         step_rhs_ -= mx_;
         x = u;
         ok = Newton(c[0] * c[0], 1.0, nullptr, step_rhs_, ess_full_, x);
         UpdateReaction(x, 0.0, 1.0);
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
   }
   f_n_ = load_; // load_ stays the current load (ExternalWork)
   converged_ = ok;
   total_linear_ += linear_iterations_;
   time_ = t1;
   ++steps_;
   SyncFields();
   return ok;
}

double Hyperelasticity::KineticEnergy() const
{
   if (v_hist_.empty()) { return 0.0; }
   Vector mv(v_hist_[0].Size());
   mv.UseDevice(true);
   m_->Mult(v_hist_[0], mv);
   return 0.5 * InnerProduct(mesh_.GetComm(), v_hist_[0], mv);
}

double Hyperelasticity::StrainEnergy() const
{
   Vector x(fes_->GetTrueVSize());
   x.UseDevice(true);
   u_->GetTrueDofs(x);
   return op_->Energy(x);
}

double Hyperelasticity::ExternalWork() const
{
   Vector x(fes_->GetTrueVSize());
   x.UseDevice(true);
   u_->GetTrueDofs(x);
   return InnerProduct(mesh_.GetComm(), load_, x);
}

double Hyperelasticity::MaxDisplacement() const
{
   const int n = fes_->GetNDofs(), dim = fes_->GetVDim();
   const double* u = u_->HostRead();
   double m = 0.0;
   for (int j = 0; j < n; j++)
   {
      double s = 0.0;
      for (int c = 0; c < dim; c++) { s += u[j + c * n] * u[j + c * n]; }
      m = std::max(m, s);
   }
   MPI_Allreduce(MPI_IN_PLACE, &m, 1, MPI_DOUBLE, MPI_MAX, mesh_.GetComm());
   return std::sqrt(m);
}

bool Hyperelasticity::DisplacementAt(const Vector& x, Vector& u) const
{
   // As LinearElasticity::DisplacementAt.
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

void Hyperelasticity::SetSolution(const Vector& x)
{
   u_->SetFromTrueDofs(x);
}

void Hyperelasticity::UpdateReaction(const Vector& x, double gamma,
                                     double scale)
{
   if (!track_reaction_) { return; }
   INCNS_PROFILE("hyperelastic::reaction");
   // G(x) with the essential rows kept: Residual() without its zeroing.
   const Vector* z = &x;
   if (g_y_)
   {
      add(g_theta_, x, 1.0 - g_theta_, *g_y_, z_);
      z = &z_;
   }
   op_->Mult(*z, r_);
   if (support_) { support_->Mult(1.0, *z, 1.0, r_); }
   if (g_c_ != 0.0)
   {
      m_->Mult(x, mx_);
      r_.Add(g_c_, mx_);
   }
   r_ -= *g_b_;
   // On the host, as LinearElasticity's: a boundary-sized loop.
   const double* rr = r_.HostRead();
   double* out = reaction_.HostReadWrite();
   const int* ess = ess_tdofs_.HostRead();
   std::vector<double> vals(ess_tdofs_.Size());
   for (int i = 0; i < ess_tdofs_.Size(); ++i)
   {
      vals[i] = scale * (rr[ess[i]] - gamma * out[ess[i]]);
   }
   reaction_.HostWrite();
   reaction_ = 0.0;
   out = reaction_.HostReadWrite();
   for (int i = 0; i < ess_tdofs_.Size(); ++i) { out[ess[i]] = vals[i]; }
}

void Hyperelasticity::TrackReaction()
{
   track_reaction_ = true;
   reaction_.SetSize(fes_->GetTrueVSize());
   reaction_.UseDevice(true);
   reaction_ = 0.0;
}

Hyperelasticity::State Hyperelasticity::ExportState() const
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

void Hyperelasticity::ImportState(const State& s)
{
   MFEM_VERIFY(s.u_hist.size() == u_hist_.size() || !initialized_,
               "hyperelasticity: the state is not this solver's");
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

} // namespace incns
