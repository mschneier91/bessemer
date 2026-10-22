#include "solid/hyperelasticity.hpp"

#include "solid/element_scaled_coefficient.hpp"

#include "precond/hypre_device.hpp"
#include "util/profiler.hpp"

#include <algorithm>
#include <cmath>

namespace incns
{

using namespace mfem;

Hyperelasticity::Hyperelasticity(ParMesh& mesh, const Options& opts)
   : mesh_(mesh), opts_(opts)
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
   for (Vector* v : {&load_, &x_, &r_, &dx_, &rhs_, &trial_})
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
   SetupPreconditioner();
   finalized_ = true;
}

void Hyperelasticity::SetupPreconditioner()
{
   INCNS_PROFILE("hyperelastic::pc_setup");
   // The tangent at u = 0 is the linear elasticity operator: its LOR-AMG on
   // the reference configuration, built once (LinearElasticity builds the same
   // one; a shared helper would need its bitwise check, spec section 5). A
   // legacy form with its own integrators: MFEM reuses them on the LOR mesh.
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

double Hyperelasticity::Residual(const Vector& x, double s, Vector& r)
{
   op_->Mult(x, r);
   r.Add(-s, load_);
   r.SetSubVector(ess_tdofs_, 0.0);
   return std::sqrt(InnerProduct(mesh_.GetComm(), r, r));
}

bool Hyperelasticity::Solve(double t)
{
   INCNS_PROFILE("hyperelastic::solve");
   Finalize();
   AssembleLoad(t);
   DirichletValues(t);
   u_->GetTrueDofs(x_);
   history_.clear();
   newton_iterations_ = 0;
   max_newton_ = 0;
   linear_iterations_ = 0;
   converged_ = true;
   const MPI_Comm comm = mesh_.GetComm();
   const int steps = opts_.load_steps;
   for (int st = 1; st <= steps && converged_; st++)
   {
      // Loads and Dirichlet data ramped together, from zero.
      const double s = static_cast<double>(st) / steps;
      ess_step_ = ess_full_;
      ess_step_ *= s;
      // The step's Dirichlet increment, taken by its first Newton step (the
      // increment prescribed on dx, the rest from the tangent). Moving the
      // Dirichlet dofs alone would strain a layer of boundary elements and
      // start Newton far from the solution.
      x_.GetSubVector(ess_tdofs_, ess_jump_);
      ess_jump_.Neg();
      ess_jump_ += ess_step_;
      bool lift = InnerProduct(comm, ess_jump_, ess_jump_) > 0.0;
      rhs_ = load_;
      rhs_.SetSubVector(ess_tdofs_, 0.0);
      const double fnorm = s * std::sqrt(InnerProduct(comm, rhs_, rhs_));
      double rnorm = Residual(x_, s, r_);
      history_.push_back(rnorm);
      double ref = std::max(rnorm, fnorm);
      double tol = std::max(opts_.newton_atol, opts_.newton_rtol * ref);
      int it = 0;
      while ((lift || rnorm > tol) && it < opts_.newton_max_iter)
      {
         // K_T dx = -r, dx the Dirichlet increment on the Dirichlet dofs (the
         // constrained tangent is the identity there; r vanishes there).
         Operator& K = op_->GetGradient(x_);
         if (!kc_)
         {
            // The tangent is one object (HyperelasticOperator::GetGradient),
            // so are its constrained form and CG's operator: set once, as CG
            // forwards SetOperator to BoomerAMG, which takes only matrices.
            kc_ = std::make_unique<ConstrainedOperator>(&K, ess_tdofs_);
            k_ = &K;
            cg_->SetOperator(*kc_);
            cg_->SetPreconditioner(*pc_);
         }
         MFEM_VERIFY(&K == k_, "hyperelasticity: the tangent moved");
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
         // Backtracking: the full step unless the residual fails to drop
         // (not on the increment's step, whose residual has new data).
         double alpha = 1.0;
         trial_ = x_;
         trial_.Add(alpha, dx_);
         if (lift) { trial_.SetSubVector(ess_tdofs_, ess_step_); }
         double trial_norm = Residual(trial_, s, r_);
         for (int h = 0; opts_.line_search && !lift && h < 10 &&
              trial_norm > (1.0 - 1e-4 * alpha) * rnorm; h++)
         {
            alpha *= 0.5;
            trial_ = x_;
            trial_.Add(alpha, dx_);
            trial_norm = Residual(trial_, s, r_);
         }
         x_ = trial_;
         rnorm = trial_norm;
         history_.push_back(rnorm);
         lift = false;
         it++;
      }
      newton_iterations_ += it;
      max_newton_ = std::max(max_newton_, it);
      converged_ = rnorm <= tol;
   }
   total_linear_ += linear_iterations_;
   u_->SetFromTrueDofs(x_);
   return converged_;
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

void Hyperelasticity::SetSolution(const Vector& x)
{
   u_->SetFromTrueDofs(x);
}

} // namespace incns
