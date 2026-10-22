#include "solid/elasticity_case.hpp"

#include "bc/boundary_names.hpp"
#include "bc/deck_boundary.hpp"
#include "config/expression.hpp"
#include "util/json.hpp"
#include "util/profiler.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <type_traits>

namespace incns
{

using namespace mfem;

LinearElasticity::Options ElasticityOptions(const Parameters& p)
{
   const ElasticityParameters& s = p.solid;
   LinearElasticity::Options o;
   o.order = s.Order(p.order_u);
   o.lambda = s.Lambda(p.mesh.dim);
   o.mu = s.Mu();
   o.rtol = s.rtol;
   o.atol = s.atol;
   o.max_iter = s.max_iter;
   o.preconditioner = s.preconditioner;
   o.density = s.density;
   o.scheme = s.time_scheme;
   o.bdf_order = s.bdf_order;
   o.rho_inf = s.rho_inf;
   return o;
}

Hyperelasticity::Options HyperelasticOptions(const Parameters& p)
{
   const ElasticityParameters& s = p.solid;
   Hyperelasticity::Options o;
   o.order = s.Order(p.order_u);
   o.lambda = s.Lambda(p.mesh.dim); // plane strain (Parameters checks)
   o.mu = s.Mu();
   switch (s.model)
   {
      case SolidModel::StVenantKirchhoff:
         o.material = HyperelasticMaterial::StVenantKirchhoff;
         break;
   }
   o.newton_rtol = s.newton_rtol;
   o.newton_atol = s.newton_atol;
   o.newton_dx_rtol = s.newton_dx_rtol;
   o.newton_max_iter = s.newton_max_iter;
   o.load_steps = s.load_steps;
   o.line_search = s.line_search;
   o.rtol = s.rtol;
   o.atol = s.atol;
   o.max_iter = s.max_iter;
   o.density = s.density;
   o.scheme = s.time_scheme;
   o.rho_inf = s.rho_inf;
   return o;
}

ElasticityCase::ElasticityCase(ParMesh& mesh, const Parameters& params)
   : mesh_(mesh), params_(params)
{
   MFEM_VERIFY(IsSolid(params_.equation), "elasticity_case: the deck's "
               "equation is not linear_elasticity or hyperelasticity");
   MFEM_VERIFY(params_.nondim.normalized,
               "elasticity_case: call Parameters::Normalize() first (LoadYAML "
               "does it)");
   if (params_.equation == Equation::Hyperelasticity)
   {
      hyper_ = std::make_unique<Hyperelasticity>(mesh_, HyperelasticOptions(params_));
      ApplyDeck(*hyper_);
   }
   else
   {
      solver_ = std::make_unique<LinearElasticity>(mesh_, ElasticityOptions(params_));
      ApplyDeck(*solver_);
   }
}

template <class SolidSolver>
void ElasticityCase::ApplyDeck(SolidSolver& solver)
{
   const int dim = mesh_.Dimension();

   // Every real boundary in exactly one group, as for a flow deck.
   {
      DeckBoundaryConditions deck(params_, mesh_);
      const std::vector<std::string> problems = deck.CoverageProblems();
      if (!problems.empty())
      {
         std::string msg;
         for (const std::string& s : problems) { msg += "\n  " + s; }
         MFEM_ABORT("boundary_conditions: every real boundary needs exactly one "
                    "group:" << msg);
      }
   }
   const std::vector<int> all = AllBoundaryAttributes(params_, mesh_);
   for (const BcSpec& s : params_.boundary_conditions)
   {
      std::vector<int> attrs = s.select_all ? all : s.attributes;
      if (!s.select_all)
      {
         for (const std::string& name : s.faces)
         {
            attrs.push_back(NamedBoundaryAttribute(params_, mesh_, name));
         }
      }
      Vector v(dim);
      for (int d = 0; d < dim; ++d) { v(d) = s.value[d]; }
      switch (s.type)
      {
         case BcType::Fixed:
            for (int a : attrs) { solver.AddFixed(a); }
            break;
         case BcType::Displacement:
            if (s.expression.empty())
            {
               coeffs_.push_back(std::make_unique<VectorConstantCoefficient>(v));
            }
            else
            {
               coeffs_.push_back(VectorExpression::Coefficient(
                                    std::make_shared<VectorExpression>(
                                       s.expression, s.expression_parameters)));
            }
            for (int a : attrs) { solver.AddDisplacement(a, *coeffs_.back()); }
            break;
         case BcType::Traction:
            coeffs_.push_back(std::make_unique<VectorConstantCoefficient>(v));
            for (int a : attrs) { solver.AddTraction(a, *coeffs_.back()); }
            break;
         case BcType::Free:
            break; // natural: zero traction
         default:
            MFEM_ABORT("elasticity_case: a flow boundary type in a solid case");
      }
   }

   const std::array<double, 3>& f = params_.solid.body_force;
   if (f[0] != 0.0 || f[1] != 0.0 || f[2] != 0.0)
   {
      Vector v(dim);
      for (int d = 0; d < dim; ++d) { v(d) = f[d]; }
      coeffs_.push_back(std::make_unique<VectorConstantCoefficient>(v));
      solver.SetBodyForce(*coeffs_.back());
   }
}

int ElasticityCase::Order() const
{
   return hyper_ ? hyper_->GetOptions().order : solver_->GetOptions().order;
}

ParGridFunction& ElasticityCase::Displacement()
{
   return hyper_ ? hyper_->Displacement() : solver_->Displacement();
}

bool ElasticityCase::Solve()
{
   const double t0 = MPI_Wtime();
   const bool ok = hyper_ ? hyper_->Solve() : solver_->Solve();
   solve_seconds_ = MPI_Wtime() - t0;
   solved_ = true;
   converged_ = ok;
   if (Mpi::Root() && hyper_)
   {
      const std::vector<double>& h = hyper_->ResidualHistory();
      const std::vector<double>& dx = hyper_->CorrectionHistory();
      std::cout << "hyperelasticity: " << hyper_->GlobalTrueDofs() << " dofs, "
                << hyper_->GetOptions().load_steps << " load step(s), Newton "
                << hyper_->NewtonIterations() << " iterations (max "
                << hyper_->MaxNewtonIterations() << " per step; residual "
                << (h.empty() ? 0.0 : h.front()) << " -> "
                << (h.empty() ? 0.0 : h.back()) << ", last correction "
                << (dx.empty() ? 0.0 : dx.back()) << " of |u|), CG "
                << hyper_->LinearIterations() << " iterations"
                << (ok ? "" : ", NOT converged") << ", " << solve_seconds_
                << " s" << std::endl;
   }
   else if (Mpi::Root())
   {
      std::cout << "elasticity: " << solver_->GlobalTrueDofs() << " dofs, CG "
                << solver_->Iterations() << " iterations (final residual "
                << solver_->FinalNorm() << ")" << (ok ? "" : ", NOT converged")
                << ", " << solve_seconds_ << " s" << std::endl;
   }
   if (params_.output.enabled)
   {
      if (Mpi::Root()) { std::filesystem::create_directories(params_.output.path); }
      ParaViewDataCollection pv(params_.output.name, &mesh_);
      pv.SetPrefixPath(params_.output.path);
      // High-order output, LOD >= k, or Q_k fields render as mush.
      pv.SetHighOrderOutput(true);
      pv.SetLevelsOfDetail(std::max(Order(), 1));
      pv.SetDataFormat(VTKFormat::BINARY);
      pv.RegisterField("displacement", &Displacement());
      pv.SetCycle(0);
      pv.SetTime(0.0);
      pv.Save();
   }
   return ok;
}

Vector ElasticityCase::ProbeDisplacement()
{
   const int dim = mesh_.Dimension();
   Vector u(dim);
   u = 0.0;
   if (params_.solid.has_probe)
   {
      Vector x(dim);
      for (int d = 0; d < dim; ++d) { x(d) = params_.solid.probe[d]; }
      if (hyper_) { hyper_->DisplacementAt(x, u); }
      else { solver_->DisplacementAt(x, u); }
   }
   return u;
}

bool ElasticityCase::Run()
{
   if (params_.solid.analysis == SolidAnalysis::Static) { return Solve(); }
   return hyper_ ? March(*hyper_) : March(*solver_);
}

template <class SolidSolver>
bool ElasticityCase::March(SolidSolver& solver)
{
   // Hyperelasticity logs its Newton iterations too.
   constexpr bool hyper = std::is_same<SolidSolver, Hyperelasticity>::value;
   INCNS_PROFILE("solid::march");
   const int dim = mesh_.Dimension();
   const double wall0 = MPI_Wtime();
   const bool root = Mpi::Root();
   const OutputParameters& out = params_.output;
   if (root && (out.history || out.enabled))
   {
      std::filesystem::create_directories(out.path);
   }
   solver.Initialize(0.0);

   std::ofstream history;
   const char* names[3] = {"x", "y", "z"};
   auto log = [&](double dt)
   {
      // Collective quantities on every rank, the row on rank 0.
      const double ke = solver.KineticEnergy();
      const double se = solver.StrainEnergy();
      const Vector u = ProbeDisplacement();
      if (!root || !history.is_open()) { return; }
      history << solver.StepCount() << "," << solver.Time() << "," << dt << ",";
      if constexpr(hyper)
      {
         history << solver.LinearIterations() << "," << solver.NewtonIterations();
      }
      else { history << solver.Iterations(); }
      history << "," << ke << "," << se << "," << ke + se;
      if (params_.solid.has_probe)
      {
         for (int d = 0; d < dim; ++d) { history << "," << u(d); }
      }
      history << "\n";
   };
   if (out.history && root)
   {
      history.open(out.path + "/" + out.name + "_history.csv");
      MFEM_VERIFY(history.good(), "elasticity_case: cannot write the history");
      history.precision(12);
      history << (hyper ? "step,t,dt,cg_iterations,newton_iterations,"
                  "kinetic_energy,strain_energy,total_energy" :
                  "step,t,dt,cg_iterations,kinetic_energy,strain_energy,"
                  "total_energy");
      if (params_.solid.has_probe)
      {
         for (int d = 0; d < dim; ++d) { history << ",u_" << names[d]; }
      }
      history << "\n";
   }
   if (out.history) { log(0.0); }

   std::unique_ptr<ParaViewDataCollection> pv;
   if (out.enabled)
   {
      pv = std::make_unique<ParaViewDataCollection>(out.name, &mesh_);
      pv->SetPrefixPath(out.path);
      // High-order output, LOD >= k, or Q_k fields render as mush.
      pv->SetHighOrderOutput(true);
      pv->SetLevelsOfDetail(std::max(solver.GetOptions().order, 1));
      pv->SetDataFormat(VTKFormat::BINARY);
      pv->RegisterField("displacement", &solver.Displacement());
      pv->RegisterField("velocity", &solver.Velocity());
      pv->SetCycle(0);
      pv->SetTime(0.0);
      pv->Save();
   }

   // Steps of time.dt; t_k = min(k dt, t_final), so the clock never drifts
   // and the last step lands on t_final.
   const double dt = params_.dt, tf = params_.t_final;
   const long long n = std::max(1LL, static_cast<long long>(
                                   std::ceil(tf / dt - 1e-9)));
   bool ok = true;
   double t_prev = 0.0;
   for (long long k = 1; k <= n && ok; ++k)
   {
      const double t = std::min(static_cast<double>(k) * dt, tf);
      ok = solver.Step(t - t_prev);
      if constexpr(hyper)
      {
         max_newton_step_ = std::max(max_newton_step_, solver.NewtonIterations());
      }
      if (out.history) { log(t - t_prev); }
      if (pv && (k % std::max(out.interval, 1) == 0 || k == n || !ok))
      {
         pv->SetCycle(static_cast<int>(k));
         pv->SetTime(t);
         pv->Save();
      }
      t_prev = t;
   }
   solve_seconds_ = MPI_Wtime() - wall0;
   solved_ = true;
   converged_ = ok;
   if (root)
   {
      if constexpr(hyper)
      {
         std::cout << "hyperelasticity: " << solver.GlobalTrueDofs() << " dofs, "
                   << solver.StepCount() << " steps to t = " << solver.Time()
                   << ", Newton " << solver.TotalNewtonIterations()
                   << " iterations (max " << max_newton_step_ << " per step), CG "
                   << solver.TotalLinearIterations() << " iterations"
                   << (ok ? "" : ", a step did NOT converge") << ", "
                   << solve_seconds_ << " s" << std::endl;
      }
      else
      {
         std::cout << "elasticity: " << solver.GlobalTrueDofs() << " dofs, "
                   << solver.StepCount() << " steps to t = " << solver.Time()
                   << ", CG " << solver.TotalIterations() << " iterations (max "
                   << solver.MaxIterations() << " per step)"
                   << (ok ? "" : ", a step did NOT converge") << ", "
                   << solve_seconds_ << " s" << std::endl;
      }
   }
   return ok;
}

void ElasticityCase::WriteSummary(const std::string& path,
                                  const std::string& status)
{
   const int dim = mesh_.Dimension();
   const ElasticityParameters& s = params_.solid;
   // Collective quantities first, on every rank.
   const long long elements = mesh_.GetGlobalNE();
   const bool dynamic = s.analysis == SolidAnalysis::Dynamic;
   double energy = 0.0, work = 0.0, umax = 0.0, kinetic = 0.0;
   Vector probe_u;
   bool probe_found = false;
   if (solved_ && hyper_)
   {
      energy = hyper_->StrainEnergy();
      if (dynamic) { kinetic = hyper_->KineticEnergy(); }
      work = hyper_->ExternalWork();
      umax = hyper_->MaxDisplacement();
      if (s.has_probe)
      {
         Vector x(dim);
         for (int d = 0; d < dim; ++d) { x(d) = s.probe[d]; }
         probe_found = hyper_->DisplacementAt(x, probe_u);
      }
   }
   else if (solved_)
   {
      energy = solver_->StrainEnergy();
      if (dynamic) { kinetic = solver_->KineticEnergy(); }
      work = solver_->ExternalWork();
      umax = solver_->MaxDisplacement();
      if (s.has_probe)
      {
         Vector x(dim);
         for (int d = 0; d < dim; ++d) { x(d) = s.probe[d]; }
         probe_found = solver_->DisplacementAt(x, probe_u);
      }
   }
   int np = 1;
   MPI_Comm_size(mesh_.GetComm(), &np);

   Json j;
   j["status"].Set(status);
   if (solved_ && !converged_)
   {
      j["divergence"].Set(hyper_ ? "Newton did not reach its tolerance" :
                          "the elasticity CG did not reach its tolerance");
   }
   Json& c = j["case"];
   c["equation"].Set(hyper_ ? "hyperelasticity" : "linear_elasticity");
   if (hyper_) { c["model"].Set("st_venant_kirchhoff"); }
   c["dim"].Set(dim);
   c["order"].Set(Order());
   c["youngs_modulus"].Set(s.youngs_modulus);
   c["poisson_ratio"].Set(s.poisson_ratio);
   if (dim == 2) { c["plane"].Set(s.plane == PlaneModel::Stress ? "stress" : "strain"); }
   c["lambda"].Set(hyper_ ? hyper_->GetOptions().lambda :
                   solver_->GetOptions().lambda);
   c["mu"].Set(hyper_ ? hyper_->GetOptions().mu : solver_->GetOptions().mu);
   c["mpi_ranks"].Set(np);
   c["mfem"].Set(GetGitStr());

   Json& m = j["mesh"];
   m["elements"].Set(elements);
   m["dofs"].Set(static_cast<long long>(hyper_ ? hyper_->GlobalTrueDofs() :
                                        solver_->GlobalTrueDofs()));

   Json& so = j["solver"];
   so["preconditioner"].Set(s.preconditioner == ElasticityPC::Jacobi ? "jacobi" :
                            "loramg");
   so["rtol"].Set(s.rtol);
   if (hyper_)
   {
      // CG over every Newton correction; the residual is Newton's.
      const std::vector<double>& h = hyper_->ResidualHistory();
      so["iterations"].Set(hyper_->LinearIterations());
      so["final_norm"].Set(h.empty() ? 0.0 : h.back());
      Json& nw = j["newton"];
      nw["rtol"].Set(s.newton_rtol);
      nw["atol"].Set(s.newton_atol);
      nw["dx_rtol"].Set(s.newton_dx_rtol);
      nw["load_steps"].Set(s.load_steps);
      nw["line_search"].Set(s.line_search);
      nw["iterations"].Set(hyper_->NewtonIterations());
      nw["max_iterations_per_load_step"].Set(hyper_->MaxNewtonIterations());
      nw["initial_residual"].Set(h.empty() ? 0.0 : h.front());
      nw["final_residual"].Set(h.empty() ? 0.0 : h.back());
      const std::vector<double>& dx = hyper_->CorrectionHistory();
      nw["final_correction"].Set(dx.empty() ? 0.0 : dx.back());
   }
   else
   {
      so["iterations"].Set(solver_->Iterations());
      so["final_norm"].Set(solver_->FinalNorm());
   }
   so["converged"].Set(solved_ && converged_);
   so["solve_seconds"].Set(solve_seconds_);

   if (dynamic)
   {
      Json& dy = j["dynamics"];
      const bool bdf = s.time_scheme == SolidTimeScheme::Bdf;
      dy["time_scheme"].Set(bdf ? "bdf" : "generalized_alpha");
      if (bdf) { dy["bdf_order"].Set(s.bdf_order); }
      else { dy["rho_inf"].Set(s.rho_inf); }
      dy["density"].Set(s.density);
      dy["dt"].Set(params_.dt);
      dy["t_final"].Set(params_.t_final);
      if (hyper_)
      {
         dy["time"].Set(hyper_->Time());
         dy["steps"].Set(hyper_->StepCount());
         dy["cg_iterations_total"].Set(hyper_->TotalLinearIterations());
         dy["newton_iterations_total"].Set(hyper_->TotalNewtonIterations());
         dy["newton_iterations_max"].Set(max_newton_step_);
      }
      else
      {
         dy["time"].Set(solver_->Time());
         dy["steps"].Set(solver_->StepCount());
         dy["cg_iterations_total"].Set(solver_->TotalIterations());
         dy["cg_iterations_max"].Set(solver_->MaxIterations());
      }
   }

   if (solved_)
   {
      Json& r = j["results"];
      r["strain_energy"].Set(energy);
      if (dynamic)
      {
         r["kinetic_energy"].Set(kinetic);
         r["total_energy"].Set(kinetic + energy);
      }
      r["external_work"].Set(work);
      r["max_displacement"].Set(umax);
      if (s.has_probe)
      {
         Json& p = r["probe"];
         p["found"].Set(probe_found);
         const char* names[3] = {"x", "y", "z"};
         for (int d = 0; d < dim; ++d)
         {
            p[std::string("point_") + names[d]].Set(s.probe[d]);
            p[std::string("u_") + names[d]].Set(probe_found ? probe_u(d) : 0.0);
         }
      }
   }

   if (Mpi::Root())
   {
      const std::filesystem::path out(path);
      if (out.has_parent_path()) { std::filesystem::create_directories(out.parent_path()); }
      std::ofstream f(path);
      MFEM_VERIFY(f.good(), "elasticity_case: cannot write the summary " << path);
      f << j.Dump();
   }
}

} // namespace incns
