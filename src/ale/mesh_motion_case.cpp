#include "ale/mesh_motion_case.hpp"

#include "ale/mesh_motion_factory.hpp"
#include "bc/boundary_names.hpp"
#include "bc/deck_boundary.hpp"
#include "config/expression.hpp"
#include "util/json.hpp"
#include "util/profiler.hpp"

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>

namespace incns
{

using namespace mfem;

namespace
{

// det F = det J / det J_0 at a point of an element of the moving mesh, J_0
// the reference mesh's Jacobian there.
class DetFCoefficient : public Coefficient
{
public:
   explicit DetFCoefficient(ParMesh& reference) : ref_(reference) {}

   real_t Eval(ElementTransformation& T, const IntegrationPoint& ip) override
   {
      T.SetIntPoint(&ip);
      ElementTransformation* T0 = ref_.GetElementTransformation(T.ElementNo);
      T0->SetIntPoint(&ip);
      return T.Weight() / T0->Weight();
   }

private:
   ParMesh& ref_;
};

} // namespace

double MinJacobianRatio(ParMesh& mesh, int order)
{
   IntegrationRules gll(0, Quadrature1D::GaussLobatto);
   const int rule_order = std::max(2 * order - 1, 1);
   double q = std::numeric_limits<double>::infinity();
   for (int e = 0; e < mesh.GetNE(); ++e)
   {
      const IntegrationRule& ir =
         gll.Get(mesh.GetElementBaseGeometry(e), rule_order);
      ElementTransformation* T = mesh.GetElementTransformation(e);
      double lo = std::numeric_limits<double>::infinity();
      double hi = -std::numeric_limits<double>::infinity();
      for (int i = 0; i < ir.GetNPoints(); ++i)
      {
         T->SetIntPoint(&ir.IntPoint(i));
         const double w = T->Weight();
         lo = std::min(lo, w);
         hi = std::max(hi, w);
      }
      // An inverted element has a negative minimum; a degenerate one 0.
      q = std::min(q, hi > 0.0 ? lo / hi : -1.0);
   }
   MPI_Allreduce(MPI_IN_PLACE, &q, 1, MPI_DOUBLE, MPI_MIN, mesh.GetComm());
   return q;
}

MeshMotionCase::MeshMotionCase(ParMesh& mesh, const Parameters& params)
   : mesh_(mesh), params_(params)
{
   MFEM_VERIFY(params_.equation == Equation::MeshMotion,
               "mesh_motion_case: the deck's equation is not mesh_motion");
   MFEM_VERIFY(params_.nondim.normalized,
               "mesh_motion_case: call Parameters::Normalize() first (LoadYAML "
               "does it)");
   if (mesh_.GetNodes())
   {
      MFEM_VERIFY(dynamic_cast<const H1_FECollection*>(
                     mesh_.GetNodes()->FESpace()->FEColl()),
                  "mesh_motion_case: the mesh's nodes are discontinuous (a "
                  "periodic mesh?): mesh motion needs a continuous nodal space");
   }
   const MeshMotionParameters& mm = params_.mesh_motion;
   const int dim = mesh_.Dimension();

   // Every real boundary in exactly one group, before anything is built.
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

   MeshMotion::Options o;
   o.order = 0; // the nodal order
   o.geometry_order = mm.Order(params_.order_u); // for a mesh without nodes
   motion_ = std::make_unique<MeshMotion>(mesh_, o, DeckMeshMotionFactory(mm));

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
      if (s.type == BcType::Fixed)
      {
         for (int a : attrs) { motion_->AddFixed(a); }
         continue;
      }
      MFEM_VERIFY(s.type == BcType::Displacement, "mesh_motion_case: boundary "
                  "types are fixed and displacement");
      Moving m;
      m.attributes = attrs;
      if (s.expression.empty())
      {
         Vector v(dim);
         for (int d = 0; d < dim; ++d) { v(d) = s.value[d]; }
         m.g = [v](const Vector&, double, Vector & d) { d = v; };
      }
      else
      {
         auto ex = std::make_shared<VectorExpression>(s.expression,
                   s.expression_parameters);
         m.g = [ex](const Vector & X, double t, Vector & d) { ex->Eval(X, t, d); };
      }
      coeffs_.push_back(std::make_unique<VectorFunctionCoefficient>(dim, m.g));
      for (int a : attrs) { motion_->AddDisplacement(a, *coeffs_.back()); }
      moving_.push_back(std::move(m));
   }

   const int k = mesh_.GetNodes()->FESpace()->FEColl()->GetOrder();
   ref_quality_ = MinJacobianRatio(motion_->ReferenceMesh(), k);
   quality_ = ref_quality_;
   detf_fec_ = std::make_unique<L2_FECollection>(k, dim, BasisType::GaussLobatto);
   detf_fes_ = std::make_unique<ParFiniteElementSpace>(&mesh_, detf_fec_.get());
   detf_ = std::make_unique<ParGridFunction>(detf_fes_.get());
   *detf_ = 1.0;
}

void MeshMotionCase::UpdateDetF()
{
   DetFCoefficient c(motion_->ReferenceMesh());
   detf_->HostReadWrite();
   detf_->ProjectCoefficient(c);
}

double MeshMotionCase::BoundaryError() const
{
   const ParGridFunction& disp = motion_->NodeDisplacement();
   const ParFiniteElementSpace& fes = *disp.ParFESpace();
   const Vector& x0 = motion_->ReferenceNodes();
   const double* d = disp.HostRead();
   const double* X0 = x0.HostRead();
   const int dim = fes.GetVDim();
   const int nattr = mesh_.bdr_attributes.Size() ? mesh_.bdr_attributes.Max() : 0;
   const double t = motion_->Time();
   double err = 0.0;
   Vector X(dim), g(dim);
   for (const Moving& m : moving_)
   {
      Array<int> marker(nattr);
      marker = 0;
      for (int a : m.attributes) { marker[a - 1] = 1; }
      Array<int> ess;
      fes.GetEssentialVDofs(marker, ess);
      for (int i = 0; i < fes.GetNDofs(); ++i)
      {
         if (!ess[fes.DofToVDof(i, 0)]) { continue; }
         for (int c = 0; c < dim; ++c) { X(c) = X0[fes.DofToVDof(i, c)]; }
         m.g(X, t, g);
         for (int c = 0; c < dim; ++c)
         {
            err = std::max(err, std::abs(d[fes.DofToVDof(i, c)] - g(c)));
         }
      }
   }
   MPI_Allreduce(MPI_IN_PLACE, &err, 1, MPI_DOUBLE, MPI_MAX, mesh_.GetComm());
   return err;
}

double MeshMotionCase::MaxDisplacement() const
{
   const ParGridFunction& disp = motion_->NodeDisplacement();
   const ParFiniteElementSpace& fes = *disp.ParFESpace();
   const double* d = disp.HostRead();
   const int dim = fes.GetVDim();
   double m = 0.0;
   for (int i = 0; i < fes.GetNDofs(); ++i)
   {
      double s = 0.0;
      for (int c = 0; c < dim; ++c)
      {
         const double v = d[fes.DofToVDof(i, c)];
         s += v * v;
      }
      m = std::max(m, std::sqrt(s));
   }
   MPI_Allreduce(MPI_IN_PLACE, &m, 1, MPI_DOUBLE, MPI_MAX, mesh_.GetComm());
   return m;
}

bool MeshMotionCase::Run()
{
   INCNS_PROFILE("mesh_motion::run");
   const double wall0 = MPI_Wtime();
   const bool root = Mpi::Root();
   const OutputParameters& out = params_.output;
   const MeshMotionParameters& mm = params_.mesh_motion;
   if (root && (out.history || out.enabled))
   {
      std::filesystem::create_directories(out.path);
   }

   std::ofstream history;
   if (out.history && root)
   {
      history.open(out.path + "/" + out.name + "_history.csv");
      MFEM_VERIFY(history.good(), "mesh_motion_case: cannot write the history");
      history.precision(12);
      history << "move,t,cg_iterations,det_f_min,det_f_max,jacobian_ratio_min,"
              "max_displacement,boundary_error\n";
   }

   std::unique_ptr<ParaViewDataCollection> pv;
   if (out.enabled)
   {
      const int k = mesh_.GetNodes()->FESpace()->FEColl()->GetOrder();
      pv = std::make_unique<ParaViewDataCollection>(out.name, &mesh_);
      pv->SetPrefixPath(out.path);
      // High-order output, LOD >= k, or the curved mesh renders as facets.
      pv->SetHighOrderOutput(true);
      pv->SetLevelsOfDetail(std::max(k, 1));
      pv->SetDataFormat(VTKFormat::BINARY);
      // ParaView only reads the field.
      pv->RegisterField("displacement", const_cast<ParGridFunction*>(
                           &motion_->NodeDisplacement()));
      pv->RegisterField("det_F", detf_.get());
      pv->SetCycle(0);
      pv->SetTime(0.0);
      pv->Save(); // the reference mesh
   }

   bool ok = true;
   const int n = mm.steps;
   for (int s = 1; s <= n; ++s)
   {
      const double t = params_.t_final * static_cast<double>(s) / n;
      const bool conv = motion_->Move(t);
      solved_ = true;
      converged_ = converged_ && conv;
      iterations_ = motion_->Model().Iterations();
      iterations_total_ += iterations_;
      motion_->JacobianRange(detf_min_, detf_max_);
      detf_worst_ = s == 1 ? detf_min_ : std::min(detf_worst_, detf_min_);
      const int k = mesh_.GetNodes()->FESpace()->FEColl()->GetOrder();
      quality_ = MinJacobianRatio(mesh_, k);
      const double umax = MaxDisplacement();
      const double berr = BoundaryError();
      if (root)
      {
         std::cout << "mesh_motion: move " << s << "/" << n << " t = " << t
                   << ", CG " << iterations_ << " iterations"
                   << (conv ? "" : " (NOT converged)") << ", det F in ["
                   << detf_min_ << ", " << detf_max_ << "], min Jacobian ratio "
                   << quality_ << ", max |d| " << umax << std::endl;
      }
      if (history.is_open())
      {
         history << s << "," << t << "," << iterations_ << "," << detf_min_
                 << "," << detf_max_ << "," << quality_ << "," << umax << ","
                 << berr << "\n";
      }
      const bool valid = detf_min_ > 0.0;
      ok = ok && conv && valid;
      if (pv && (s % std::max(out.interval, 1) == 0 || s == n || !valid))
      {
         UpdateDetF();
         pv->SetCycle(s);
         pv->SetTime(t);
         pv->Save();
      }
      if (!valid)
      {
         if (root)
         {
            std::cout << "mesh_motion: the mesh is inverted at t = " << t
                      << " (min det F " << detf_min_ << "): stopping"
                      << std::endl;
         }
         break;
      }
   }
   solve_seconds_ = MPI_Wtime() - wall0;
   return ok;
}

void MeshMotionCase::WriteSummary(const std::string& path,
                                  const std::string& status)
{
   const int dim = mesh_.Dimension();
   const MeshMotionParameters& mm = params_.mesh_motion;
   // Collective quantities first, on every rank.
   const long long elements = mesh_.GetGlobalNE();
   const long long dofs = motion_->Model().Space().GlobalTrueVSize();
   double umax = 0.0, berr = 0.0;
   if (solved_)
   {
      umax = MaxDisplacement();
      berr = BoundaryError();
   }
   int np = 1;
   MPI_Comm_size(mesh_.GetComm(), &np);

   Json j;
   j["status"].Set(status);
   if (solved_ && !converged_)
   {
      j["divergence"].Set("the mesh-motion CG did not reach its tolerance");
   }
   else if (solved_ && detf_worst_ <= 0.0)
   {
      j["divergence"].Set("the deformed mesh is inverted (min det F <= 0)");
   }
   Json& c = j["case"];
   c["equation"].Set("mesh_motion");
   c["dim"].Set(dim);
   c["order"].Set(motion_->Order());
   c["model"].Set(motion_->Model().Name());
   c["poisson_ratio"].Set(mm.poisson_ratio);
   c["stiffening"].Set(mm.stiffening);
   c["mpi_ranks"].Set(np);
   c["mfem"].Set(GetGitStr());

   Json& m = j["mesh"];
   m["elements"].Set(elements);
   m["dofs"].Set(dofs);
   m["reference_jacobian_ratio_min"].Set(ref_quality_);

   Json& so = j["solver"];
   so["preconditioner"].Set(mm.preconditioner == ElasticityPC::Jacobi ?
                            "jacobi" : "loramg");
   so["rtol"].Set(mm.rtol);
   so["iterations"].Set(iterations_);
   so["iterations_total"].Set(iterations_total_);
   so["converged"].Set(solved_ && converged_);
   so["solve_seconds"].Set(solve_seconds_);

   Json& mo = j["motion"];
   mo["steps"].Set(mm.steps);
   mo["t_final"].Set(params_.t_final);
   mo["moves"].Set(motion_->MoveCount());
   mo["time"].Set(motion_->Time());

   if (solved_)
   {
      Json& r = j["results"];
      r["det_f_min"].Set(detf_min_);
      r["det_f_max"].Set(detf_max_);
      r["det_f_min_over_moves"].Set(detf_worst_);
      r["jacobian_ratio_min"].Set(quality_);
      r["max_displacement"].Set(umax);
      r["boundary_error"].Set(berr);
   }

   if (Mpi::Root())
   {
      const std::filesystem::path out(path);
      if (out.has_parent_path()) { std::filesystem::create_directories(out.parent_path()); }
      std::ofstream f(path);
      MFEM_VERIFY(f.good(), "mesh_motion_case: cannot write the summary " << path);
      f << j.Dump();
   }
}

} // namespace incns
