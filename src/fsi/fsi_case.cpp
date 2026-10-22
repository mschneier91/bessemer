#include "fsi/fsi_case.hpp"

#include "bc/boundary_names.hpp"
#include "config/initial_conditions.hpp"
#include "mesh/case_mesh.hpp"
#include "solid/elasticity_case.hpp"
#include "util/profiler.hpp"

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>

namespace incns
{

using namespace mfem;

namespace
{

/// The attributes of the deck's boundary groups of type interface.
std::vector<int> InterfaceAttributes(const Parameters& p, ParMesh& mesh,
                                     std::vector<std::string>* names)
{
   std::vector<int> attrs;
   for (const BcSpec& s : p.boundary_conditions)
   {
      if (s.type != BcType::Interface) { continue; }
      if (s.select_all)
      {
         for (int a : AllBoundaryAttributes(p, mesh)) { attrs.push_back(a); }
         continue;
      }
      for (int a : s.attributes) { attrs.push_back(a); }
      for (const std::string& n : s.faces)
      {
         attrs.push_back(NamedBoundaryAttribute(p, mesh, n));
         if (names) { names->push_back(n); }
      }
   }
   std::sort(attrs.begin(), attrs.end());
   attrs.erase(std::unique(attrs.begin(), attrs.end()), attrs.end());
   return attrs;
}

/// The attributes of @p names on @p mesh.
std::vector<int> NamedAttributes(const Parameters& p, ParMesh& mesh,
                                 const std::vector<std::string>& names)
{
   std::vector<int> attrs;
   for (const std::string& n : names)
   {
      attrs.push_back(NamedBoundaryAttribute(p, mesh, n));
   }
   return attrs;
}

/// A boundary marker of @p mesh with @p attrs set.
Array<int> Marker(ParMesh& mesh, const std::vector<int>& attrs)
{
   const int n = mesh.bdr_attributes.Size() ? mesh.bdr_attributes.Max() : 0;
   Array<int> m(n);
   m = 0;
   for (int a : attrs)
   {
      MFEM_VERIFY(a >= 1 && a <= n, "fsi_case: boundary attribute " << a
                  << " is not on the mesh");
      m[a - 1] = 1;
   }
   return m;
}

} // namespace

FsiCase::FsiCase(ParMesh& fluid_mesh, const Parameters& params)
   : fluid_mesh_(fluid_mesh), params_(params), solid_params_(params)
{
   const FsiParameters& f = params_.fsi;
   MFEM_VERIFY(f.enabled, "fsi_case: the deck's fsi section is not enabled");
   // A failed fluid solve inside the coupling must not change dt.
   params_.on_solver_failure = SolverFailure::Abort;
   MFEM_VERIFY(params_.geometry == MeshGeometry::CylinderChannel &&
               params_.cylinder_channel.flag_length > 0.0,
               "fsi_case: the solid's mesh is the solid part of the geometry; "
               "today only mesh.geometry cylinder_channel with a flag "
               "(flag_length > 0) has one");
   MFEM_VERIFY(!params_.amr.enabled, "fsi_case: no AMR on a moving mesh");
   const int dim = fluid_mesh_.Dimension();

   // The flow: its interface groups follow Case::InterfaceDisplacement().
   fluid_ = std::make_unique<Case>(fluid_mesh_, params_);
   u0_ = MakeInitialVelocity(params_);
   fluid_->SetInitialVelocity(*u0_);
   MFEM_VERIFY(fluid_->InterfaceDisplacement(), "fsi_case: the flow has no "
               "interface group");
   std::vector<std::string> fluid_names;
   fluid_attrs_ = InterfaceAttributes(params_, fluid_mesh_, &fluid_names);
   fluid_marker_ = Marker(fluid_mesh_, fluid_attrs_);

   // The solid, on the geometry's solid part.
   solid_params_.cylinder_channel.solid = true;
   solid_params_.equation = Equation::LinearElasticity;
   solid_mesh_ = MakeCaseMesh(solid_params_, fluid_mesh_.GetComm());
   solid_ = std::make_unique<LinearElasticity>(*solid_mesh_,
            ElasticityOptions(solid_params_));
   MFEM_VERIFY(!f.solid_fixed.empty(), "fsi_case: fsi.solid_fixed names no "
               "boundary (the solid's clamp)");
   for (int a : NamedAttributes(solid_params_, *solid_mesh_, f.solid_fixed))
   {
      solid_->AddFixed(a);
   }
   const std::vector<std::string>& snames =
      f.solid_interface.empty() ? fluid_names : f.solid_interface;
   MFEM_VERIFY(!snames.empty(), "fsi_case: name the solid's interface "
               "(fsi.solid_interface) when the fluid's interface groups give "
               "attributes, not names");
   solid_attrs_ = NamedAttributes(solid_params_, *solid_mesh_, snames);
   solid_traction_ = std::make_unique<ParGridFunction>(&solid_->Space());
   *solid_traction_ = 0.0;
   traction_coeff_ = std::make_unique<VectorGridFunctionCoefficient>(
                        solid_traction_.get());
   for (int a : solid_attrs_) { solid_->AddTraction(a, *traction_coeff_); }
   dynamic_ = params_.solid.analysis == SolidAnalysis::Dynamic;
   if (dynamic_) { solid_->Initialize(0.0); }

   // The interface maps, both in the reference configurations.
   MeshMotion& motion = fluid_->Motion();
   ParFiniteElementSpace& vfes = fluid_->Spaces().Velocity();
   ref_vfes_ = std::make_unique<ParFiniteElementSpace>(
                  &motion.ReferenceMesh(), vfes.FEColl(), vfes.GetVDim(),
                  vfes.GetOrdering());
   MFEM_VERIFY(ref_vfes_->GetVSize() == vfes.GetVSize(),
               "fsi_case: the reference mesh's velocity space differs from the "
               "flow's");
   ref_traction_ = std::make_unique<ParGridFunction>(ref_vfes_.get());
   *ref_traction_ = 0.0;
   to_solid_ = std::make_unique<InterfaceTransfer>(motion.ReferenceMesh(),
               solid_->Space(), solid_attrs_);
   to_fluid_ = std::make_unique<InterfaceTransfer>(*solid_mesh_,
               motion.Model().Space(), fluid_attrs_);
   interface_force_.SetSize(dim);
   interface_force_ = 0.0;

   accel_ = MakeInterfaceAccelerator(f, fluid_mesh_.GetComm());

   fluid_->SetSummaryHook([this](Json & j,
                                 std::vector<std::pair<std::string, double>>& got)
   {
      Json& s = j["fsi"];
      s["coupling"].Set("dirichlet_neumann");
      const FsiAcceleration a = params_.fsi.acceleration;
      s["acceleration"].Set(a == FsiAcceleration::Aitken ? "aitken" :
                            a == FsiAcceleration::IqnIls ? "iqn_ils" : "constant");
      s["steps"].Set(static_cast<long long>(history_.size()));
      s["subiterations"].Set(total_iterations_);
      s["max_subiterations"].Set(max_iterations_);
      s["mean_subiterations"].Set(history_.empty() ? 0.0 :
                                  static_cast<double>(total_iterations_) /
                                  static_cast<double>(history_.size()));
      s["wall_seconds"].Set(wall_);
      if (!failure_.empty()) { s["failure"].Set(failure_); }
      Json& so = s["solid"];
      so["analysis"].Set(dynamic_ ? "dynamic" : "static");
      so["order"].Set(solid_->GetOptions().order);
      so["elements"].Set(static_cast<long long>(solid_mesh_->GetGlobalNE()));
      so["dofs"].Set(static_cast<long long>(solid_->GlobalTrueDofs()));
      so["cg_iterations"].Set(solid_->TotalIterations());
      so["max_displacement"].Set(solid_->MaxDisplacement());
      for (int c = 0; c < interface_force_.Size(); ++c)
      {
         s["interface_force"][std::string(1, static_cast<char>('x' + c))]
         .Set(interface_force_(c));
      }
      if (params_.solid.has_probe)
      {
         Vector x(solid_mesh_->Dimension()), u(solid_mesh_->Dimension());
         for (int c = 0; c < x.Size(); ++c) { x(c) = params_.solid.probe[c]; }
         solid_->DisplacementAt(x, u); // collective
         Json& pr = s["probe"];
         pr["x"].Set(x(0));
         pr["y"].Set(x(1));
         pr["displacement_x"].Set(u(0));
         pr["displacement_y"].Set(u(1));
         got.push_back({"displacement_x", u(0)});
         got.push_back({"displacement_y", u(1)});
      }
   });
}

FsiCase::~FsiCase() = default;

void FsiCase::UpdateTraction()
{
   INCNS_PROFILE("fsi::traction");
   ParFiniteElementSpace& vfes = fluid_->Spaces().Velocity();
   const double rho = params_.fsi.fluid_density;
   Vector r;
   fluid_->Integrator().MomentumResidual(r);
   const int n = vfes.GetTrueVSize();
   MFEM_VERIFY(r.Size() == n, "fsi_case: residual size");

   // The interface's mass on the moved mesh: M t = -rho r at the interface
   // dofs gives the traction field whose moments are the residual's.
   ParBilinearForm m(&vfes);
   m.AddBoundaryIntegrator(new VectorMassIntegrator(), fluid_marker_);
   m.Assemble();
   m.Finalize();
   std::unique_ptr<HypreParMatrix> M(m.ParallelAssemble());
   Array<int> iface;
   vfes.GetEssentialTrueDofs(fluid_marker_, iface);
   std::vector<char> on(n, 0);
   for (int i : iface) { on[i] = 1; }
   Array<int> other;
   for (int i = 0; i < n; ++i) { if (!on[i]) { other.Append(i); } }
   std::unique_ptr<HypreParMatrix> Me(M->EliminateRowsCols(other));

   Vector b(n);
   const double* rr = r.HostRead();
   double* bb = b.HostWrite();
   const int vdim = vfes.GetVDim();
   const int per = n / vdim;
   MFEM_VERIFY(vfes.GetOrdering() == Ordering::byNODES,
               "fsi_case: the velocity space is ordered by nodes");
   interface_force_ = 0.0;
   for (int i = 0; i < n; ++i)
   {
      bb[i] = on[i] ? -rho * rr[i] : 0.0;
      interface_force_(i / per) += bb[i];
   }
   MPI_Allreduce(MPI_IN_PLACE, interface_force_.GetData(), vdim, MPI_DOUBLE,
                 MPI_SUM, vfes.GetComm());

   Vector t(n);
   t = 0.0;
   HypreSmoother jacobi(*M, HypreSmoother::Jacobi);
   CGSolver cg(vfes.GetComm());
   cg.SetOperator(*M);
   cg.SetPreconditioner(jacobi);
   cg.SetRelTol(1e-12);
   cg.SetAbsTol(0.0);
   cg.SetMaxIter(1000);
   cg.SetPrintLevel(0);
   cg.Mult(b, t);
   MFEM_VERIFY(cg.GetConverged(), "fsi_case: the interface mass solve did not "
               "converge");

   ParGridFunction tg(&vfes);
   tg.SetFromTrueDofs(t);
   tg.HostRead();
   *ref_traction_ = tg; // the same layout on the reference mesh
   to_solid_->Transfer(*ref_traction_, *solid_traction_);
}

void FsiCase::SolidInterfaceDisplacement(ParGridFunction& d)
{
   d = 0.0;
   to_fluid_->Transfer(solid_->Displacement(), d);
}

void FsiCase::Predict(Vector& d) const
{
   const int q = std::min<int>(params_.fsi.predictor,
                               static_cast<int>(committed_.size()) - 1);
   if (committed_.empty())
   {
      d = 0.0;
      return;
   }
   d = committed_[0];
   if (q == 1)
   {
      d *= 2.0;
      d.Add(-1.0, committed_[1]);
   }
   else if (q >= 2)
   {
      d *= 3.0;
      d.Add(-3.0, committed_[1]);
      d.Add(1.0, committed_[2]);
   }
}

bool FsiCase::Step()
{
   INCNS_PROFILE("fsi::step");
   const FsiParameters& f = params_.fsi;
   const double wall0 = MPI_Wtime();
   MPI_Comm comm = fluid_mesh_.GetComm();
   ParGridFunction& dg = *fluid_->InterfaceDisplacement();
   ParFiniteElementSpace& mfes = *dg.ParFESpace();
   const int n = mfes.GetTrueVSize();
   Vector d(n), dt_vec(n), r(n);
   Predict(d);
   ParGridFunction solid_d(&mfes);
   LinearElasticity::State s0;
   if (dynamic_) { s0 = solid_->ExportState(); }
   const double dt = params_.dt;
   accel_->NewStep();
   StepRecord rec;
   for (int k = 1; k <= f.max_iterations; ++k)
   {
      dg.SetFromTrueDofs(d);
      fluid_->TrialStep();
      UpdateTraction();
      bool solid_ok = true;
      if (dynamic_)
      {
         solid_->ImportState(s0);
         solid_ok = solid_->Step(dt);
      }
      else
      {
         solid_ok = solid_->Solve();
      }
      if (!solid_ok)
      {
         failure_ = "the solid's solve did not converge";
         fluid_->UndoTrialStep();
         return false;
      }
      SolidInterfaceDisplacement(solid_d);
      solid_d.GetTrueDofs(dt_vec);
      subtract(dt_vec, d, r);
      const double rn = std::sqrt(InnerProduct(comm, r, r));
      const double dn = std::sqrt(InnerProduct(comm, dt_vec, dt_vec));
      rec.iterations = k;
      rec.residual = rn / std::max(dn, 1e-300);
      if (!std::isfinite(rn))
      {
         failure_ = "the interface residual is not finite";
         fluid_->UndoTrialStep();
         return false;
      }
      if (rn <= f.rtol * dn + f.atol)
      {
         fluid_->CommitTrialStep();
         committed_.insert(committed_.begin(), d);
         if (committed_.size() > 3) { committed_.pop_back(); }
         rec.time = fluid_->Time();
         history_.push_back(rec);
         total_iterations_ += k;
         max_iterations_ = std::max(max_iterations_, k);
         wall_ += MPI_Wtime() - wall0;
         LogStep(rec);
         if (fluid_->Diverged())
         {
            failure_ = "the flow diverged";
            return false;
         }
         return true;
      }
      fluid_->UndoTrialStep();
      Vector next(n);
      rec.omega = accel_->Next(d, dt_vec, next);
      if (f.print_level > 0 && Mpi::Root())
      {
         mfem::out << "[fsi] t = " << fluid_->Time() << " subiteration " << k
                   << ": |r| / |d~| = " << rec.residual << ", |d~| = " << dn
                   << ", omega = " << rec.omega << std::endl;
      }
      d = next;
   }
   failure_ = "the subiterations did not converge in fsi.max_iterations at t = " +
              std::to_string(fluid_->Time() + dt);
   if (Mpi::Root()) { mfem::out << "[incns] FSI: " << failure_ << std::endl; }
   return false;
}

bool FsiCase::Run()
{
   INCNS_PROFILE("fsi::run");
   bool ok = true;
   while (!fluid_->Done())
   {
      if (!Step())
      {
         ok = false;
         break;
      }
   }
   fluid_->FinishRun();
   return ok;
}

void FsiCase::LogStep(const StepRecord& r)
{
   Vector u;
   const bool probe = params_.solid.has_probe;
   if (probe)
   {
      Vector x(solid_mesh_->Dimension());
      u.SetSize(x.Size());
      for (int c = 0; c < x.Size(); ++c) { x(c) = params_.solid.probe[c]; }
      solid_->DisplacementAt(x, u); // collective
   }
   if (!Mpi::Root()) { return; }
   const std::string path = params_.output.path + "/" + params_.output.name +
                            "_fsi.csv";
   std::ofstream out;
   if (!log_started_)
   {
      std::filesystem::create_directories(params_.output.path);
      out.open(path);
      out << "step,t,subiterations,residual,omega,force_x,force_y";
      if (probe) { out << ",displacement_x,displacement_y"; }
      out << "\n";
      log_started_ = true;
   }
   else { out.open(path, std::ios::app); }
   out << std::setprecision(12) << history_.size() << "," << r.time << ","
       << r.iterations << "," << r.residual << "," << r.omega << ","
       << interface_force_(0) << "," << interface_force_(1);
   if (probe) { out << "," << u(0) << "," << u(1); }
   out << "\n";
}

void FsiCase::WriteSummary(const std::string& path, const std::string& status)
{
   fluid_->WriteSummary(path, status);
}

} // namespace incns
