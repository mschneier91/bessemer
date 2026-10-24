#include "fsi/fsi_case.hpp"

#include "bc/boundary_names.hpp"
#include "config/initial_conditions.hpp"
#include "mesh/case_mesh.hpp"
#include "post/checkpoint.hpp"
#include "precond/hypre_device.hpp"
#include "solid/elasticity_case.hpp"
#include "solid/hyperelasticity.hpp"
#include "solid/linear_elasticity.hpp"
#include "util/profiler.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <limits>

#include <yaml-cpp/yaml.h>

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

/// A value per boundary element (zero on any other): the Robin coefficient
/// face by face (boundary integrators evaluate it on boundary elements).
class BoundaryElementCoefficient : public Coefficient
{
public:
   /// @param values One value per local boundary element.
   explicit BoundaryElementCoefficient(std::vector<double> values)
      : values_(std::move(values)) {}

   /// @return The value of the boundary element @p T is of.
   double Eval(ElementTransformation& T, const IntegrationPoint&) override
   {
      if (T.ElementType != ElementTransformation::BDR_ELEMENT) { return 0.0; }
      return values_[T.ElementNo];
   }

private:
   std::vector<double> values_; ///< Per local boundary element.
};

/// The distance from @p x along the unit vector @p n to the first of the
/// faces (segments in 2D, triangles in 3D, @p dim points each, packed) it
/// crosses beyond @p tmin; infinity if none.
double RayDistance(int dim, const double* x, const double* n,
                   const std::vector<double>& faces, double tmin)
{
   double best = std::numeric_limits<double>::infinity();
   const int per = dim * dim;
   for (std::size_t f = 0; f + per <= faces.size(); f += per)
   {
      const double* a = &faces[f];
      double t = -1.0;
      if (dim == 2)
      {
         // x + t n = a + s (b - a), s in [0, 1].
         const double ex = a[2] - a[0], ey = a[3] - a[1];
         const double det = -n[0] * ey + n[1] * ex;
         if (std::abs(det) < 1e-300) { continue; }
         const double rx = a[0] - x[0], ry = a[1] - x[1];
         t = (-rx * ey + ry * ex) / det;
         const double s = (n[0] * ry - n[1] * rx) / det;
         if (s < -1e-12 || s > 1.0 + 1e-12) { continue; }
      }
      else
      {
         // Moller-Trumbore.
         const double* b = a + 3;
         const double* c = a + 6;
         double e1[3], e2[3], p[3], q[3], s[3];
         for (int i = 0; i < 3; ++i)
         {
            e1[i] = b[i] - a[i];
            e2[i] = c[i] - a[i];
            s[i] = x[i] - a[i];
         }
         p[0] = n[1] * e2[2] - n[2] * e2[1];
         p[1] = n[2] * e2[0] - n[0] * e2[2];
         p[2] = n[0] * e2[1] - n[1] * e2[0];
         const double det = e1[0] * p[0] + e1[1] * p[1] + e1[2] * p[2];
         if (std::abs(det) < 1e-300) { continue; }
         const double u = (s[0] * p[0] + s[1] * p[1] + s[2] * p[2]) / det;
         if (u < -1e-12 || u > 1.0 + 1e-12) { continue; }
         q[0] = s[1] * e1[2] - s[2] * e1[1];
         q[1] = s[2] * e1[0] - s[0] * e1[2];
         q[2] = s[0] * e1[1] - s[1] * e1[0];
         const double v = (n[0] * q[0] + n[1] * q[1] + n[2] * q[2]) / det;
         if (v < -1e-12 || u + v > 1.0 + 1e-12) { continue; }
         t = (e2[0] * q[0] + e2[1] * q[1] + e2[2] * q[2]) / det;
      }
      if (t > tmin) { best = std::min(best, t); }
   }
   return best;
}

/**
 * @brief Solve M_G t = b on the interface's dofs, M_G the vector boundary
 *        mass of the faces @p marker selects: the field whose moments are
 *        @p b. Collective.
 * @param fes    Vector H1 space.
 * @param marker Boundary attribute marker of the interface.
 * @param b      Moments (true dofs; read where @p on is set, else zero).
 * @param on     Per true dof: whether it is an interface unknown.
 * @param t      The field (true dofs; zero off the interface).
 */
void InterfaceMassSolve(ParFiniteElementSpace& fes, Array<int>& marker,
                        const Vector& b, const std::vector<char>& on, Vector& t)
{
   const int n = fes.GetTrueVSize();
   ParBilinearForm m(&fes);
   m.AddBoundaryIntegrator(new VectorMassIntegrator(), marker);
   m.Assemble();
   m.Finalize();
   std::unique_ptr<HypreParMatrix> M(m.ParallelAssemble());
   Array<int> other;
   for (int i = 0; i < n; ++i) { if (!on[i]) { other.Append(i); } }
   std::unique_ptr<HypreParMatrix> Me(M->EliminateRowsCols(other));
   Vector bb(n);
   const double* src = b.HostRead();
   double* dst = bb.HostWrite();
   for (int i = 0; i < n; ++i) { dst[i] = on[i] ? src[i] : 0.0; }
   t.SetSize(n);
   t = 0.0;
   // Jacobi on M's diagonal, 1 where a row has none: off the interface the
   // boundary mass's rows are empty (EliminateRowsCols cannot put a 1 there),
   // and hypre's Jacobi read past them -- now and then a NaN on a GPU
   // (compute-sanitizer initcheck, job 49677951).
   Vector dg;
   HypreDiagonal(*M, dg);
   double* dgh = dg.HostReadWrite();
   for (int i = 0; i < n; ++i) { if (dgh[i] == 0.0) { dgh[i] = 1.0; } }
   const Array<int> none;
   OperatorJacobiSmoother jacobi(dg, none);
   CGSolver cg(fes.GetComm());
   cg.SetOperator(*M);
   cg.SetPreconditioner(jacobi);
   cg.SetRelTol(1e-12);
   cg.SetAbsTol(0.0);
   cg.SetMaxIter(1000);
   cg.SetPrintLevel(0);
   cg.Mult(bb, t);
   MFEM_VERIFY(cg.GetConverged(), "fsi_case: the interface mass solve did not "
               "converge");
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
   const bool hyper = f.solid == FsiSolidModel::Hyperelasticity;
   solid_params_.equation = hyper ? Equation::Hyperelasticity :
                            Equation::LinearElasticity;
   solid_mesh_ = MakeCaseMesh(solid_params_, fluid_mesh_.GetComm());
   if (hyper)
   {
      solid_ = std::make_unique<FsiSolidAdapter<Hyperelasticity>>(
                  std::make_unique<Hyperelasticity>(
                     *solid_mesh_, HyperelasticOptions(solid_params_)),
                  "hyperelasticity");
   }
   else
   {
      solid_ = std::make_unique<FsiSolidAdapter<LinearElasticity>>(
                  std::make_unique<LinearElasticity>(
                     *solid_mesh_, ElasticityOptions(solid_params_)),
                  "linear_elasticity");
   }
   MFEM_VERIFY(!f.solid_fixed.empty(), "fsi_case: fsi.solid_fixed names no "
               "boundary (the solid's clamp)");
   solid_fixed_attrs_ = NamedAttributes(solid_params_, *solid_mesh_,
                                        f.solid_fixed);
   for (int a : solid_fixed_attrs_) { solid_->AddFixed(a); }
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
   const bool nd = f.coupling == FsiCoupling::NeumannDirichlet;
   if (nd)
   {
      // Neumann-Dirichlet: the solid's interface follows the iterate.
      solid_dirichlet_ = std::make_unique<ParGridFunction>(&solid_->Space());
      *solid_dirichlet_ = 0.0;
      solid_dirichlet_coeff_ = std::make_unique<VectorGridFunctionCoefficient>(
                                  solid_dirichlet_.get());
      for (int a : solid_attrs_)
      {
         solid_->AddDisplacement(a, *solid_dirichlet_coeff_);
      }
      solid_->TrackReaction();
   }
   else
   {
      for (int a : solid_attrs_) { solid_->AddTraction(a, *traction_coeff_); }
   }
   if (f.coupling == FsiCoupling::RobinRobin)
   {
      MFEM_VERIFY(f.alpha_solid > 0.0, "fsi_case: robin_robin needs "
                  "fsi.alpha_solid > 0");
      for (int a : solid_attrs_) { solid_->AddSupport(a, SupportStiffness()); }
   }
   dynamic_ = params_.solid.analysis == SolidAnalysis::Dynamic;
   if (dynamic_) { solid_->Initialize(); }

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
   if (nd)
   {
      to_fluid_v_ = std::make_unique<InterfaceTransfer>(*solid_mesh_,
                    *ref_vfes_, fluid_attrs_);
   }
   interface_force_.SetSize(dim);
   interface_force_ = 0.0;

   accel_ = MakeInterfaceAccelerator(f, fluid_mesh_.GetComm());
   if ((f.coupling == FsiCoupling::RobinNeumann ||
        f.coupling == FsiCoupling::RobinRobin) && f.alpha_fluid <= 0.0 &&
       f.structure_thickness <= 0.0)
   {
      BuildRobinCoefficient();
   }

   // The coupling's state travels in the flow's checkpoint and comes back
   // with a restart (the flow and its mesh motion restored theirs above).
   fluid_->SetCheckpointHook([this](const std::string & dir)
   {
      WriteCheckpointState(dir);
   });
   if (!params_.restart_from.empty()) { ReadCheckpointState(params_.restart_from); }

   fluid_->SetSummaryHook([this](Json & j,
                                 std::vector<std::pair<std::string, double>>& got)
   {
      Json& s = j["fsi"];
      const FsiCoupling cp = params_.fsi.coupling;
      const bool robin = cp == FsiCoupling::RobinNeumann ||
                         cp == FsiCoupling::RobinRobin;
      s["coupling"].Set(cp == FsiCoupling::RobinNeumann ? "robin_neumann" :
                        cp == FsiCoupling::RobinRobin ? "robin_robin" :
                        cp == FsiCoupling::NeumannDirichlet ?
                        "neumann_dirichlet" : "dirichlet_neumann");
      if (cp == FsiCoupling::RobinRobin)
      {
         s["alpha_solid"].Set(params_.fsi.alpha_solid);
      }
      if (robin && robin_alpha_)
      {
         s["alpha_fluid_min"].Set(robin_alpha_min_ * params_.fsi.fluid_density);
         s["alpha_fluid_max"].Set(robin_alpha_max_ * params_.fsi.fluid_density);
      }
      else if (robin)
      {
         s["alpha_fluid"].Set(params_.FsiFluidRobin() *
                              params_.fsi.fluid_density);
      }
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
      so["model"].Set(solid_->Name());
      so["analysis"].Set(dynamic_ ? "dynamic" : "static");
      so["order"].Set(solid_->Order());
      so["elements"].Set(static_cast<long long>(solid_mesh_->GetGlobalNE()));
      so["dofs"].Set(static_cast<long long>(solid_->GlobalTrueDofs()));
      so["cg_iterations"].Set(restored_cg_ + solid_->TotalIterations());
      const long long newton = restored_newton_ + solid_->TotalNewtonIterations();
      if (newton > 0) { so["newton_iterations"].Set(newton); }
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

void FsiCase::BuildRobinCoefficient()
{
   // alpha_f = rho_s h / dt face by face, h the solid's thickness behind the
   // face: the distance through the solid along the face's normal, capped
   // at the depth the solid's response reaches in a step (c dt, c the
   // P-wave speed). Badia, Nobile and Vergara's rho_s H / dt for a thin
   // wall; the tip of a long flag gets its length's (docs/fsi.md).
   const int dim = fluid_mesh_.Dimension();
   MPI_Comm comm = fluid_mesh_.GetComm();
   // The solid's boundary, linearized (segments; triangles in 3D), all of
   // it on every rank.
   std::vector<double> faces;
   for (int b = 0; b < solid_mesh_->GetNBE(); ++b)
   {
      Array<int> v;
      solid_mesh_->GetBdrElementVertices(b, v);
      auto put = [&](int i)
      {
         const double* c = solid_mesh_->GetVertex(v[i]);
         for (int d = 0; d < dim; ++d) { faces.push_back(c[d]); }
      };
      if (dim == 2) { put(0); put(1); }
      else
      {
         MFEM_VERIFY(v.Size() == 4, "fsi_case: hexahedral solid meshes only");
         put(0); put(1); put(2);
         put(0); put(2); put(3);
      }
   }
   int nloc = static_cast<int>(faces.size()), np = 1;
   MPI_Comm_size(comm, &np);
   std::vector<int> counts(np), displs(np, 0);
   MPI_Allgather(&nloc, 1, MPI_INT, counts.data(), 1, MPI_INT, comm);
   for (int r = 1; r < np; ++r) { displs[r] = displs[r - 1] + counts[r - 1]; }
   std::vector<double> all(displs[np - 1] + counts[np - 1]);
   MPI_Allgatherv(faces.data(), nloc, MPI_DOUBLE, all.data(), counts.data(),
                  displs.data(), MPI_DOUBLE, comm);

   const ElasticityParameters& sp = params_.solid;
   const double nu = sp.poisson_ratio;
   const double wave = std::sqrt(sp.youngs_modulus * (1.0 - nu) /
                                 ((1.0 + nu) * (1.0 - 2.0 * nu) * sp.density));
   const double dt = params_.dt;
   const double cap = wave * dt;
   std::vector<double> alpha(fluid_mesh_.GetNBE(), 0.0);
   double amax = 0.0, amin = std::numeric_limits<double>::infinity();
   for (int b = 0; b < fluid_mesh_.GetNBE(); ++b)
   {
      if (!fluid_marker_[fluid_mesh_.GetBdrAttribute(b) - 1]) { continue; }
      ElementTransformation& T = *fluid_mesh_.GetBdrElementTransformation(b);
      const IntegrationPoint& c =
         Geometries.GetCenter(fluid_mesh_.GetBdrElementGeometry(b));
      T.SetIntPoint(&c);
      Vector x(dim), n(dim);
      T.Transform(c, x);
      CalcOrtho(T.Jacobian(), n);
      n /= n.Norml2();
      // Out of the fluid (into the solid): away from the adjacent element.
      int el = -1, info = 0;
      fluid_mesh_.GetBdrElementAdjacentElement(b, el, info);
      Vector xe(dim);
      ElementTransformation& Te = *fluid_mesh_.GetElementTransformation(el);
      Te.Transform(Geometries.GetCenter(fluid_mesh_.GetElementGeometry(el)), xe);
      xe -= x;
      if (n * xe > 0.0) { n.Neg(); }
      const double size = std::pow(T.Weight(), 1.0 / (dim - 1));
      const double h = RayDistance(dim, x.GetData(), n.GetData(), all,
                                   1e-6 * size);
      alpha[b] = sp.density * std::min(h, cap) / dt / params_.fsi.fluid_density;
      amax = std::max(amax, alpha[b]);
      amin = std::min(amin, alpha[b]);
   }
   MPI_Allreduce(MPI_IN_PLACE, &amax, 1, MPI_DOUBLE, MPI_MAX, comm);
   MPI_Allreduce(MPI_IN_PLACE, &amin, 1, MPI_DOUBLE, MPI_MIN, comm);
   MFEM_VERIFY(amax > 0.0, "fsi_case: no interface face for the Robin "
               "coefficient");
   robin_alpha_min_ = amin;
   robin_alpha_max_ = amax;
   robin_alpha_ = std::make_unique<BoundaryElementCoefficient>(std::move(alpha));
   fluid_->SetRobinCoefficient(*robin_alpha_, amax);
}

void FsiCase::UpdateTraction()
{
   UpdateLoad();
   LoadSolid(last_load_);
}

void FsiCase::UpdateLoad()
{
   ParFiniteElementSpace& vfes = fluid_->Spaces().Velocity();
   Vector r;
   fluid_->Integrator().MomentumResidual(r);
   const int n = vfes.GetTrueVSize();
   MFEM_VERIFY(r.Size() == n, "fsi_case: residual size");
   Array<int> iface;
   vfes.GetEssentialTrueDofs(fluid_marker_, iface);
   last_load_.SetSize(n);
   last_load_ = 0.0;
   const double* rr = r.HostRead();
   double* ll = last_load_.HostReadWrite();
   for (int i : iface) { ll[i] = rr[i]; }
}

void FsiCase::LoadSolid(const Vector& load)
{
   Vector t;
   TractionFromLoad(load, t); // and InterfaceForce()
   // The large-deformation solid takes the traction per reference area; the
   // small-strain one the current one (the same to first order; bitwise the
   // F3/F4 results, which every coupling shares).
   if (params_.fsi.solid == FsiSolidModel::Hyperelasticity)
   {
      NominalTractionFromLoad(load, t);
   }
   LoadSolidTraction(t);
}

void FsiCase::NominalTractionFromLoad(const Vector& load, Vector& t)
{
   INCNS_PROFILE("fsi::nominal_traction");
   // The moments of the Cauchy traction on the moved interface are those of
   // the nominal one on the reference interface (sigma n dGamma = P N
   // dGamma_0): the reference interface's mass gives P N.
   const int n = ref_vfes_->GetTrueVSize();
   MFEM_VERIFY(load.Size() == n, "fsi_case: load size");
   Array<int> iface;
   ref_vfes_->GetEssentialTrueDofs(fluid_marker_, iface);
   std::vector<char> on(n, 0);
   for (int i : iface) { on[i] = 1; }
   InterfaceMassSolve(*ref_vfes_, fluid_marker_, load, on, t);
}

void FsiCase::TractionFromLoad(const Vector& load, Vector& t)
{
   INCNS_PROFILE("fsi::traction");
   ParFiniteElementSpace& vfes = fluid_->Spaces().Velocity();
   const double rho = params_.fsi.fluid_density;
   const int n = vfes.GetTrueVSize();
   MFEM_VERIFY(load.Size() == n, "fsi_case: load size");

   // The interface's mass on the current mesh: M t = r at the interface
   // dofs gives the traction field whose moments are the residual's.
   Array<int> iface;
   vfes.GetEssentialTrueDofs(fluid_marker_, iface);
   std::vector<char> on(n, 0);
   for (int i : iface) { on[i] = 1; }
   const double* ll = load.HostRead();
   const int vdim = vfes.GetVDim();
   const int per = n / vdim;
   MFEM_VERIFY(vfes.GetOrdering() == Ordering::byNODES,
               "fsi_case: the velocity space is ordered by nodes");
   interface_force_ = 0.0;
   for (int i = 0; i < n; ++i)
   {
      if (on[i]) { interface_force_(i / per) -= rho * ll[i]; }
   }
   MPI_Allreduce(MPI_IN_PLACE, interface_force_.GetData(), vdim, MPI_DOUBLE,
                 MPI_SUM, vfes.GetComm());
   InterfaceMassSolve(vfes, fluid_marker_, load, on, t);
}

void FsiCase::ReactionTraction(Vector& tau)
{
   INCNS_PROFILE("fsi::reaction");
   // The solid's reaction at its interface dofs (not the clamp's) as a
   // traction field on the solid, moved to the fluid's interface in the
   // reference configurations: tau = -t_s / rho_f (the fluid's kinematic
   // traction; the fluid pulls on the solid with t_s).
   ParFiniteElementSpace& sfes = solid_->Space();
   const int n = sfes.GetTrueVSize();
   Array<int> smarker = Marker(*solid_mesh_, solid_attrs_);
   Array<int> fixed = Marker(*solid_mesh_, solid_fixed_attrs_);
   Array<int> iface, clamp;
   sfes.GetEssentialTrueDofs(smarker, iface);
   sfes.GetEssentialTrueDofs(fixed, clamp);
   std::vector<char> on(n, 0);
   for (int i : iface) { on[i] = 1; }
   for (int i : clamp) { on[i] = 0; }
   const Vector& R = solid_->Reaction();
   const double* rr = R.HostRead();
   const int vdim = sfes.GetVDim();
   const int per = n / vdim;
   MFEM_VERIFY(sfes.GetOrdering() == Ordering::byNODES,
               "fsi_case: the solid's space is ordered by nodes");
   interface_force_ = 0.0;
   for (int i = 0; i < n; ++i)
   {
      if (on[i]) { interface_force_(i / per) += rr[i]; }
   }
   MPI_Allreduce(MPI_IN_PLACE, interface_force_.GetData(), vdim, MPI_DOUBLE,
                 MPI_SUM, sfes.GetComm());
   Vector ts;
   InterfaceMassSolve(sfes, smarker, R, on, ts);
   solid_traction_->SetFromTrueDofs(ts);
   *ref_traction_ = 0.0;
   to_fluid_v_->Transfer(*solid_traction_, *ref_traction_);
   ParFiniteElementSpace& vfes = fluid_->Spaces().Velocity();
   ParGridFunction tg(&vfes);
   ref_traction_->HostRead();
   tg = *ref_traction_; // the same layout on the current mesh
   tg *= -1.0 / params_.fsi.fluid_density;
   tg.GetTrueDofs(tau);
}

void FsiCase::LoadSolidTraction(const Vector& t)
{
   ParFiniteElementSpace& vfes = fluid_->Spaces().Velocity();
   ParGridFunction tg(&vfes);
   tg.SetFromTrueDofs(t);
   tg *= -params_.fsi.fluid_density; // the fluid's pull on the solid
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
   if (params_.fsi.coupling == FsiCoupling::RobinNeumann ||
       params_.fsi.coupling == FsiCoupling::RobinRobin)
   {
      return StepRobinNeumann();
   }
   if (params_.fsi.coupling == FsiCoupling::NeumannDirichlet)
   {
      return StepNeumannDirichlet();
   }
   const FsiParameters& f = params_.fsi;
   const double wall0 = MPI_Wtime();
   MPI_Comm comm = fluid_mesh_.GetComm();
   ParGridFunction& dg = *fluid_->InterfaceDisplacement();
   ParFiniteElementSpace& mfes = *dg.ParFESpace();
   const int n = mfes.GetTrueVSize();
   Vector d(n), dt_vec(n), r(n);
   Predict(d);
   ParGridFunction solid_d(&mfes);
   if (dynamic_) { solid_->SaveState(); }
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
         solid_->RestoreState();
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
         // The step's record before the flow's commit, whose checkpoint
         // (WriteCheckpointState()) holds it.
         committed_.insert(committed_.begin(), d);
         if (committed_.size() > 3) { committed_.pop_back(); }
         rec.time = fluid_->Time();
         history_.push_back(rec);
         total_iterations_ += k;
         max_iterations_ = std::max(max_iterations_, k);
         LogStep(rec);
         fluid_->CommitTrialStep();
         wall_ += MPI_Wtime() - wall0;
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

void FsiCase::PredictLoad(Vector& load) const
{
   const int q = std::min<int>(params_.fsi.predictor,
                               static_cast<int>(committed_load_.size()) - 1);
   if (committed_load_.empty())
   {
      load = 0.0;
      return;
   }
   load = committed_load_[0];
   if (q == 1)
   {
      load *= 2.0;
      load.Add(-1.0, committed_load_[1]);
   }
   else if (q >= 2)
   {
      load *= 3.0;
      load.Add(-3.0, committed_load_[1]);
      load.Add(1.0, committed_load_[2]);
   }
}

double FsiCase::SupportStiffness() const
{
   // alpha_s times the mesh velocity's BDF weight of the newest level at
   // the march's order (the linearization of alpha_s w in d).
   const double c0 = (params_.BdfOrder() >= 3 ? 11.0 / 6.0 : 1.5) / params_.dt;
   return params_.fsi.alpha_solid * c0;
}

bool FsiCase::StepRobinNeumann()
{
   // The interface variable is the fluid's traction field tau (kinematic,
   // sigma_f n on the interface): the solid takes -rho_f tau, the mesh
   // follows its interface, the fluid's Robin condition takes tau and the
   // solid's velocity, and its traction is the next tau. The map tau ->
   // tau~ depends on tau alone, as the accelerators need (iterating on d
   // would leave tau as a hidden state). tau is a field, not the moments
   // r: the moments of one traction change as the interface moves.
   //
   // Robin-Robin: the variable is the solid's Robin data g (kinematic, as
   // tau): the solid takes -rho_f g with its support k = alpha_s c0, i.e.
   // sigma_s n_s = -rho_f g - k d; the fluid's Robin data is then
   // -sigma_s n_s / rho_f = g + kappa d (kappa = k / rho_f), and its
   // traction tau~, velocity u and mesh velocity w give the next
   // g~ = tau~ - a_s (u - w) - kappa d (a_s = alpha_s / rho_f). a_s = 0 is
   // Robin-Neumann.
   const FsiParameters& f = params_.fsi;
   const bool rr = f.coupling == FsiCoupling::RobinRobin;
   const double rho = f.fluid_density;
   const double a_s = rr ? f.alpha_solid / rho : 0.0;
   const double kappa = rr ? SupportStiffness() / rho : 0.0;
   const double wall0 = MPI_Wtime();
   MPI_Comm comm = fluid_mesh_.GetComm();
   ParGridFunction& dg = *fluid_->InterfaceDisplacement();
   ParFiniteElementSpace& mfes = *dg.ParFESpace();
   ParFiniteElementSpace& vfes = fluid_->Spaces().Velocity();
   const int n = vfes.GetTrueVSize();
   if (!robin_tau_)
   {
      robin_tau_ = std::make_unique<ParGridFunction>(&vfes);
      *robin_tau_ = 0.0;
      robin_tau_coeff_ =
         std::make_unique<VectorGridFunctionCoefficient>(robin_tau_.get());
   }
   Vector tau(n), tau_tilde(n), r(n), d(mfes.GetTrueVSize());
   Vector data(n), dv(n), uw(n);
   Array<int> iface;
   vfes.GetEssentialTrueDofs(fluid_marker_, iface);
   ParGridFunction dv_ref(ref_vfes_.get());
   PredictLoad(tau);
   ParGridFunction solid_d(&mfes);
   if (dynamic_) { solid_->SaveState(); }
   const double dt = params_.dt;
   fluid_->Integrator().SetBoundaryTraction(robin_tau_coeff_.get());
   accel_->NewStep();
   StepRecord rec;
   for (int k = 1; k <= f.max_iterations; ++k)
   {
      LoadSolidTraction(tau);
      bool solid_ok = true;
      if (dynamic_)
      {
         solid_->RestoreState();
         solid_ok = solid_->Step(dt);
      }
      else
      {
         solid_ok = solid_->Solve();
      }
      if (!solid_ok)
      {
         failure_ = "the solid's solve did not converge";
         return false;
      }
      SolidInterfaceDisplacement(solid_d);
      solid_d.GetTrueDofs(d);
      dg.SetFromTrueDofs(d);
      data = tau;
      if (rr)
      {
         // d on the velocity space's interface nodes.
         dv_ref.ProjectGridFunction(dg);
         Vector all;
         dv_ref.GetTrueDofs(all);
         dv = 0.0;
         const double* a = all.HostRead();
         double* o = dv.HostReadWrite();
         for (int i : iface) { o[i] = a[i]; }
         data.Add(kappa, dv);
      }
      robin_tau_->SetFromTrueDofs(data);
      fluid_->TrialStep();
      UpdateLoad();
      TractionFromLoad(last_load_, tau_tilde); // on the moved mesh
      if (rr)
      {
         ParGridFunction w(&vfes);
         w = fluid_->Velocity();
         w -= fluid_->MeshVelocity();
         Vector all;
         w.GetTrueDofs(all);
         uw = 0.0;
         const double* a = all.HostRead();
         double* o = uw.HostReadWrite();
         for (int i : iface) { o[i] = a[i]; }
         tau_tilde.Add(-a_s, uw);
         tau_tilde.Add(-kappa, dv);
      }
      subtract(tau_tilde, tau, r);
      const double rn = std::sqrt(InnerProduct(comm, r, r));
      const double tn = std::sqrt(InnerProduct(comm, tau_tilde, tau_tilde));
      rec.iterations = k;
      rec.residual = rn / std::max(tn, 1e-300);
      if (!std::isfinite(rn))
      {
         failure_ = "the interface residual is not finite";
         fluid_->UndoTrialStep();
         return false;
      }
      if (rn <= f.rtol * tn + f.atol)
      {
         // The step's record before the flow's commit, whose checkpoint
         // (WriteCheckpointState()) holds it.
         committed_.insert(committed_.begin(), d);
         if (committed_.size() > 3) { committed_.pop_back(); }
         committed_load_.insert(committed_load_.begin(), tau_tilde);
         if (committed_load_.size() > 3) { committed_load_.pop_back(); }
         rec.time = fluid_->Time();
         history_.push_back(rec);
         total_iterations_ += k;
         max_iterations_ = std::max(max_iterations_, k);
         LogStep(rec);
         fluid_->CommitTrialStep();
         wall_ += MPI_Wtime() - wall0;
         if (fluid_->Diverged())
         {
            failure_ = "the flow diverged";
            return false;
         }
         return true;
      }
      fluid_->UndoTrialStep();
      Vector next(n);
      rec.omega = accel_->Next(tau, tau_tilde, next);
      if (f.print_level > 0 && Mpi::Root())
      {
         mfem::out << "[fsi] t = " << fluid_->Time() << " subiteration " << k
                   << ": |r| / |tau~| = " << rec.residual << ", |tau~| = " << tn
                   << ", omega = " << rec.omega << std::endl;
      }
      tau = next;
   }
   failure_ = "the subiterations did not converge in fsi.max_iterations at t = " +
              std::to_string(fluid_->Time() + dt);
   if (Mpi::Root()) { mfem::out << "[incns] FSI: " << failure_ << std::endl; }
   return false;
}

bool FsiCase::StepNeumannDirichlet()
{
   // The interface variable is the displacement d (motion true dofs): the
   // solid with its interface at d gives its reaction, the fluid on the
   // mesh at d takes it as a traction, and the displacement its interface
   // velocity integrates to, d~ = d + (u - w) / c0 (w = c0 d + the older
   // levels' part, the mesh velocity's BDF), is the next d. At convergence
   // u = w on the interface and the tractions balance.
   const FsiParameters& f = params_.fsi;
   const double wall0 = MPI_Wtime();
   MPI_Comm comm = fluid_mesh_.GetComm();
   ParGridFunction& dg = *fluid_->InterfaceDisplacement();
   ParFiniteElementSpace& mfes = *dg.ParFESpace();
   ParFiniteElementSpace& vfes = fluid_->Spaces().Velocity();
   const int n = mfes.GetTrueVSize();
   if (!robin_tau_)
   {
      robin_tau_ = std::make_unique<ParGridFunction>(&vfes);
      *robin_tau_ = 0.0;
      robin_tau_coeff_ =
         std::make_unique<VectorGridFunctionCoefficient>(robin_tau_.get());
   }
   Vector d(n), dt_vec(n), r(n), tau(vfes.GetTrueVSize());
   Predict(d);
   ParGridFunction dtilde(&mfes), slip(&mfes);
   ParGridFunction uw_ref(ref_vfes_.get());
   if (dynamic_) { solid_->SaveState(); }
   const double dt = params_.dt;
   fluid_->Integrator().SetBoundaryTraction(robin_tau_coeff_.get());
   accel_->NewStep();
   StepRecord rec;
   for (int k = 1; k <= f.max_iterations; ++k)
   {
      dg.SetFromTrueDofs(d);
      to_solid_->Transfer(dg, *solid_dirichlet_);
      bool solid_ok = true;
      if (dynamic_)
      {
         solid_->RestoreState();
         solid_ok = solid_->Step(dt);
      }
      else
      {
         solid_ok = solid_->Solve();
      }
      if (!solid_ok)
      {
         failure_ = "the solid's solve did not converge";
         return false;
      }
      ReactionTraction(tau);
      robin_tau_->SetFromTrueDofs(tau);
      fluid_->TrialStep();
      // u - w at the interface's motion nodes (the reference configuration
      // has the current mesh's layout).
      const double c0 = fluid_->MeshVelocityLead();
      MFEM_VERIFY(c0 > 0.0, "fsi_case: no mesh velocity weight");
      {
         ParGridFunction uw(&vfes);
         uw = fluid_->Velocity();
         uw -= fluid_->MeshVelocity();
         uw.HostRead();
         uw_ref = uw;
         slip.ProjectGridFunction(uw_ref);
      }
      dtilde = dg;
      {
         const double* s = slip.HostRead();
         double* t = dtilde.HostReadWrite();
         for (int i : to_fluid_->Vdofs()) { t[i] += s[i] / c0; }
      }
      dtilde.GetTrueDofs(dt_vec);
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
         // The step's record before the flow's commit, whose checkpoint
         // (WriteCheckpointState()) holds it.
         committed_.insert(committed_.begin(), d);
         if (committed_.size() > 3) { committed_.pop_back(); }
         rec.time = fluid_->Time();
         history_.push_back(rec);
         total_iterations_ += k;
         max_iterations_ = std::max(max_iterations_, k);
         LogStep(rec);
         fluid_->CommitTrialStep();
         wall_ += MPI_Wtime() - wall0;
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

void FsiCase::WriteCheckpointState(const std::string& dir)
{
   INCNS_PROFILE("fsi::checkpoint");
   const MPI_Comm comm = fluid_mesh_.GetComm();
   const int rank = fluid_mesh_.GetMyRank();
   auto file = [&](const std::string & base)
   {
      return Checkpoint::RankFile(dir, base, rank);
   };
   solid_->WriteState(dir, "fsi_solid");
   for (std::size_t i = 0; i < committed_.size(); ++i)
   {
      Checkpoint::WriteVector(file("fsi_committed" + std::to_string(i)),
                              committed_[i]);
   }
   for (std::size_t i = 0; i < committed_load_.size(); ++i)
   {
      Checkpoint::WriteVector(file("fsi_committed_load" + std::to_string(i)),
                              committed_load_[i]);
   }
   const InterfaceAccelerator::State a = accel_->ExportState();
   for (std::size_t i = 0; i < a.vectors.size(); ++i)
   {
      Checkpoint::WriteVector(file("fsi_accelerator" + std::to_string(i)),
                              a.vectors[i]);
   }
   if (rank == 0)
   {
      std::ofstream m(dir + "/fsi.yaml");
      MFEM_VERIFY(m.good(), "fsi_case: cannot write " << dir << "/fsi.yaml");
      m << std::setprecision(17)
        << "# incns FSI coupling state (same-np restart contract)\n"
        << "steps: " << history_.size() << "\n"
        << "subiterations: " << total_iterations_ << "\n"
        << "max_subiterations: " << max_iterations_ << "\n"
        << "wall_seconds: " << wall_ << "\n"
        << "committed: " << committed_.size() << "\n"
        << "committed_load: " << committed_load_.size() << "\n"
        << "accelerator_counts: [";
      for (std::size_t i = 0; i < a.counts.size(); ++i)
      {
         m << (i ? ", " : "") << a.counts[i];
      }
      m << "]\naccelerator_vectors: " << a.vectors.size() << "\n"
        << "solid_cg_iterations: " << restored_cg_ + solid_->TotalIterations()
        << "\nsolid_newton_iterations: "
        << restored_newton_ + solid_->TotalNewtonIterations() << "\n";
      MFEM_VERIFY(m.good(), "fsi_case: writing fsi.yaml failed");
      // The step records: time, subiterations, residual, omega per step.
      std::ofstream h(dir + "/fsi_history.bin", std::ios::binary);
      const std::int64_t n = static_cast<std::int64_t>(history_.size());
      h.write(reinterpret_cast<const char*>(&n), sizeof(n));
      for (const StepRecord& r : history_)
      {
         const double row[4] = {r.time, static_cast<double>(r.iterations),
                                r.residual, r.omega
                               };
         h.write(reinterpret_cast<const char*>(row), sizeof(row));
      }
      MFEM_VERIFY(h.good(), "fsi_case: writing fsi_history.bin failed");
   }
   MPI_Barrier(comm); // the checkpoint is complete on return, on every rank
}

void FsiCase::ReadCheckpointState(const std::string& dir)
{
   MFEM_VERIFY(std::filesystem::exists(dir + "/fsi.yaml"), "fsi_case: " << dir
               << " holds no FSI coupling state (a flow's checkpoint?)");
   const YAML::Node m = YAML::LoadFile(dir + "/fsi.yaml");
   const std::size_t steps = m["steps"].as<std::size_t>();
   MFEM_VERIFY(steps == static_cast<std::size_t>(fluid_->Integrator().StepCount()),
               "fsi_case: the checkpoint's coupling is at step " << steps
               << ", its flow at step " << fluid_->Integrator().StepCount());
   const int rank = fluid_mesh_.GetMyRank();
   auto file = [&](const std::string & base)
   {
      return Checkpoint::RankFile(dir, base, rank);
   };
   solid_->ReadState(dir, "fsi_solid");
   const int nd = fluid_->InterfaceDisplacement()->ParFESpace()->GetTrueVSize();
   const int nv = fluid_->Spaces().Velocity().GetTrueVSize();
   committed_.resize(m["committed"].as<std::size_t>());
   for (std::size_t i = 0; i < committed_.size(); ++i)
   {
      Checkpoint::ReadVector(file("fsi_committed" + std::to_string(i)), nd,
                             committed_[i]);
   }
   committed_load_.resize(m["committed_load"].as<std::size_t>());
   for (std::size_t i = 0; i < committed_load_.size(); ++i)
   {
      Checkpoint::ReadVector(file("fsi_committed_load" + std::to_string(i)), nv,
                             committed_load_[i]);
   }
   // The accelerator's vectors are the iterate's: the traction under the
   // Robin couplings, the displacement otherwise.
   const FsiCoupling cp = params_.fsi.coupling;
   const int ni = (cp == FsiCoupling::RobinNeumann ||
                   cp == FsiCoupling::RobinRobin) ? nv : nd;
   InterfaceAccelerator::State a;
   for (const auto& c : m["accelerator_counts"]) { a.counts.push_back(c.as<int>()); }
   a.vectors.resize(m["accelerator_vectors"].as<std::size_t>());
   for (std::size_t i = 0; i < a.vectors.size(); ++i)
   {
      Checkpoint::ReadVector(file("fsi_accelerator" + std::to_string(i)), ni,
                             a.vectors[i]);
   }
   accel_->ImportState(a);
   total_iterations_ = m["subiterations"].as<long long>();
   max_iterations_ = m["max_subiterations"].as<int>();
   wall_ = m["wall_seconds"].as<double>();
   restored_cg_ = m["solid_cg_iterations"].as<long long>();
   restored_newton_ = m["solid_newton_iterations"].as<long long>();
   std::ifstream h(dir + "/fsi_history.bin", std::ios::binary);
   std::int64_t n = 0;
   h.read(reinterpret_cast<char*>(&n), sizeof(n));
   MFEM_VERIFY(h.good() && static_cast<std::size_t>(n) == steps,
               "fsi_case: " << dir << "/fsi_history.bin does not hold the "
               "checkpoint's steps");
   history_.resize(static_cast<std::size_t>(n));
   for (StepRecord& r : history_)
   {
      double row[4] = {0.0, 0.0, 0.0, 0.0};
      h.read(reinterpret_cast<char*>(row), sizeof(row));
      r.time = row[0];
      r.iterations = static_cast<int>(row[1]);
      r.residual = row[2];
      r.omega = row[3];
   }
   MFEM_VERIFY(h.good(), "fsi_case: short read from " << dir
               << "/fsi_history.bin");
   // The coupling log continues (as the flow's history, appended).
   log_started_ = std::filesystem::exists(params_.output.path + "/" +
                                          params_.output.name + "_fsi.csv");
}

void FsiCase::WriteSummary(const std::string& path, const std::string& status)
{
   fluid_->WriteSummary(path, status);
}

} // namespace incns
