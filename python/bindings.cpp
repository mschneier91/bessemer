// pybind11 bindings for the incns unsteady Stokes job driver. Thin layer over
// Case/Parameters: Python configures a case and supplies analytic
// IC/BC/forcing, C++ does all numerics. No MFEM type crosses the boundary; the
// case builds its own ParMesh from Parameters. See CLAUDE.md "Python interface".

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "bc/boundary_conditions.hpp"
#include "config/nondimensionalization.hpp"
#include "config/parameters.hpp"
#include "mesh/periodic_box.hpp"
#include "solver/case.hpp"
#include "mfem.hpp"

#include <cstdint>
#include <map>
#include <memory>
#include <string>
#include <vector>

namespace py = pybind11;
using namespace mfem;
using namespace incns;

namespace
{

// The Tier-2 fast-path ABI: a raw C function pointer (from numba @cfunc) called
// once per quad point with no Python/GIL involvement. Signature
// void(const double* x, double t, double* out); the dimension is implicit (the
// coefficient's vdim), so it is not passed.
using FieldFn = void (*)(const double*, double, double*);

// Evaluate physical coordinates of an integration point into xout[0..2].
void PhysCoords(ElementTransformation& T, const IntegrationPoint& ip,
                double xout[3])
{
   xout[0] = xout[1] = xout[2] = 0.0;
   Vector tip(xout, T.GetSpaceDim());
   T.Transform(ip, tip);
}

// Tier 2: wrap a numba @cfunc address as a vector coefficient. No GIL.
class CFuncVectorCoefficient : public VectorCoefficient
{
public:
   CFuncVectorCoefficient(std::uintptr_t address, int dim)
      : VectorCoefficient(dim), fn_(reinterpret_cast<FieldFn>(address)) {}

   using VectorCoefficient::Eval;
   void Eval(Vector& V, ElementTransformation& T,
             const IntegrationPoint& ip) override
   {
      double x[3];
      PhysCoords(T, ip, x);
      V.SetSize(vdim);
      fn_(x, GetTime(), V.GetData());
   }

private:
   FieldFn fn_;
};

// Tier 1: wrap a plain Python callable f(x, t) -> sequence. Acquires the GIL per
// evaluation (slow; fine for ICs, documented cost for hot paths).
class CallbackVectorCoefficient : public VectorCoefficient
{
public:
   CallbackVectorCoefficient(py::object fn, int dim)
      : VectorCoefficient(dim), fn_(std::move(fn)) {}

   using VectorCoefficient::Eval;
   void Eval(Vector& V, ElementTransformation& T,
             const IntegrationPoint& ip) override
   {
      double x[3];
      PhysCoords(T, ip, x);
      V.SetSize(vdim);
      py::gil_scoped_acquire gil;
      py::list xl;
      for (int i = 0; i < vdim; ++i) { xl.append(x[i]); }
      py::object r = fn_(xl, GetTime());
      for (int i = 0; i < vdim; ++i) { V(i) = r[py::int_(i)].cast<double>(); }
   }

private:
   py::object fn_;
};

std::unique_ptr<VectorCoefficient> MakeCoefficient(const py::object& f, int dim)
{
   // A compiled Tier-2 field carries `.address` (uintptr) and `.dim`.
   if (py::hasattr(f, "address"))
   {
      const auto address = f.attr("address").cast<std::uintptr_t>();
      const int fdim = f.attr("dim").cast<int>();
      MFEM_VERIFY(fdim == dim,
                  "incns: compiled field dim does not match the case dimension");
      return std::make_unique<CFuncVectorCoefficient>(address, dim);
   }
   return std::make_unique<CallbackVectorCoefficient>(f, dim);
}

// Owns the mesh + case + boundary conditions built from a Parameters block. The
// Python-facing Case: the ParMesh is an internal detail.
class PyCase
{
public:
   explicit PyCase(Parameters params) : params_(std::move(params))
   {
      params_.Normalize(); // idempotent; the case requires it before the mesh
      serial_ = MakeBoxMesh(params_.mesh);
      pmesh_ = std::make_unique<ParMesh>(MPI_COMM_WORLD, serial_);
      case_ = std::make_unique<Case>(*pmesh_, params_);
      bc_ = std::make_unique<BoundaryConditions>(case_->Spaces().Velocity());
   }

   void SetInitialVelocity(const py::object& f)
   {
      ic_ = MakeCoefficient(f, params_.mesh.dim);
      case_->SetInitialVelocity(*ic_);
   }

   void SetForcing(const py::object& f)
   {
      forcing_ = MakeCoefficient(f, params_.mesh.dim);
      case_->SetForcing(*forcing_);
   }

   void VelocityDirichlet(const py::object& attrs, const py::object& f)
   {
      auto coeff = MakeCoefficient(f, params_.mesh.dim);
      for (int a : AsAttrs(attrs)) { bc_->AddVelocityDirichlet(a, *coeff); }
      coeffs_.push_back(std::move(coeff));
   }

   void Outflow(const py::object& attrs)
   {
      for (int a : AsAttrs(attrs)) { bc_->AddOutflow(a); }
   }

   // No-slip walls (u = 0) on a flexible selection: attribute ints, box face
   // names, the string "all", or a list mixing them.
   void NoSlip(const py::object& selection)
   {
      for (int a : ResolveSelection(selection)) { bc_->AddNoSlip(a); }
   }

   // Bind a velocity field to a deck-declared Dirichlet group (by group name).
   // The field is applied to that group's attributes when the case is wired.
   void SetDirichletField(const std::string& group, const py::object& f)
   {
      auto coeff = MakeCoefficient(f, params_.mesh.dim);
      dirichlet_fields_[group] = coeff.get();
      coeffs_.push_back(std::move(coeff));
   }

   void Run() { Wire(); py::gil_scoped_release rel; case_->Run(); }
   void Step() { Wire(); py::gil_scoped_release rel; case_->Step(); }

   double Time() const { return case_->Time(); }
   double TimeDimensional() const { return case_->TimeDimensional(); }
   bool Done() const { return case_->Done(); }
   int StepCount() { return case_->Integrator().StepCount(); }
   int Iterations() { return case_->Integrator().LastIterations(); }

   // Scalar diagnostic: global L2 error of the velocity vs an analytic field,
   // evaluated at the current time with an elevated Gauss-Legendre rule.
   double VelocityL2Error(const py::object& f)
   {
      auto coeff = MakeCoefficient(f, params_.mesh.dim);
      coeff->SetTime(case_->Time());
      const int dim = params_.mesh.dim;
      const Geometry::Type geom = (dim == 3) ? Geometry::CUBE : Geometry::SQUARE;
      IntegrationRules gl(0, Quadrature1D::GaussLegendre);
      const IntegrationRule* irs[Geometry::NumGeom] = {nullptr};
      irs[geom] = &gl.Get(geom, 2 * params_.order_u + 4);
      return case_->Velocity().ComputeL2Error(*coeff, irs);
   }

   // Box-face name -> boundary attribute (geometric; robust to MFEM numbering).
   int Face(const std::string& name) { return ResolveFace(name); }

   std::vector<int> Faces(const py::args& names)
   {
      std::vector<int> out;
      for (auto n : names) { out.push_back(ResolveFace(n.cast<std::string>())); }
      return out;
   }

   std::vector<int> AllFaces()
   {
      static const char axes[3] = {'x', 'y', 'z'};
      std::vector<int> out;
      for (int ci = 0; ci < params_.mesh.dim; ++ci)
      {
         if (params_.mesh.periodic[ci]) { continue; } // periodic: no real face
         out.push_back(ResolveFace(std::string(1, axes[ci]) + "min"));
         out.push_back(ResolveFace(std::string(1, axes[ci]) + "max"));
      }
      return out;
   }

private:
   void Wire()
   {
      if (wired_) { return; }
      ApplyDeckBcs();
      case_->SetBoundaryConditions(*bc_);
      wired_ = true;
   }

   // Apply the deck's boundary_conditions groups: resolve each selection to
   // attributes, and for a Dirichlet group use the field bound by group name.
   void ApplyDeckBcs()
   {
      for (const BcSpec& s : params_.boundary_conditions)
      {
         std::vector<int> attrs;
         if (s.select_all) { attrs = AllFaces(); }
         else
         {
            attrs = s.attributes;
            for (const auto& name : s.faces)
            {
               attrs.push_back(ResolveFace(name));
            }
         }
         if (s.type == BcType::Outflow)
         {
            for (int a : attrs) { bc_->AddOutflow(a); }
         }
         else if (s.type == BcType::NoSlip)
         {
            for (int a : attrs) { bc_->AddNoSlip(a); }
         }
         else
         {
            auto it = dirichlet_fields_.find(s.group);
            MFEM_VERIFY(it != dirichlet_fields_.end(),
                        "incns: no Dirichlet field bound to boundary group '" +
                        s.group + "' -- call set_dirichlet_field(group, field)");
            for (int a : attrs) { bc_->AddVelocityDirichlet(a, *it->second); }
         }
      }
   }

   std::vector<int> AsAttrs(const py::object& a)
   {
      std::vector<int> v;
      if (py::isinstance<py::int_>(a)) { v.push_back(a.cast<int>()); }
      else { for (auto x : a) { v.push_back(x.cast<int>()); } }
      return v;
   }

   // Resolve a flexible selection (int attr | face-name str | "all" | list of
   // those) to a concrete attribute list.
   std::vector<int> ResolveSelection(const py::object& sel)
   {
      std::vector<int> out;
      auto one = [&](const py::handle & x)
      {
         if (py::isinstance<py::int_>(x)) { out.push_back(x.cast<int>()); }
         else
         {
            const std::string name = x.cast<std::string>();
            if (name == "all")
            {
               const std::vector<int> af = AllFaces();
               out.insert(out.end(), af.begin(), af.end());
            }
            else { out.push_back(ResolveFace(name)); }
         }
      };
      if (py::isinstance<py::int_>(sel) || py::isinstance<py::str>(sel))
      {
         one(sel);
      }
      else { for (auto x : sel) { one(x); } }
      return out;
   }

   int ResolveFace(const std::string& name)
   {
      MFEM_VERIFY(name.size() == 4, "incns: face name must be like 'xmin'/'xmax'");
      const int ci = (name[0] == 'x') ? 0 : (name[0] == 'y') ? 1 : 2;
      MFEM_VERIFY(ci < params_.mesh.dim, "incns: face axis exceeds dimension");
      const bool is_max = (name.compare(1, 3, "max") == 0);
      const double val = is_max ? params_.mesh.lengths[ci] : 0.0;
      int attr = 0;
      for (int be = 0; be < pmesh_->GetNBE(); ++be)
      {
         ElementTransformation* T = pmesh_->GetBdrElementTransformation(be);
         Vector c;
         T->Transform(Geometries.GetCenter(T->GetGeometryType()), c);
         if (std::abs(c(ci) - val) < 1e-9) { attr = pmesh_->GetBdrAttribute(be); }
      }
      MPI_Allreduce(MPI_IN_PLACE, &attr, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
      MFEM_VERIFY(attr > 0,
                  "incns: face '" + name + "' has no real boundary (periodic?)");
      return attr;
   }

   Parameters params_;
   Mesh serial_;
   std::unique_ptr<ParMesh> pmesh_;
   std::unique_ptr<Case> case_;
   std::unique_ptr<BoundaryConditions> bc_;
   std::unique_ptr<VectorCoefficient> ic_;
   std::unique_ptr<VectorCoefficient> forcing_;
   std::vector<std::unique_ptr<VectorCoefficient>> coeffs_;
   std::map<std::string, VectorCoefficient*> dirichlet_fields_;
   bool wired_ = false;
};

// Tuple <-> std::array<T,3> helpers for the mesh spec properties.
template <typename T>
py::tuple ArrayToTuple(const std::array<T, 3>& a, int n)
{
   py::list l;
   for (int i = 0; i < n; ++i) { l.append(a[i]); }
   return py::tuple(l);
}

template <typename T>
void SeqToArray(const py::sequence& s, std::array<T, 3>& a)
{
   int i = 0;
   for (auto x : s)
   {
      MFEM_VERIFY(i < 3, "incns: mesh spec sequence longer than 3");
      a[i++] = x.cast<T>();
   }
}

} // namespace

PYBIND11_MODULE(_core, m)
{
   m.doc() = "incns unsteady Stokes -- Python job driver (see CLAUDE.md)";

   // Initialize MPI + HYPRE once at import (SPMD: every rank imports).
   static bool initialized = false;
   if (!initialized)
   {
      if (!Mpi::IsInitialized()) { Mpi::Init(); }
      Hypre::Init();
      initialized = true;
   }

   m.def("rank", [] { return Mpi::WorldRank(); }, "This rank's index.");
   m.def("size", [] { return Mpi::WorldSize(); }, "Number of ranks.");
   m.def("on_root", [] { return Mpi::Root(); }, "True on rank 0.");

   py::enum_<ScalingMode>(m, "ScalingMode")
   .value("Dimensionless", ScalingMode::Dimensionless)
   .value("Dimensional", ScalingMode::Dimensional);

   py::enum_<Equation>(m, "Equation")
   .value("Stokes", Equation::Stokes)
   .value("NavierStokes", Equation::NavierStokes);

   py::class_<Nondimensionalization>(m, "Nondimensionalization")
   .def_readwrite("mode", &Nondimensionalization::mode)
   .def_readwrite("L_ref", &Nondimensionalization::L_ref)
   .def_readwrite("U_ref", &Nondimensionalization::U_ref)
   .def_readwrite("rho", &Nondimensionalization::rho)
   .def_readonly("Re", &Nondimensionalization::Re);

   py::class_<AdaptiveControllerOptions>(m, "AdaptiveControllerOptions")
   .def_readwrite("atol", &AdaptiveControllerOptions::atol)
   .def_readwrite("rtol", &AdaptiveControllerOptions::rtol);

   py::class_<OutputParameters>(m, "OutputParameters")
   .def_readwrite("enabled", &OutputParameters::enabled)
   .def_readwrite("path", &OutputParameters::path)
   .def_readwrite("name", &OutputParameters::name)
   .def_readwrite("interval", &OutputParameters::interval);

   py::class_<CheckpointParameters>(m, "CheckpointParameters")
   .def_readwrite("enabled", &CheckpointParameters::enabled)
   .def_readwrite("path", &CheckpointParameters::path)
   .def_readwrite("interval", &CheckpointParameters::interval);

   py::class_<BoxSpec>(m, "BoxSpec")
   .def_readwrite("dim", &BoxSpec::dim)
   .def_property(
      "num_elems",
   [](const BoxSpec & s) { return ArrayToTuple(s.num_elems, s.dim); },
   [](BoxSpec & s, const py::sequence & v) { SeqToArray(v, s.num_elems); })
   .def_property(
      "lengths",
   [](const BoxSpec & s) { return ArrayToTuple(s.lengths, s.dim); },
   [](BoxSpec & s, const py::sequence & v) { SeqToArray(v, s.lengths); })
   .def_property(
      "periodic",
   [](const BoxSpec & s) { return ArrayToTuple(s.periodic, s.dim); },
   [](BoxSpec & s, const py::sequence & v) { SeqToArray(v, s.periodic); })
   .def("box",
        [](BoxSpec & s, int dim, const py::sequence & elements,
           const py::object & lengths, const py::object & periodic)
   {
      s.dim = dim;
      SeqToArray(elements.cast<py::sequence>(), s.num_elems);
      if (!lengths.is_none())
      {
         SeqToArray(lengths.cast<py::sequence>(), s.lengths);
      }
      if (!periodic.is_none())
      {
         SeqToArray(periodic.cast<py::sequence>(), s.periodic);
      }
      else { s.periodic = {false, false, false}; }
   },
   py::arg("dim"), py::arg("elements"), py::arg("lengths") = py::none(),
   py::arg("periodic") = py::none(),
   "Configure the Cartesian box: dim, elements, lengths (default 2*pi), "
   "periodic (default all False).");

   py::class_<Parameters>(m, "Parameters")
   .def(py::init<>())
   .def_static("from_yaml", &Parameters::LoadYAML, py::arg("path"))
   .def("normalize", &Parameters::Normalize)
   .def_readwrite("equation", &Parameters::equation)
   .def_readwrite("nu", &Parameters::nu)
   .def_readwrite("grad_div", &Parameters::grad_div)
   .def_readwrite("order_u", &Parameters::order_u)
   .def_readwrite("order_p", &Parameters::order_p)
   .def_readwrite("collocated_mass", &Parameters::collocated_mass)
   .def_readwrite("dt", &Parameters::dt)
   .def_readwrite("t_final", &Parameters::t_final)
   .def_readwrite("time_order", &Parameters::time_order)
   .def_readwrite("adaptive", &Parameters::adaptive)
   .def_readwrite("krylov_rtol", &Parameters::krylov_rtol)
   .def_readwrite("krylov_atol", &Parameters::krylov_atol)
   .def_readwrite("max_iter", &Parameters::max_iter)
   .def_readwrite("kdim", &Parameters::kdim)
   .def_readwrite("print_level", &Parameters::print_level)
   .def_readwrite("initial_velocity", &Parameters::initial_velocity)
   .def_readwrite("restart_from", &Parameters::restart_from)
   .def_readwrite("mesh", &Parameters::mesh)
   .def_readwrite("controller", &Parameters::controller)
   .def_readwrite("output", &Parameters::output)
   .def_readwrite("checkpoint", &Parameters::checkpoint)
   .def_readwrite("nondim", &Parameters::nondim)
   .def_property(
   "reynolds", [](const Parameters & p) { return 1.0 / p.nu; },
   [](Parameters & p, double re) { p.nu = 1.0 / re; },
   "Convenience: get/set nu = 1/Re (dimensionless mode).");

   py::class_<PyCase>(m, "Case")
   .def(py::init<Parameters>(), py::arg("params"))
   .def_static(
      "from_yaml",
      [](const std::string & path)
   {
      return std::make_unique<PyCase>(Parameters::LoadYAML(path));
   },
   py::arg("path"),
   "Build a case from a YAML deck (config + boundary-condition topology).")
   .def("set_initial_velocity", &PyCase::SetInitialVelocity, py::arg("f"))
   .def("set_forcing", &PyCase::SetForcing, py::arg("f"))
   .def("set_dirichlet_field", &PyCase::SetDirichletField,
        py::arg("group"), py::arg("f"))
   .def("velocity_dirichlet", &PyCase::VelocityDirichlet,
        py::arg("attributes"), py::arg("f"))
   .def("outflow", &PyCase::Outflow, py::arg("attributes"))
   .def("no_slip", &PyCase::NoSlip, py::arg("selection"))
   .def("run", &PyCase::Run)
   .def("step", &PyCase::Step)
   .def("velocity_l2_error", &PyCase::VelocityL2Error, py::arg("f"))
   .def("face", &PyCase::Face, py::arg("name"))
   .def("faces", &PyCase::Faces)
   .def("all_faces", &PyCase::AllFaces)
   .def_property_readonly("time", &PyCase::Time)
   .def_property_readonly("time_dimensional", &PyCase::TimeDimensional)
   .def_property_readonly("done", &PyCase::Done)
   .def_property_readonly("step_count", &PyCase::StepCount)
   .def_property_readonly("iterations", &PyCase::Iterations);
}
