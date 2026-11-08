#include "bc/deck_boundary.hpp"

#include "bc/boundary_names.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <map>

namespace incns
{

using namespace mfem;

namespace
{
// s(t) of a deck-given boundary velocity.
double TimeFactor(const BcSpec& s, double t)
{
   switch (s.time_profile)
   {
      case BcTimeProfile::Constant: return 1.0;
      case BcTimeProfile::Ramp:
      {
         if (t >= s.time_scale) { return 1.0; }
         const double r = std::sin(0.5 * M_PI * t / s.time_scale);
         return r * r;
      }
      case BcTimeProfile::Sine: return std::sin(M_PI * t / s.time_scale);
   }
   return 1.0;
}

// The pipe profile's frame and perturbation phases, from its spec.
struct PipeFrame
{
   double a[3], e1[3], e2[3];                   // axis and cross-section basis
   double phi[kPipePerturbationModes];          // azimuthal phases
   double psi[kPipePerturbationModes];          // temporal phases
};

PipeFrame MakePipeFrame(const BcSpec& s)
{
   PipeFrame f;
   const double n = std::sqrt(s.axis[0] * s.axis[0] + s.axis[1] * s.axis[1] +
                              s.axis[2] * s.axis[2]);
   for (int d = 0; d < 3; ++d) { f.a[d] = s.axis[d] / n; }
   // e1: the coordinate direction least aligned with a, made orthogonal.
   int c = 0;
   for (int d = 1; d < 3; ++d)
   {
      if (std::abs(f.a[d]) < std::abs(f.a[c])) { c = d; }
   }
   double e[3] = {0.0, 0.0, 0.0};
   e[c] = 1.0;
   double dot = f.a[c];
   double m = 0.0;
   for (int d = 0; d < 3; ++d)
   {
      f.e1[d] = e[d] - dot * f.a[d];
      m += f.e1[d] * f.e1[d];
   }
   m = std::sqrt(m);
   for (int d = 0; d < 3; ++d) { f.e1[d] /= m; }
   f.e2[0] = f.a[1] * f.e1[2] - f.a[2] * f.e1[1];
   f.e2[1] = f.a[2] * f.e1[0] - f.a[0] * f.e1[2];
   f.e2[2] = f.a[0] * f.e1[1] - f.a[1] * f.e1[0];
   // Phases from a 64-bit linear congruential generator (Knuth's MMIX
   // constants): the same sequence on every rank and platform.
   std::uint64_t state = 0x9E3779B97F4A7C15ull ^
                         static_cast<std::uint64_t>(s.perturbation_seed);
   auto next = [&state]()
   {
      state = state * 6364136223846793005ull + 1442695040888963407ull;
      return static_cast<double>(state >> 11) * (1.0 / 9007199254740992.0);
   };
   for (int k = 0; k < kPipePerturbationModes; ++k)
   {
      f.phi[k] = 2.0 * M_PI * next();
      f.psi[k] = 2.0 * M_PI * next();
   }
   return f;
}

void PipeVelocityInFrame(const BcSpec& s, const PipeFrame& f, const Vector& x,
                         double t, Vector& u)
{
   u.SetSize(3);
   u = 0.0;
   double r[3], ra = 0.0;
   for (int d = 0; d < 3; ++d)
   {
      r[d] = x(d) - s.center[d];
      ra += r[d] * f.a[d];
   }
   double p1 = 0.0, p2 = 0.0;
   for (int d = 0; d < 3; ++d)
   {
      const double q = r[d] - ra * f.a[d];
      p1 += q * f.e1[d];
      p2 += q * f.e2[d];
   }
   const double rho = std::sqrt(p1 * p1 + p2 * p2) / s.radius;
   if (rho >= 1.0) { return; }
   const double w = 1.0 - rho * rho;
   double ua = 2.0 * s.u_mean * w;
   if (s.perturbation > 0.0)
   {
      const int K = kPipePerturbationModes;
      const double theta = std::atan2(p2, p1);
      const double amp = s.perturbation * s.u_mean / std::sqrt(double(K));
      for (int k = 0; k < K; ++k)
      {
         const int mk = 1 + k % 3;
         // rho^m (1 - rho^2) peaks at rho^2 = m / (m + 2).
         const double r2 = double(mk) / (mk + 2.0);
         const double peak = std::pow(r2, 0.5 * mk) * (1.0 - r2);
         const double shape = std::pow(rho, mk) * w / peak;
         const double fk = (k + 1.0) * s.perturbation_frequency / K;
         ua += amp * shape * std::cos(mk * theta + f.phi[k]) *
               std::sin(2.0 * M_PI * fk * t + f.psi[k]);
      }
   }
   for (int d = 0; d < 3; ++d) { u(d) = ua * f.a[d]; }
}

// u = sc g(x) of a deck-given boundary velocity, sc multiplied first (as the
// data always was): sc = 1 gives the shape g bit for bit.
void Profile(const BcSpec& spec, int dim, const Vector& x, double sc, Vector& u)
{
   u = 0.0;
   if (spec.profile == BcProfile::Constant)
   {
      for (int d = 0; d < dim; ++d) { u(d) = sc * spec.value[d]; }
   }
   else if (spec.profile == BcProfile::Pipe)
   {
      // Unperturbed (the separable form): perturbed pipes take
      // PipeVelocityInFrame directly.
      PipeVelocityInFrame(spec, MakePipeFrame(spec), x, 0.0, u);
      u *= sc;
   }
   else
   {
      const double h = spec.height, y = x(1);
      u(0) = sc * 4.0 * spec.u_max * y * (h - y) / (h * h);
      if (dim == 3)
      {
         // The DFG 3D inflow: a product of parabolas in y and z.
         const double w = (spec.width > 0.0) ? spec.width : h;
         u(0) *= 4.0 * x(2) * (w - x(2)) / (w * w);
      }
   }
}
} // namespace

void PipeVelocity(const BcSpec& spec, const Vector& x, double t, Vector& u)
{
   MFEM_VERIFY(spec.profile == BcProfile::Pipe, "PipeVelocity: not a pipe");
   PipeVelocityInFrame(spec, MakePipeFrame(spec), x, t, u);
}

DeckBoundaryConditions::DeckBoundaryConditions(const Parameters& p,
      ParMesh& mesh)
   : params_(p)
{
   real_ = AllBoundaryAttributes(p, mesh);
   for (const std::string& n : BoundaryNames(p))
   {
      attr_name_[NamedBoundaryAttribute(p, mesh, n)] = n;
   }
   for (const BcSpec& s : p.boundary_conditions)
   {
      std::vector<int> a;
      if (s.select_all) { a = real_; }
      else
      {
         a = s.attributes;
         for (const std::string& name : s.faces)
         {
            a.push_back(NamedBoundaryAttribute(p, mesh, name));
         }
      }
      attrs_.push_back(a);
   }
}

std::vector<std::string> DeckBoundaryConditions::CoverageProblems() const
{
   std::map<int, int> count;
   for (const auto& a : attrs_)
   {
      for (int x : a) { ++count[x]; }
   }
   std::vector<std::string> out;
   const std::vector<std::string> names = BoundaryNames(params_);
   auto label = [&](int attr)
   {
      const auto it = attr_name_.find(attr);
      return (it != attr_name_.end() ? "boundary '" + it->second + "' (attribute "
              : std::string("(attribute ")) + std::to_string(attr) + ")";
   };
   for (int attr : real_)
   {
      if (!count.count(attr))
      {
         out.push_back(label(attr) + " has no boundary condition");
      }
      else if (count[attr] > 1)
      {
         out.push_back(label(attr) + " is selected by " +
                       std::to_string(count[attr]) + " groups");
      }
   }
   if (!out.empty())
   {
      std::string n;
      for (const std::string& s : names) { n += (n.empty() ? "" : ", ") + s; }
      out.push_back("(boundary names here: " + n + ")");
   }
   return out;
}

void DeckBoundaryConditions::Apply(BoundaryConditions& bc,
                                   const FieldLookup& field,
                                   bool require_coverage,
                                   VectorCoefficient* mesh_velocity,
                                   Coefficient* robin_alpha,
                                   double robin_alpha_max)
{
   if (require_coverage)
   {
      const std::vector<std::string> problems = CoverageProblems();
      if (!problems.empty())
      {
         std::string msg;
         for (const std::string& s : problems) { msg += "\n  " + s; }
         MFEM_ABORT("boundary_conditions: every real boundary needs exactly one "
                    "group:" << msg);
      }
   }
   const int dim = params_.mesh.dim;
   for (std::size_t g = 0; g < params_.boundary_conditions.size(); ++g)
   {
      const BcSpec& s = params_.boundary_conditions[g];
      switch (s.type)
      {
         case BcType::Outflow:
            for (int a : attrs_[g]) { bc.AddOutflow(a); }
            break;
         case BcType::NoSlip:
            if (!s.motion.empty())
            {
               // A moving wall: u = w (the fluid sticks to the moving mesh).
               MFEM_VERIFY(mesh_velocity, "boundary_conditions: the no_slip "
                           "group " << g << " moves, but no mesh velocity is "
                           "given (the case's mesh does not move?)");
               for (int a : attrs_[g]) { bc.AddVelocityDirichlet(a, *mesh_velocity); }
               break;
            }
            for (int a : attrs_[g]) { bc.AddNoSlip(a); }
            break;
         case BcType::Interface:
            // The wetted surface moves with the solid: u = w, imposed
            // (Dirichlet-Neumann) or weakly through the Robin condition
            // (Robin-Neumann; the time integrator's boundary load carries the
            // traction).
            MFEM_VERIFY(mesh_velocity, "boundary_conditions: the interface "
                        "group " << g << " needs the mesh velocity (an fsi "
                        "case)");
            if (params_.fsi.enabled &&
                (params_.fsi.coupling == FsiCoupling::RobinNeumann ||
                 params_.fsi.coupling == FsiCoupling::RobinRobin))
            {
               for (int a : attrs_[g])
               {
                  if (robin_alpha)
                  {
                     bc.AddRobin(a, *robin_alpha, robin_alpha_max, *mesh_velocity);
                  }
                  else
                  {
                     bc.AddRobin(a, params_.FsiFluidRobin(), *mesh_velocity);
                  }
               }
               break;
            }
            if (params_.fsi.enabled &&
                params_.fsi.coupling == FsiCoupling::NeumannDirichlet)
            {
               // A traction boundary: the boundary traction carries the
               // solid's reaction.
               for (int a : attrs_[g]) { bc.AddRobin(a, 0.0, *mesh_velocity); }
               break;
            }
            for (int a : attrs_[g]) { bc.AddVelocityDirichlet(a, *mesh_velocity); }
            break;
         case BcType::VelocityDirichlet:
         {
            VectorCoefficient* f = field ? field(s.group) : nullptr;
            MFEM_VERIFY(f, "boundary_conditions: no field is bound to the "
                        "velocity_dirichlet group '" << s.group << "' (bind one "
                        "from Python or C++, or use type velocity with a value "
                        "or profile in the deck)");
            for (int a : attrs_[g]) { bc.AddVelocityDirichlet(a, *f); }
            break;
         }
         case BcType::Velocity:
         {
            const BcSpec spec = s; // the coefficient keeps its own copy
            if (spec.profile == BcProfile::Pipe && spec.perturbation > 0.0)
            {
               // Not separable in time: projected at every use.
               const PipeFrame frame = MakePipeFrame(spec);
               coeffs_.push_back(std::make_unique<VectorFunctionCoefficient>(
                                    dim,
                                    [spec, frame](const Vector & x, double t,
                                                  Vector & u)
               {
                  PipeVelocityInFrame(spec, frame, x, t, u);
                  u *= TimeFactor(spec, t);
               }));
               for (int a : attrs_[g])
               {
                  bc.AddVelocityDirichlet(a, *coeffs_.back());
               }
               break;
            }
            // s(t) g(x): the OIFS substeps project g once and scale it.
            coeffs_.push_back(std::make_unique<SeparableVectorCoefficient>(
                                 dim,
                                 [spec, dim](const Vector & x, double t, Vector & u)
            {
               Profile(spec, dim, x, TimeFactor(spec, t), u);
            },
            [spec, dim](const Vector & x, Vector & u)
            {
               Profile(spec, dim, x, 1.0, u);
            },
            [spec](double t) { return TimeFactor(spec, t); }));
            for (int a : attrs_[g]) { bc.AddVelocityDirichlet(a, *coeffs_.back()); }
            break;
         }
         case BcType::Fixed:
         case BcType::Displacement:
         case BcType::Traction:
         case BcType::Free:
            MFEM_ABORT("boundary_conditions: group " << g << " has a solid "
                       "boundary type (fixed, displacement, traction, free) "
                       "in a flow case");
            break;
      }
   }
}

} // namespace incns
