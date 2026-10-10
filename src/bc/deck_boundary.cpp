#include "bc/deck_boundary.hpp"

#include "bc/boundary_names.hpp"

#include <algorithm>
#include <cmath>
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
} // namespace

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
                                   bool require_coverage)
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
            for (int a : attrs_[g]) { bc.AddNoSlip(a); }
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
            coeffs_.push_back(std::make_unique<VectorFunctionCoefficient>(
                                 dim, [spec, dim](const Vector & x, double t, Vector & u)
            {
               const double sc = TimeFactor(spec, t);
               u = 0.0;
               if (spec.profile == BcProfile::Constant)
               {
                  for (int d = 0; d < dim; ++d) { u(d) = sc * spec.value[d]; }
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
            }));
            for (int a : attrs_[g]) { bc.AddVelocityDirichlet(a, *coeffs_.back()); }
            break;
         }
      }
   }
}

} // namespace incns
