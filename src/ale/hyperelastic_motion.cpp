#include "ale/hyperelastic_motion.hpp"

#include "ale/linear_elastic_motion.hpp"

#include <cstdint>
#include <istream>
#include <ostream>

namespace incns
{

using namespace mfem;

HyperelasticMotion::HyperelasticMotion(ParMesh& reference, int order,
                                       const Options& opts)
   : material_(opts.material)
{
   const double nu = opts.poisson_ratio;
   MFEM_VERIFY(nu > -1.0 && nu < 0.5, "mesh_motion: the Poisson ratio must be "
               "in (-1, 1/2)");
   MFEM_VERIFY(opts.stiffening >= 0.0, "mesh_motion: the stiffening exponent "
               "must be >= 0");
   MFEM_VERIFY(opts.material == HyperelasticMaterial::NeoHookean ||
               opts.material == HyperelasticMaterial::StVenantKirchhoff,
               "mesh_motion: a hyperelastic material");
   Hyperelasticity::Options o;
   o.order = order;
   o.material = opts.material;
   // E = 1: the displacement of a problem driven by Dirichlet data alone
   // does not depend on it. Plane strain in 2D.
   o.mu = 1.0 / (2.0 * (1.0 + nu));
   o.lambda = nu / ((1.0 + nu) * (1.0 - 2.0 * nu));
   o.rtol = opts.rtol;
   o.atol = opts.atol;
   o.max_iter = opts.max_iter;
   o.print_level = opts.print_level;
   o.newton_rtol = opts.newton_rtol;
   o.newton_dx_rtol = opts.newton_dx_rtol;
   o.newton_max_iter = opts.newton_max_iter;
   o.load_steps = opts.load_steps;
   o.line_search = opts.line_search;
   if (opts.stiffening > 0.0)
   {
      factors_ = LinearElasticMotion::StiffeningFactors(reference, opts.stiffening);
      o.element_stiffness = &factors_;
   }
   solver_ = std::make_unique<Hyperelasticity>(reference, o);
}

bool HyperelasticMotion::Solve(double t, const ParGridFunction& guess)
{
   x_.SetSize(solver_->Space().GetTrueVSize());
   guess.GetTrueDofs(x_);
   solver_->SetSolution(x_);
   return solver_->Solve(t);
}

void HyperelasticMotion::SaveState(std::ostream& out) const
{
   // The displacement Newton starts from when no guess is given.
   Vector x(solver_->Space().GetTrueVSize());
   static_cast<const Hyperelasticity&>(*solver_).Displacement().GetTrueDofs(x);
   const std::int64_t n = x.Size();
   out.write(reinterpret_cast<const char*>(&n), sizeof(n));
   out.write(reinterpret_cast<const char*>(x.HostRead()),
             static_cast<std::streamsize>(n * sizeof(double)));
}

void HyperelasticMotion::LoadState(std::istream& in)
{
   std::int64_t n = 0;
   in.read(reinterpret_cast<char*>(&n), sizeof(n));
   MFEM_VERIFY(in && n == solver_->Space().GetTrueVSize(), "mesh_motion: the "
               "checkpoint's hyperelastic state does not match this mesh");
   x_.SetSize(static_cast<int>(n));
   in.read(reinterpret_cast<char*>(x_.HostWrite()),
           static_cast<std::streamsize>(n * sizeof(double)));
   MFEM_VERIFY(in, "mesh_motion: reading the hyperelastic state failed");
   solver_->SetSolution(x_);
}

MeshMotionModelFactory HyperelasticMotion::Factory(const Options& opts)
{
   return [opts](ParMesh & reference, int order)
   {
      return std::unique_ptr<MeshMotionModel>(
                new HyperelasticMotion(reference, order, opts));
   };
}

} // namespace incns
