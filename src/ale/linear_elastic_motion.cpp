#include "ale/linear_elastic_motion.hpp"

#include <cmath>

namespace incns
{

using namespace mfem;

LinearElasticMotion::LinearElasticMotion(ParMesh& reference, int order,
      const Options& opts)
{
   const double nu = opts.poisson_ratio;
   MFEM_VERIFY(nu > -1.0 && nu < 0.5, "mesh_motion: the Poisson ratio must be "
               "in (-1, 1/2)");
   MFEM_VERIFY(opts.stiffening >= 0.0, "mesh_motion: the stiffening exponent "
               "must be >= 0");
   LinearElasticity::Options o;
   o.order = order;
   // E = 1: the displacement of a problem driven by Dirichlet data alone
   // does not depend on it. Plane strain in 2D.
   o.mu = 1.0 / (2.0 * (1.0 + nu));
   o.lambda = nu / ((1.0 + nu) * (1.0 - 2.0 * nu));
   o.rtol = opts.rtol;
   o.atol = opts.atol;
   o.max_iter = opts.max_iter;
   o.preconditioner = opts.preconditioner;
   o.print_level = opts.print_level;
   if (opts.stiffening > 0.0)
   {
      factors_ = StiffeningFactors(reference, opts.stiffening);
      o.element_stiffness = &factors_;
   }
   solver_ = std::make_unique<LinearElasticity>(reference, o);
}

MeshMotionModelFactory LinearElasticMotion::Factory(const Options& opts)
{
   return [opts](ParMesh & reference, int order)
   {
      return std::unique_ptr<MeshMotionModel>(
                new LinearElasticMotion(reference, order, opts));
   };
}

Vector LinearElasticMotion::StiffeningFactors(ParMesh& mesh, double chi)
{
   const int ne = mesh.GetNE();
   Vector f(ne);
   double sums[2] = {0.0, static_cast<double>(ne)};
   for (int e = 0; e < ne; ++e)
   {
      f(e) = mesh.GetElementVolume(e);
      MFEM_VERIFY(f(e) > 0.0, "mesh_motion: element " << e << " of the "
                  "reference mesh has a non-positive volume");
      sums[0] += f(e);
   }
   MPI_Allreduce(MPI_IN_PLACE, sums, 2, MPI_DOUBLE, MPI_SUM, mesh.GetComm());
   const double mean = sums[0] / sums[1];
   for (int e = 0; e < ne; ++e) { f(e) = std::pow(mean / f(e), chi); }
   return f;
}

} // namespace incns
