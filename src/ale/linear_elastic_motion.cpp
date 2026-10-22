#include "ale/linear_elastic_motion.hpp"

#include <cmath>
#include <cstdint>
#include <istream>
#include <ostream>

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
   opts_ = opts;
}

bool LinearElasticMotion::Solve(double t, const ParGridFunction* guess)
{
   if (opts_.basis_size <= 0)
   {
      if (!guess) { return solver_->Solve(t); }
      guess->GetTrueDofs(guess_);
      return solver_->Solve(t, guess_);
   }
   // The data's coordinates in the kept (orthonormal) data: classical
   // Gram-Schmidt, twice.
   MPI_Comm comm = solver_->Space().GetComm();
   solver_->DirichletValues(t, g_);
   const double gnorm = std::sqrt(InnerProduct(comm, g_, g_));
   const std::size_t m = basis_g_.size();
   std::vector<double> c(m, 0.0);
   Vector r(g_);
   for (int pass = 0; pass < 2; ++pass)
   {
      std::vector<double> d(m);
      for (std::size_t i = 0; i < m; ++i) { d[i] = basis_g_[i] * r; }
      if (m > 0)
      {
         MPI_Allreduce(MPI_IN_PLACE, d.data(), static_cast<int>(m),
                       MPI_DOUBLE, MPI_SUM, comm);
      }
      for (std::size_t i = 0; i < m; ++i)
      {
         r.Add(-d[i], basis_g_[i]);
         c[i] += d[i];
      }
   }
   const double rnorm = std::sqrt(InnerProduct(comm, r, r));
   const int n = solver_->Space().GetTrueVSize();
   x_.SetSize(n);
   x_ = 0.0;
   for (std::size_t i = 0; i < m; ++i) { x_.Add(c[i], basis_x_[i]); }
   // In the span (to roundoff): the same combination of the solutions.
   // The load is zero (a mesh motion has only boundary data), so the
   // displacement is linear in the data.
   if (rnorm <= 1e-12 * gnorm || gnorm == 0.0)
   {
      solver_->SetSolution(x_);
      ++reused_;
      return true;
   }
   bool ok = false;
   if (guess)
   {
      guess->GetTrueDofs(guess_);
      ok = solver_->Solve(t, guess_);
   }
   else { ok = solver_->Solve(t); }
   if (ok && static_cast<int>(m) < opts_.basis_size)
   {
      // Keep the new direction: data r / |r|, solution (x - sum c_i x_i) / |r|.
      Vector x(n);
      solver_->Displacement().GetTrueDofs(x);
      x -= x_;
      x /= rnorm;
      r /= rnorm;
      basis_g_.push_back(r);
      basis_x_.push_back(x);
   }
   return ok;
}

void LinearElasticMotion::SaveState(std::ostream& out) const
{
   const std::int64_t m = static_cast<std::int64_t>(basis_x_.size());
   const std::int64_t ng = m > 0 ? basis_g_[0].Size() : 0;
   const std::int64_t nx = m > 0 ? basis_x_[0].Size() : 0;
   out.write(reinterpret_cast<const char*>(&m), sizeof(m));
   out.write(reinterpret_cast<const char*>(&ng), sizeof(ng));
   out.write(reinterpret_cast<const char*>(&nx), sizeof(nx));
   for (std::int64_t i = 0; i < m; ++i)
   {
      out.write(reinterpret_cast<const char*>(basis_g_[i].HostRead()),
                static_cast<std::streamsize>(ng * sizeof(double)));
      out.write(reinterpret_cast<const char*>(basis_x_[i].HostRead()),
                static_cast<std::streamsize>(nx * sizeof(double)));
   }
}

void LinearElasticMotion::LoadState(std::istream& in)
{
   std::int64_t m = 0, ng = 0, nx = 0;
   in.read(reinterpret_cast<char*>(&m), sizeof(m));
   in.read(reinterpret_cast<char*>(&ng), sizeof(ng));
   in.read(reinterpret_cast<char*>(&nx), sizeof(nx));
   MFEM_VERIFY(in && m >= 0 && (m == 0 ||
                                nx == solver_->Space().GetTrueVSize()),
               "mesh_motion: the checkpoint's linear elastic state does not "
               "match this mesh");
   basis_g_.assign(m, Vector(ng));
   basis_x_.assign(m, Vector(nx));
   for (std::int64_t i = 0; i < m; ++i)
   {
      in.read(reinterpret_cast<char*>(basis_g_[i].HostWrite()),
              static_cast<std::streamsize>(ng * sizeof(double)));
      in.read(reinterpret_cast<char*>(basis_x_[i].HostWrite()),
              static_cast<std::streamsize>(nx * sizeof(double)));
   }
   MFEM_VERIFY(in, "mesh_motion: reading the linear elastic state failed");
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
