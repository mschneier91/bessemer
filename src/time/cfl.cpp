#include "time/cfl.hpp"

#include "mfem/general/forall.hpp"

#include <cmath>

// This TU defines a device kernel (mfem::forall). A host compiler would build
// it as a host loop over device pointers, which segfaults on a GPU;
// src/CMakeLists.txt marks this file LANGUAGE CUDA, and this line turns a lost
// entry there into a compile error.
#if defined(MFEM_USE_CUDA) && !defined(__CUDACC__)
#error "cfl.cpp defines device kernels and must be compiled by nvcc -- see the nvcc TU list in src/CMakeLists.txt"
#endif

namespace incns
{

using namespace mfem;

// Kernel in a NAMED namespace: nvcc rejects extended lambdas whose enclosing
// function has internal linkage.
namespace cfl_kernels
{

// rate[e] = max over the GLL nodes q = (i, j[, l]) (lexicographic, x
// fastest) of sum_d |(J_q^{-1} u_q)_d| * inv_dxi[index_d(q)], from u at the
// nodes U(q, c, e) (byNODES) and Jacobians J(q, i, j, e) = dx_i/dxi_j.
void ElementRates(int ne, int d1, int dim, const Vector& uq, const Vector& jac,
                  const Vector& inv_dxi, Vector& rate)
{
   const int nq = (dim == 3) ? d1 * d1 * d1 : d1 * d1;
   const auto U = Reshape(uq.Read(), nq, dim, ne);
   const auto J = Reshape(jac.Read(), nq, dim, dim, ne);
   const auto X = inv_dxi.Read();
   auto R = rate.Write();
   mfem::forall(ne, [ = ] MFEM_HOST_DEVICE(int e)
   {
      real_t rmax = 0.0;
      for (int q = 0; q < nq; ++q)
      {
         const int ix = q % d1, iy = (q / d1) % d1, iz = q / (d1 * d1);
         real_t s = 0.0;
         if (dim == 2)
         {
            const real_t a = J(q, 0, 0, e), b = J(q, 0, 1, e);
            const real_t c = J(q, 1, 0, e), d = J(q, 1, 1, e);
            const real_t det = a * d - b * c;
            const real_t u0 = U(q, 0, e), u1 = U(q, 1, e);
            // J^{-1} = [d -b; -c a] / det
            s = fabs((d * u0 - b * u1) / det) * X[ix] +
                fabs((-c * u0 + a * u1) / det) * X[iy];
         }
         else
         {
            real_t m[3][3];
            for (int i = 0; i < 3; ++i)
               for (int j = 0; j < 3; ++j) { m[i][j] = J(q, i, j, e); }
            const real_t a00 = m[1][1] * m[2][2] - m[1][2] * m[2][1];
            const real_t a01 = m[0][2] * m[2][1] - m[0][1] * m[2][2];
            const real_t a02 = m[0][1] * m[1][2] - m[0][2] * m[1][1];
            const real_t a10 = m[1][2] * m[2][0] - m[1][0] * m[2][2];
            const real_t a11 = m[0][0] * m[2][2] - m[0][2] * m[2][0];
            const real_t a12 = m[0][2] * m[1][0] - m[0][0] * m[1][2];
            const real_t a20 = m[1][0] * m[2][1] - m[1][1] * m[2][0];
            const real_t a21 = m[0][1] * m[2][0] - m[0][0] * m[2][1];
            const real_t a22 = m[0][0] * m[1][1] - m[0][1] * m[1][0];
            const real_t det = m[0][0] * a00 + m[0][1] * a10 + m[0][2] * a20;
            const real_t u0 = U(q, 0, e), u1 = U(q, 1, e), u2 = U(q, 2, e);
            s = fabs((a00 * u0 + a01 * u1 + a02 * u2) / det) * X[ix] +
                fabs((a10 * u0 + a11 * u1 + a12 * u2) / det) * X[iy] +
                fabs((a20 * u0 + a21 * u1 + a22 * u2) / det) * X[iz];
         }
         rmax = (s > rmax) ? s : rmax;
      }
      R[e] = rmax;
   });
}

} // namespace cfl_kernels

Vector ConvectiveCfl::InverseNodeSpacing(int k)
{
   // GLL points on [0,1] (MFEM's Poly_1D); Nek's getdr on the same points.
   const real_t* z = poly1d.GetPoints(k, BasisType::GaussLobatto);
   Vector inv(k + 1);
   for (int i = 0; i <= k; ++i)
   {
      const real_t dz = (i == 0) ? z[1] - z[0]
                        : (i == k) ? z[k] - z[k - 1]
                        : 0.5 * (z[i + 1] - z[i - 1]);
      inv(i) = 1.0 / dz;
   }
   return inv;
}

ConvectiveCfl::ConvectiveCfl(const ParFiniteElementSpace& vfes,
                             const RuleBook& rules)
   : vfes_(vfes)
{
   ParMesh& mesh = *vfes.GetParMesh();
   dim_ = mesh.Dimension();
   ne_ = vfes.GetNE();
   order_ = vfes.GetTypicalFE()->GetOrder();
   MFEM_VERIFY(order_ >= 1, "cfl: velocity order must be >= 1");
   MFEM_VERIFY(vfes.GetVDim() == dim_, "cfl: velocity vdim must equal dim");
   const Geometry::Type geom = (dim_ == 3) ? Geometry::CUBE : Geometry::SQUARE;
   // The k+1 GLL points per direction: the H1 nodes, where Nek evaluates.
   ir_ = &rules.Get(geom, 2 * order_ - 1, Rule1D::GaussLobatto);
   MFEM_VERIFY(ir_->GetNPoints() == ((dim_ == 3) ? (order_ + 1) * (order_ + 1) *
                                     (order_ + 1) : (order_ + 1) * (order_ + 1)),
               "cfl: expected k+1 GLL points per direction");
   restr_ = vfes.GetElementRestriction(ElementDofOrdering::LEXICOGRAPHIC);
   qi_ = vfes.GetQuadratureInterpolator(*ir_);
   qi_->SetOutputLayout(QVectorLayout::byNODES);
   geom_ = mesh.GetGeometricFactors(*ir_, GeometricFactors::JACOBIANS,
                                    Device::GetDeviceMemoryType());
   inv_dxi_ = InverseNodeSpacing(order_);
   inv_dxi_.UseDevice(true);
   ue_.SetSize(restr_->Height());
   ue_.UseDevice(true);
   uq_.SetSize(ir_->GetNPoints() * dim_ * ne_);
   uq_.UseDevice(true);
   rate_.SetSize(ne_);
   rate_.UseDevice(true);
}

double ConvectiveCfl::Rate(const ParGridFunction& u) const
{
   MFEM_VERIFY(u.ParFESpace() == &vfes_, "cfl: u must live on the CFL space");
   double local = 0.0;
   if (ne_ > 0)
   {
      restr_->Mult(u, ue_);
      qi_->SetOutputLayout(QVectorLayout::byNODES); // shared object: set it
      qi_->Values(ue_, uq_);
      cfl_kernels::ElementRates(ne_, order_ + 1, dim_, uq_, geom_->J, inv_dxi_,
                                rate_);
      local = rate_.Max(); // device-aware reduction
   }
   double global = 0.0;
   MPI_Allreduce(&local, &global, 1, MPI_DOUBLE, MPI_MAX, vfes_.GetComm());
   return global;
}

} // namespace incns
