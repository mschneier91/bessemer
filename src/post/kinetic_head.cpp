#include "post/kinetic_head.hpp"

#include "mfem/general/forall.hpp"

// This TU defines device kernels (forall_2D + MFEM_SHARED tiles); see
// src/CMakeLists.txt's nvcc list. A host compiler would build them as host
// loops over device pointers -- this line turns a lost entry into a compile
// error.
#if defined(MFEM_USE_CUDA) && !defined(__CUDACC__)
#error "kinetic_head.cpp defines device kernels and must be compiled by nvcc -- see the nvcc TU list in src/CMakeLists.txt"
#endif

namespace incns
{

using namespace mfem;

namespace
{
/**
 * @brief 1/2 |u_h|^2 at a point (the host reference's coefficient).
 */
class HalfSpeedSquared : public Coefficient
{
   const GridFunction& u_; ///< The velocity field (not owned).
   Vector U_;              ///< Scratch: u_h at the current point.

public:
   /**
    * @brief Coefficient for the field @p u.
    * @param u Velocity GridFunction (must outlive the coefficient).
    */
   explicit HalfSpeedSquared(const GridFunction& u) : u_(u) { }
   /**
    * @brief Evaluate 1/2 |u_h|^2.
    * @param T  Element transformation (its element selects u's dofs).
    * @param ip Point in the reference element.
    * @return 1/2 |u_h(x)|^2.
    */
   real_t Eval(ElementTransformation& T, const IntegrationPoint& ip) override
   {
      u_.GetVectorValue(T, ip, U_);
      return 0.5 * (U_ * U_);
   }
};
} // namespace

// Kernels in a NAMED namespace: nvcc rejects extended lambdas whose enclosing
// function has internal linkage.
namespace kinetic_head
{

// u at the pressure nodes by sum factorization (x, then y, then z), then
// Y(q) = sum_c 1/2 u_c(q)^2. B(q, d) is the 1D velocity basis at the 1D
// pressure nodes. Threads cover max(D1D, Q1D) in each tile dimension; each
// (qx, qy) thread owns its output column, so the per-component accumulation
// into Y needs no atomics.
void Eval2D(int ne, int D1D, int Q1D, const Vector& b, const Vector& ue,
            Vector& ye)
{
   const auto B = Reshape(b.Read(), Q1D, D1D);
   const auto U = Reshape(ue.Read(), D1D, D1D, 2, ne);
   auto Y = Reshape(ye.Write(), Q1D, Q1D, ne);
   const int T = (D1D > Q1D) ? D1D : Q1D;
   mfem::forall_2D(ne, T, T, [ = ] MFEM_HOST_DEVICE(int e)
   {
      constexpr int MD = DofQuadLimits::MAX_D1D;
      constexpr int MQ = DofQuadLimits::MAX_Q1D;
      MFEM_SHARED real_t sB[MQ][MD];
      MFEM_SHARED real_t t1[MD][MQ]; // [dy][qx]
      MFEM_FOREACH_THREAD(d, y, D1D)
      {
         MFEM_FOREACH_THREAD(q, x, Q1D) { sB[q][d] = B(q, d); }
      }
      MFEM_SYNC_THREAD;
      for (int c = 0; c < 2; ++c)
      {
         MFEM_FOREACH_THREAD(dy, y, D1D)
         {
            MFEM_FOREACH_THREAD(qx, x, Q1D)
            {
               real_t s = 0.0;
               for (int dx = 0; dx < D1D; ++dx) { s += sB[qx][dx] * U(dx, dy, c, e); }
               t1[dy][qx] = s;
            }
         }
         MFEM_SYNC_THREAD;
         MFEM_FOREACH_THREAD(qy, y, Q1D)
         {
            MFEM_FOREACH_THREAD(qx, x, Q1D)
            {
               real_t s = 0.0;
               for (int dy = 0; dy < D1D; ++dy) { s += sB[qy][dy] * t1[dy][qx]; }
               const real_t h = 0.5 * s * s;
               if (c == 0) { Y(qx, qy, e) = h; }
               else { Y(qx, qy, e) += h; }
            }
         }
         MFEM_SYNC_THREAD; // t1 is reused by the next component
      }
   });
}

void Eval3D(int ne, int D1D, int Q1D, const Vector& b, const Vector& ue,
            Vector& ye)
{
   const auto B = Reshape(b.Read(), Q1D, D1D);
   const auto U = Reshape(ue.Read(), D1D, D1D, D1D, 3, ne);
   auto Y = Reshape(ye.Write(), Q1D, Q1D, Q1D, ne);
   const int T = (D1D > Q1D) ? D1D : Q1D;
   mfem::forall_2D(ne, T, T, [ = ] MFEM_HOST_DEVICE(int e)
   {
      constexpr int MD = DofQuadLimits::MAX_D1D;
      constexpr int MQ = DofQuadLimits::MAX_Q1D;
      MFEM_SHARED real_t sB[MQ][MD];
      MFEM_SHARED real_t t1[MD][MD][MQ]; // [dz][dy][qx]
      MFEM_SHARED real_t t2[MD][MQ][MQ]; // [dz][qy][qx]
      MFEM_FOREACH_THREAD(d, y, D1D)
      {
         MFEM_FOREACH_THREAD(q, x, Q1D) { sB[q][d] = B(q, d); }
      }
      MFEM_SYNC_THREAD;
      for (int c = 0; c < 3; ++c)
      {
         // x: threads (qx, dy), loop dz, dx.
         MFEM_FOREACH_THREAD(dy, y, D1D)
         {
            MFEM_FOREACH_THREAD(qx, x, Q1D)
            {
               for (int dz = 0; dz < D1D; ++dz)
               {
                  real_t s = 0.0;
                  for (int dx = 0; dx < D1D; ++dx)
                  {
                     s += sB[qx][dx] * U(dx, dy, dz, c, e);
                  }
                  t1[dz][dy][qx] = s;
               }
            }
         }
         MFEM_SYNC_THREAD;
         // y: threads (qx, qy), loop dz, dy.
         MFEM_FOREACH_THREAD(qy, y, Q1D)
         {
            MFEM_FOREACH_THREAD(qx, x, Q1D)
            {
               for (int dz = 0; dz < D1D; ++dz)
               {
                  real_t s = 0.0;
                  for (int dy = 0; dy < D1D; ++dy) { s += sB[qy][dy] * t1[dz][dy][qx]; }
                  t2[dz][qy][qx] = s;
               }
            }
         }
         MFEM_SYNC_THREAD;
         // z: threads (qx, qy), loop qz, dz; accumulate 1/2 u_c^2.
         MFEM_FOREACH_THREAD(qy, y, Q1D)
         {
            MFEM_FOREACH_THREAD(qx, x, Q1D)
            {
               for (int qz = 0; qz < Q1D; ++qz)
               {
                  real_t s = 0.0;
                  for (int dz = 0; dz < D1D; ++dz) { s += sB[qz][dz] * t2[dz][qy][qx]; }
                  const real_t h = 0.5 * s * s;
                  if (c == 0) { Y(qx, qy, qz, e) = h; }
                  else { Y(qx, qy, qz, e) += h; }
               }
            }
         }
         MFEM_SYNC_THREAD; // t1, t2 are reused by the next component
      }
   });
}

} // namespace kinetic_head

KineticHeadInterpolator::KineticHeadInterpolator(
   const ParFiniteElementSpace& vfes, const ParFiniteElementSpace& pfes)
{
   const Mesh& mesh = *vfes.GetMesh();
   dim_ = mesh.Dimension();
   ne_ = vfes.GetNE();
   MFEM_VERIFY(pfes.GetMesh() == vfes.GetMesh(),
               "KineticHeadInterpolator: velocity and pressure must share a mesh");
   MFEM_VERIFY(dim_ == 2 || dim_ == 3, "KineticHeadInterpolator: dim 2 or 3");
   MFEM_VERIFY(vfes.GetVDim() == dim_ && pfes.GetVDim() == 1,
               "KineticHeadInterpolator: vector velocity, scalar pressure");
   const auto* ufe =
      dynamic_cast<const TensorBasisElement*>(vfes.GetTypicalFE());
   const auto* pfe =
      dynamic_cast<const TensorBasisElement*>(pfes.GetTypicalFE());
   MFEM_VERIFY(ufe && pfe && vfes.GetTypicalFE()->GetMapType() ==
               FiniteElement::VALUE, "KineticHeadInterpolator: tensor H1 "
               "(quad/hex) spaces required");
   const int pb = pfe->GetBasisType();
   MFEM_VERIFY(BasisType::GetQuadrature1D(pb) != Quadrature1D::Invalid,
               "KineticHeadInterpolator: the pressure basis must be nodal "
               "(nodal interpolation)");

   // B(q, d) = velocity 1D basis d at the q-th 1D pressure node, both in
   // lexicographic 1D order -- the order of the E-vectors below.
   const int kp = pfes.GetTypicalFE()->GetOrder();
   const real_t* xp = poly1d.GetPoints(kp, pb);
   q1d_ = kp + 1;
   d1d_ = vfes.GetTypicalFE()->GetOrder() + 1;
   B_.SetSize(q1d_ * d1d_);
   Vector shape(d1d_);
   for (int q = 0; q < q1d_; ++q)
   {
      ufe->GetBasis1D().Eval(xp[q], shape);
      for (int d = 0; d < d1d_; ++d) { B_(q + q1d_ * d) = shape(d); }
   }
   B_.UseDevice(true);

   const DeviceDofQuadLimits& lim = DeviceDofQuadLimits::Get();
   device_path_ = (d1d_ <= lim.MAX_D1D && q1d_ <= lim.MAX_Q1D);

   u_restr_ = vfes.GetElementRestriction(ElementDofOrdering::LEXICOGRAPHIC);
   p_restr_ = dynamic_cast<const ElementRestriction*>(
                 pfes.GetElementRestriction(ElementDofOrdering::LEXICOGRAPHIC));
   MFEM_VERIFY(p_restr_, "KineticHeadInterpolator: the pressure space has no "
               "(conforming) ElementRestriction");
   u_e_.SetSize(u_restr_->Height());
   ke_e_.SetSize(p_restr_->Height());
   u_e_.UseDevice(true);
   ke_e_.UseDevice(true);
}

void KineticHeadInterpolator::Interpolate(const ParGridFunction& u,
      ParGridFunction& ke) const
{
   if (!device_path_)
   {
      InterpolateHost(u, ke);
      return;
   }
   u_restr_->Mult(u, u_e_);
   if (dim_ == 2) { kinetic_head::Eval2D(ne_, d1d_, q1d_, B_, u_e_, ke_e_); }
   else { kinetic_head::Eval3D(ne_, d1d_, q1d_, B_, u_e_, ke_e_); }
   p_restr_->MultLeftInverse(ke_e_, ke);
}

void KineticHeadInterpolator::InterpolateHost(const ParGridFunction& u,
      ParGridFunction& ke)
{
   HalfSpeedSquared c(u);
   ke.ProjectCoefficient(c);
}

} // namespace incns
