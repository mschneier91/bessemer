/**
 * @file oseen_wind.cpp
 * @brief OseenWindData and the quadrature-function coefficients.
 */
#include "oseen/oseen_wind.hpp"

// This TU defines device kernels (mfem::forall). A host compiler would build
// them as host loops over device pointers, which segfaults on a GPU;
// src/CMakeLists.txt marks this file LANGUAGE CUDA, and this line turns a lost
// entry there into a compile error.
#if defined(MFEM_USE_CUDA) && !defined(__CUDACC__)
#error "oseen_wind.cpp defines device kernels and must be compiled by nvcc -- see the nvcc TU list in src/CMakeLists.txt"
#endif

#include "operators/kernel_specializations.hpp"

#include "mfem/fem/qinterp/eval.hpp"
#include "mfem/fem/qinterp/grad.hpp"

namespace incns
{

using namespace mfem;

namespace
{

Geometry::Type TensorGeometry(int dim)
{
   return (dim == 3) ? Geometry::CUBE : Geometry::SQUARE;
}

// The wind's values (byVDIM, vdim = dim) at D1D dofs and Q1D points per
// direction.
template <int DIM, int D1D, int Q1D>
void AddWindInterpolation()
{
   using QI = QuadratureInterpolator;
   constexpr auto byVDIM = QVectorLayout::byVDIM;
   QI::AddTensorEvalSpecializations<DIM, byVDIM, DIM, D1D, Q1D>();
}

// MFEM's quadrature interpolator holds 8 points per direction in its
// runtime-size kernels on a GPU (DofQuadLimits::MAX_INTERP_1D), dofs and
// points alike, unchecked: the wind at the convection rule k + 2m (10 points
// per direction at Q6, 9 for PCD's m = k - 1) wrote past its shared arrays,
// and every Q6 oseen_case run stopped with an illegal address in its first
// step (2026-10-20). Compile-time kernels for the wind's values at the
// sizes the rules reach for velocity orders 6 and 7 (7 and 8 dofs; the OSS
// rule 2m + 2 has 9 points at Q7); not 11 points in 3D (MFEM's kernel
// launches 11^3 threads per element, more than a block holds). Its
// derivatives go through the mesh Jacobians' kernels
// (AddInterpolationKernels; see Update). Registered by the Oseen code only,
// so nothing else dispatches differently. Process-wide and idempotent.
void AddWindInterpolationKernels()
{
   static const bool added = []
   {
      AddInterpolationKernels();
      AddWindInterpolation<2, 7, 9>();
      AddWindInterpolation<2, 7, 10>();
      AddWindInterpolation<2, 8, 9>();
      AddWindInterpolation<2, 8, 10>();
      AddWindInterpolation<2, 8, 11>();
      AddWindInterpolation<3, 7, 9>();
      AddWindInterpolation<3, 7, 10>();
      AddWindInterpolation<3, 8, 9>();
      AddWindInterpolation<3, 8, 10>();
      return true;
   }();
   (void)added;
}

// J^-1 of a column-major D x D Jacobian (D = 2, 3): adj(J) / det J.
MFEM_HOST_DEVICE inline void InvertJacobian(int D, const real_t* Jm, real_t* Ji)
{
   if (D == 2)
   {
      const real_t det = Jm[0] * Jm[3] - Jm[1] * Jm[2];
      Ji[0] = Jm[3] / det; Ji[1] = -Jm[1] / det;
      Ji[2] = -Jm[2] / det; Ji[3] = Jm[0] / det;
      return;
   }
   // Column-major J: a(i, j) = Jm[i + 3 j].
   auto a = [&](int i, int j) { return Jm[i + 3 * j]; };
   const real_t c00 = a(1, 1) * a(2, 2) - a(1, 2) * a(2, 1);
   const real_t c01 = a(1, 2) * a(2, 0) - a(1, 0) * a(2, 2);
   const real_t c02 = a(1, 0) * a(2, 1) - a(1, 1) * a(2, 0);
   const real_t det = a(0, 0) * c00 + a(0, 1) * c01 + a(0, 2) * c02;
   Ji[0 + 3 * 0] = c00 / det;
   Ji[0 + 3 * 1] = (a(0, 2) * a(2, 1) - a(0, 1) * a(2, 2)) / det;
   Ji[0 + 3 * 2] = (a(0, 1) * a(1, 2) - a(0, 2) * a(1, 1)) / det;
   Ji[1 + 3 * 0] = c01 / det;
   Ji[1 + 3 * 1] = (a(0, 0) * a(2, 2) - a(0, 2) * a(2, 0)) / det;
   Ji[1 + 3 * 2] = (a(0, 2) * a(1, 0) - a(0, 0) * a(1, 2)) / det;
   Ji[2 + 3 * 0] = c02 / det;
   Ji[2 + 3 * 1] = (a(0, 1) * a(2, 0) - a(0, 0) * a(2, 1)) / det;
   Ji[2 + 3 * 2] = (a(0, 0) * a(1, 1) - a(0, 1) * a(1, 0)) / det;
}

} // namespace

real_t QFScalarCoefficient::Eval(ElementTransformation& T,
                                 const IntegrationPoint& ip)
{
   q_.HostRead();
   Vector v;
   q_.GetValues(T.ElementNo, ip.index, v);
   return v(0);
}

void QFScalarCoefficient::Project(QuadratureFunction& qf)
{
   MFEM_VERIFY(qf.Size() == q_.Size(), "QFScalarCoefficient: another rule");
   qf = q_;
}

void QFVectorCoefficient::Eval(Vector& V, ElementTransformation& T,
                               const IntegrationPoint& ip)
{
   q_.HostRead();
   q_.GetValues(T.ElementNo, ip.index, V);
}

void QFVectorCoefficient::Project(QuadratureFunction& qf)
{
   MFEM_VERIFY(qf.Size() == q_.Size(), "QFVectorCoefficient: another rule");
   qf = q_;
}

void QFMatrixCoefficient::Eval(DenseMatrix& K, ElementTransformation& T,
                               const IntegrationPoint& ip)
{
   q_.HostRead();
   Vector v;
   q_.GetValues(T.ElementNo, ip.index, v);
   K.SetSize(height);
   for (int j = 0; j < height; j++)
   {
      for (int i = 0; i < height; i++) { K(i, j) = v(i + height * j); }
   }
}

void QFMatrixCoefficient::Project(QuadratureFunction& qf, bool /*transpose*/)
{
   MFEM_VERIFY(qf.Size() == q_.Size(), "QFMatrixCoefficient: another rule");
   qf = q_;
}

OseenWindData::OseenWindData(ParFiniteElementSpace& vfes, const RuleBook& rules,
                             double nu, int space_order)
   : vfes_(vfes), nu_(nu), dim_(vfes.GetMesh()->Dimension()),
     k_(vfes.GetMaxElementOrder()),
     m_(space_order > 0 ? space_order : k_),
     ir_conv_(&rules.Get(TensorGeometry(dim_), k_ + 2 * m_)),
     ir_oss_(&rules.Get(TensorGeometry(dim_), 2 * m_ + 2)),
     ir_gll_(&rules.CollocatedMass(TensorGeometry(dim_), m_)),
     ir_center_(&rules.Get(TensorGeometry(dim_), 1)),
     qs_conv_(*vfes.GetMesh(), *ir_conv_), qs_oss_(*vfes.GetMesh(), *ir_oss_),
     qs_gll_(*vfes.GetMesh(), *ir_gll_), qs_center_(*vfes.GetMesh(), *ir_center_),
     wind_conv_(&qs_conv_, dim_), half_divw_(&qs_conv_, 1),
     wind_oss_(&qs_oss_, dim_), tau_ww_(&qs_oss_, dim_ * dim_),
     tau_w_(&qs_oss_, dim_), tau_gll_(&qs_gll_, 1),
     wind_center_(&qs_center_, dim_),
     wind_c_(wind_conv_), half_divw_c_(half_divw_),
     tau_ww_c_(tau_ww_, dim_), tau_w_c_(tau_w_), tau_gll_c_(tau_gll_)
{
   MFEM_VERIFY(vfes.GetVDim() == dim_ && vfes.GetOrdering() == Ordering::byNODES,
               "OseenWindData: a vector velocity space ordered by nodes");
   MFEM_VERIFY(ir_center_->GetNPoints() == 1, "OseenWindData: one centre point");
   AddWindInterpolationKernels(); // before the first interpolation
   for (QuadratureFunction* q :
        {
           &wind_conv_, &half_divw_, &wind_oss_, &tau_ww_, &tau_w_, &tau_gll_,
           &wind_center_
        })
   {
      q->UseDevice(true);
   }
   tau_c_.SetSize(vfes.GetMesh()->GetNE());
   tau_c_.UseDevice(true);
   e_vec_.UseDevice(true);
   q_der_.UseDevice(true);
}

void OseenWindData::Update(const ParGridFunction& wind)
{
   ParMesh& mesh = *vfes_.GetParMesh();
   const int NE = mesh.GetNE();
   const int D = dim_;
   const int k = k_;
   const real_t nu = nu_;

   wind_conv_.ProjectGridFunction(wind);
   wind_oss_.ProjectGridFunction(wind);
   wind_center_.ProjectGridFunction(wind);

   // 1/2 div w at the convection rule: the wind's reference derivatives
   // (byNODES: point, component, direction, element -- the layout and the
   // compile-time kernels of the mesh Jacobians, AddInterpolationKernels)
   // contracted with J^-1. MFEM's physical-derivative kernel at 10 points
   // per direction launches 10^3 threads per element and asks for more
   // registers than a block has (job 49531904).
   const Operator* R = vfes_.GetElementRestriction(GetEVectorOrdering(vfes_));
   e_vec_.SetSize(R->Height());
   R->Mult(wind, e_vec_);
   const QuadratureInterpolator* qi = vfes_.GetQuadratureInterpolator(*ir_conv_);
   qi->SetOutputLayout(QVectorLayout::byNODES);
   const int NQc = ir_conv_->GetNPoints();
   q_der_.SetSize(NQc * D * D * NE);
   qi->Derivatives(e_vec_, q_der_);
   if (NE > 0)
   {
      const GeometricFactors* gc =
         mesh.GetGeometricFactors(*ir_conv_, GeometricFactors::JACOBIANS);
      const auto der = Reshape(q_der_.Read(), NQc, D, D, NE);
      const auto J = Reshape(gc->J.Read(), NQc, D, D, NE);
      auto hd = Reshape(half_divw_.Write(), NQc, NE);
      mfem::forall(NQc * NE, [ = ] MFEM_HOST_DEVICE(int i)
      {
         const int q = i % NQc, e = i / NQc;
         real_t Jm[9] = {}, Ji[9] = {};
         for (int j = 0; j < D; j++)
         {
            for (int a = 0; a < D; a++) { Jm[a + D * j] = J(q, a, j, e); }
         }
         InvertJacobian(D, Jm, Ji);
         // d w_c / d x_c = sum_r (d w_c / d xi_r) (J^-1)_{r c}.
         real_t s = 0.0;
         for (int c = 0; c < D; c++)
         {
            for (int r = 0; r < D; r++) { s += der(q, c, r, e) * Ji[r + D * c]; }
         }
         hd(q, e) = 0.5 * s;
      });
   }

   // tau_c per element: the wind and J^{-1} at the centre.
   if (NE > 0)
   {
      const GeometricFactors* gf =
         mesh.GetGeometricFactors(*ir_center_, GeometricFactors::JACOBIANS);
      const auto J = Reshape(gf->J.Read(), 1, D, D, NE);
      const auto W = Reshape(wind_center_.Read(), D, 1, NE);
      auto tc = tau_c_.Write();
      mfem::forall(NE, [ = ] MFEM_HOST_DEVICE(int e)
      {
         real_t Jm[9] = {}, Ji[9] = {}, w[3] = {};
         for (int j = 0; j < D; j++)
         {
            for (int i = 0; i < D; i++) { Jm[i + D * j] = J(0, i, j, e); }
         }
         InvertJacobian(D, Jm, Ji);
         for (int d = 0; d < D; d++) { w[d] = W(d, 0, e); }
         tc[e] = OssTau(D, w, Ji, nu, k);
      });
   }

   // tau_c w w^T and tau_c w at the OSS rule, tau_c at the GLL rule.
   const int NQo = ir_oss_->GetNPoints();
   const int NQg = ir_gll_->GetNPoints();
   {
      const auto tc = tau_c_.Read();
      const auto W = Reshape(wind_oss_.Read(), D, NQo, NE);
      auto tww = Reshape(tau_ww_.Write(), D * D, NQo, NE);
      auto tw = Reshape(tau_w_.Write(), D, NQo, NE);
      mfem::forall(NQo * NE, [ = ] MFEM_HOST_DEVICE(int i)
      {
         const int q = i % NQo, e = i / NQo;
         for (int a = 0; a < D; a++)
         {
            tw(a, q, e) = tc[e] * W(a, q, e);
            for (int b = 0; b < D; b++)
            {
               tww(a + D * b, q, e) = tc[e] * W(a, q, e) * W(b, q, e);
            }
         }
      });
      auto tg = Reshape(tau_gll_.Write(), NQg, NE);
      mfem::forall(NQg * NE, [ = ] MFEM_HOST_DEVICE(int i)
      {
         tg(i % NQg, i / NQg) = tc[i / NQg];
      });
   }
}

} // namespace incns
