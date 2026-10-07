#include "precond/rotational_schur.hpp"

#include "util/profiler.hpp"

#include "mfem/general/forall.hpp"

#include <cmath>

// This TU defines device kernels (mfem::forall). A host compiler would build
// them as host loops over device pointers, which segfaults on a GPU;
// src/CMakeLists.txt marks this file LANGUAGE CUDA, and this line turns a lost
// entry there into a compile error.
#if defined(MFEM_USE_CUDA) && !defined(__CUDACC__)
#error "rotational_schur.cpp defines device kernels and must be compiled by nvcc -- see the nvcc TU list in src/CMakeLists.txt"
#endif

namespace incns
{

using namespace mfem;

// Kernels in a NAMED namespace: nvcc rejects extended lambdas whose enclosing
// function has internal linkage.
namespace rotational_schur_kernels
{

// Q(:, :, q, e) = T (or T^T) from the physical derivatives D(q, c, d, e) =
// dw_c/dx_d (QVectorLayout::byNODES).
void FillTensor(int dim, int nq, int ne, real_t sigma, real_t alpha,
                real_t scale, bool transpose, const Vector& der,
                QuadratureFunction& qf)
{
   const real_t s = sigma, a = alpha, f = scale;
   const bool tr = transpose;
   if (dim == 3)
   {
      const auto D = Reshape(der.Read(), nq, 3, 3, ne);
      auto Q = Reshape(qf.Write(), 3, 3, nq, ne);
      mfem::forall(nq * ne, [ = ] MFEM_HOST_DEVICE(int i)
      {
         const int q = i % nq, e = i / nq;
         const real_t o0 = a * (D(q, 2, 1, e) - D(q, 1, 2, e));
         const real_t o1 = a * (D(q, 0, 2, e) - D(q, 2, 0, e));
         const real_t o2 = a * (D(q, 1, 0, e) - D(q, 0, 1, e));
         real_t K[9];
         RotatingDarcyTensor::Fill3(s, o0, o1, o2, K);
         for (int c = 0; c < 3; ++c)
         {
            for (int r = 0; r < 3; ++r)
            {
               Q(r, c, q, e) = f * (tr ? K[c + 3 * r] : K[r + 3 * c]);
            }
         }
      });
   }
   else
   {
      const auto D = Reshape(der.Read(), nq, 2, 2, ne);
      auto Q = Reshape(qf.Write(), 2, 2, nq, ne);
      mfem::forall(nq * ne, [ = ] MFEM_HOST_DEVICE(int i)
      {
         const int q = i % nq, e = i / nq;
         const real_t o = a * (D(q, 1, 0, e) - D(q, 0, 1, e));
         real_t K[4];
         RotatingDarcyTensor::Fill2(s, o, K);
         for (int c = 0; c < 2; ++c)
         {
            for (int r = 0; r < 2; ++r)
            {
               Q(r, c, q, e) = f * (tr ? K[c + 2 * r] : K[r + 2 * c]);
            }
         }
      });
   }
}

// Per point: mu = |alpha omega| / sigma, the point volume w_q det J, and that
// volume again where mu > threshold (else 0).
void PointStats(int dim, int nq, int ne, real_t alpha, real_t sigma,
                real_t threshold, const Vector& der, const Array<real_t>& w,
                const Vector& detj, Vector& mu, Vector& vol, Vector& all)
{
   const real_t a = alpha, inv_s = 1.0 / sigma, thr = threshold;
   const auto W = w.Read();
   const auto J = Reshape(detj.Read(), nq, ne);
   auto M = Reshape(mu.Write(), nq, ne);
   auto V = Reshape(vol.Write(), nq, ne);
   auto A = Reshape(all.Write(), nq, ne);
   if (dim == 3)
   {
      const auto D = Reshape(der.Read(), nq, 3, 3, ne);
      mfem::forall(nq * ne, [ = ] MFEM_HOST_DEVICE(int i)
      {
         const int q = i % nq, e = i / nq;
         const real_t o0 = D(q, 2, 1, e) - D(q, 1, 2, e);
         const real_t o1 = D(q, 0, 2, e) - D(q, 2, 0, e);
         const real_t o2 = D(q, 1, 0, e) - D(q, 0, 1, e);
         const real_t m = fabs(a) * sqrt(o0 * o0 + o1 * o1 + o2 * o2) * inv_s;
         const real_t v = W[q] * fabs(J(q, e));
         M(q, e) = m;
         A(q, e) = v;
         V(q, e) = (m > thr) ? v : 0.0;
      });
   }
   else
   {
      const auto D = Reshape(der.Read(), nq, 2, 2, ne);
      mfem::forall(nq * ne, [ = ] MFEM_HOST_DEVICE(int i)
      {
         const int q = i % nq, e = i / nq;
         const real_t m = fabs(a * (D(q, 1, 0, e) - D(q, 0, 1, e))) * inv_s;
         const real_t v = W[q] * fabs(J(q, e));
         M(q, e) = m;
         A(q, e) = v;
         V(q, e) = (m > thr) ? v : 0.0;
      });
   }
}

} // namespace rotational_schur_kernels

// ---------------------------------------------------------------------------
// RotationNumber

RotationNumber::RotationNumber(const ParFiniteElementSpace& vfes,
                               const IntegrationRule& ir)
   : vfes_(vfes), ir_(ir)
{
   const int dim = vfes.GetParMesh()->Dimension();
   MFEM_VERIFY(vfes.GetVDim() == dim, "rotation_number: velocity vdim must "
               "equal the dimension");
   restr_ = vfes.GetElementRestriction(ElementDofOrdering::LEXICOGRAPHIC);
   qi_ = vfes.GetQuadratureInterpolator(ir_);
   geom_ = vfes.GetParMesh()->GetGeometricFactors(
              ir_, GeometricFactors::DETERMINANTS, Device::GetDeviceMemoryType());
   const int ne = vfes.GetNE(), nq = ir_.GetNPoints();
   we_.SetSize(restr_->Height());
   der_.SetSize(nq * dim * dim * ne);
   mu_.SetSize(nq * ne);
   vol_.SetSize(nq * ne);
   all_.SetSize(nq * ne);
   for (Vector* v : {&we_, &der_, &mu_, &vol_, &all_}) { v->UseDevice(true); }
}

RotationNumberStats RotationNumber::Compute(const ParGridFunction& w,
      double alpha, double sigma,
      double threshold) const
{
   MFEM_VERIFY(w.ParFESpace() == &vfes_, "rotation_number: w must live on the "
               "bound space");
   MFEM_VERIFY(sigma > 0.0, "rotation_number: sigma must be positive");
   const int dim = vfes_.GetParMesh()->Dimension();
   const int ne = vfes_.GetNE(), nq = ir_.GetNPoints();
   double loc[3] = {0.0, 0.0, 0.0}; // max mu; volume above; total volume
   if (ne > 0)
   {
      restr_->Mult(w, we_);
      qi_->SetOutputLayout(QVectorLayout::byNODES); // shared object: set it
      qi_->PhysDerivatives(we_, der_);
      rotational_schur_kernels::PointStats(dim, nq, ne, alpha, sigma, threshold,
                                           der_, ir_.GetWeights(),
                                           geom_->detJ, mu_, vol_, all_);
      loc[0] = mu_.Max(); // device-aware reductions
      loc[1] = vol_.Sum();
      loc[2] = all_.Sum();
   }
   double mx = 0.0, sums[2] = {loc[1], loc[2]};
   MPI_Allreduce(&loc[0], &mx, 1, MPI_DOUBLE, MPI_MAX, vfes_.GetComm());
   MPI_Allreduce(MPI_IN_PLACE, sums, 2, MPI_DOUBLE, MPI_SUM, vfes_.GetComm());
   RotationNumberStats s;
   s.max_mu = mx;
   s.vol_fraction = (sums[1] > 0.0) ? sums[0] / sums[1] : 0.0;
   s.threshold = threshold;
   return s;
}

// ---------------------------------------------------------------------------
// RotatingDarcyTensor

RotatingDarcyTensor::RotatingDarcyTensor(const GridFunction& w_star,
      real_t sigma, real_t alpha)
   : MatrixCoefficient(w_star.FESpace()->GetMesh()->Dimension()), w_(w_star),
     sigma_(sigma), alpha_(alpha)
{
   const int dim = w_star.FESpace()->GetMesh()->Dimension();
   MFEM_VERIFY(dim == 2 || dim == 3, "rotating_darcy: 2D or 3D only");
   MFEM_VERIFY(w_star.FESpace()->GetVDim() == dim, "rotating_darcy: w* must "
               "be a vector field with vdim = dim");
}

void RotatingDarcyTensor::Eval(DenseMatrix& T, ElementTransformation& Tr,
                               const IntegrationPoint& ip)
{
   Tr.SetIntPoint(&ip);
   Vector o;
   w_.GetCurl(Tr, o); // host; legacy assembly (tests) only
   const int dim = GetHeight();
   T.SetSize(dim);
   if (dim == 3)
   {
      real_t K[9];
      Fill3(sigma_, alpha_ * o(0), alpha_ * o(1), alpha_ * o(2), K);
      for (int k = 0; k < 9; ++k) { T.GetData()[k] = scale_ * K[k]; }
   }
   else
   {
      real_t K[4];
      Fill2(sigma_, alpha_ * o(0), K);
      for (int k = 0; k < 4; ++k) { T.GetData()[k] = scale_ * K[k]; }
   }
}

void RotatingDarcyTensor::Project(QuadratureFunction& qf, bool transpose)
{
   const int dim = GetHeight();
   MFEM_VERIFY(qf.GetVDim() == dim * dim, "rotating_darcy: expected vdim "
               << dim * dim);
   const FiniteElementSpace& fes = *w_.FESpace();
   // The rule of the form being assembled (QuadratureSpace stores the
   // pointer). It must outlive fes's interpolator cache (spec 5.7, item 5):
   // RuleBook and IntRules rules do.
   const IntegrationRule& ir = qf.GetSpace()->GetIntRule(0);
   const int ne = fes.GetNE(), nq = ir.GetNPoints();
   if (ne == 0) { return; }
   const Operator* R = fes.GetElementRestriction(
                          ElementDofOrdering::LEXICOGRAPHIC);
   w_e_.SetSize(R->Height());
   w_e_.UseDevice(true);
   der_.SetSize(nq * dim * dim * ne);
   der_.UseDevice(true);
   R->Mult(w_, w_e_);
   const QuadratureInterpolator* qi = fes.GetQuadratureInterpolator(ir);
   qi->SetOutputLayout(QVectorLayout::byNODES); // D(q, c, d, e) = dw_c/dx_d
   qi->PhysDerivatives(w_e_, der_);
   rotational_schur_kernels::FillTensor(dim, nq, ne, sigma_, alpha_, scale_,
                                        transpose, der_, qf);
}

// ---------------------------------------------------------------------------
// RotationalSchurPreconditioner

RotationalSchurPreconditioner::RotationalSchurPreconditioner(
   ParFiniteElementSpace& pfes, const Array<int>& p_ess_tdofs,
   const IntegrationRule& ir, const GridFunction& w_star, real_t alpha,
   real_t nu, Solver& cc, Solver& mass_inv, Solver& lap_inv,
   const Options& opt)
   : Solver(pfes.GetTrueVSize()), opt_(opt), ess_(p_ess_tdofs), nu_(nu),
     cc_(cc), mass_inv_(mass_inv), lap_(lap_inv),
     T_coeff_(w_star, 1.0, alpha), comm_(pfes.GetComm())
{
   iterative_mode = false;
   MFEM_VERIFY(w_star.FESpace()->GetMesh() == pfes.GetMesh(),
               "rotational_schur: w* must live on the pressure space's mesh");
   MFEM_VERIFY(opt.mu_off <= opt.mu_on, "rotational_schur: mu_off must not "
               "exceed mu_on");
   MFEM_VERIFY(opt.vol_off <= opt.vol_on, "rotational_schur: vol_off must not "
               "exceed vol_on");
   MFEM_VERIFY(opt.inner_iterations >= 1, "rotational_schur: inner_iterations "
               "must be >= 1");
   MFEM_VERIFY(cc.Height() == height && mass_inv.Height() == height &&
               lap_inv.Height() == height, "rotational_schur: the CC, mass and "
               "Laplacian solvers must act on the pressure true dofs");

   T_form_ = std::make_unique<ParBilinearForm>(&pfes);
   T_form_->SetAssemblyLevel(AssemblyLevel::PARTIAL);
   auto* di = new DiffusionIntegrator(T_coeff_);
   di->SetIntRule(&ir);
   T_form_->AddDomainIntegrator(di);

   inner_ = std::make_unique<FGMRESSolver>(comm_);
   inner_->SetKDim(opt.inner_iterations);
   inner_->SetMaxIter(opt.inner_iterations);
   inner_->SetRelTol(1e-12); // effectively a fixed count; a breakdown guard
   inner_->SetAbsTol(0.0);
   inner_->SetPrintLevel(IterativeSolver::PrintLevel().None());
   inner_->iterative_mode = false;

   t_.SetSize(height);
   t_.UseDevice(true);
   r0_.SetSize(height);
   r0_.UseDevice(true);
   tensor_active_ = (opt.mode == Mode::Tensor);
}

void RotationalSchurPreconditioner::Update(real_t sigma,
      const RotationNumberStats& stats)
{
   MFEM_VERIFY(sigma > 0.0, "rotational_schur: sigma must be positive");
   bool want = tensor_active_;
   switch (opt_.mode)
   {
      case Mode::CahouetChabard: want = false; break;
      case Mode::Tensor: want = true; break;
      case Mode::Auto:
         if (opt_.criterion == Criterion::MaxMu)
         {
            if (!tensor_active_ && stats.max_mu > opt_.mu_on) { want = true; }
            if (tensor_active_ && stats.max_mu < opt_.mu_off) { want = false; }
         }
         else
         {
            MFEM_VERIFY(stats.threshold == opt_.mu_on, "rotational_schur: "
                        "VolumeFraction needs stats computed with threshold "
                        "mu_on");
            if (!tensor_active_ && stats.vol_fraction > opt_.vol_on)
            {
               want = true;
            }
            if (tensor_active_ && stats.vol_fraction < opt_.vol_off)
            {
               want = false;
            }
         }
         break;
   }
   if (want != tensor_active_)
   {
      ++n_switches_;
      tensor_active_ = want;
   }
   if (tensor_active_) { AssembleTensor(sigma); }
}

void RotationalSchurPreconditioner::AssembleTensor(real_t sigma)
{
   INCNS_PROFILE("rotational_schur::assemble_tensor");
   // The form is sigma L_T = -div(sigma T grad): exactly L_p at omega = 0,
   // so the unscaled Laplacian solver is its natural preconditioner.
   sigma_ = sigma;
   T_coeff_.SetSigma(sigma);
   T_coeff_.SetScale(sigma);
   T_form_->Assemble(); // re-runs the partial-assembly setup in place
   if (!T_op_.Ptr())    // first activation only
   {
      T_form_->FormSystemMatrix(ess_, T_op_);
      inner_->SetOperator(*T_op_); // before SetPreconditioner (spec 5.7, 3)
      inner_->SetPreconditioner(lap_);
   }
}

void RotationalSchurPreconditioner::Mult(const Vector& r, Vector& z) const
{
   if (!tensor_active_)
   {
      cc_.Mult(r, z); // the unchanged Cahouet-Chabard path
      return;
   }
   INCNS_PROFILE("rotational_schur::tensor_apply");
   const Vector* rhs = &r;
   if (opt_.remove_mean)
   {
      r0_ = r;
      RemoveMean(r0_);
      rhs = &r0_;
   }
   inner_->Mult(*rhs, z); // k FGMRES iterations on sigma L_T
   if (opt_.remove_mean) { RemoveMean(z); }
   z *= sigma_;           // (sigma L_T)^-1 -> L_T^-1
   mass_inv_.Mult(r, t_);
   z.Add(nu_, t_);
}

void RotationalSchurPreconditioner::RemoveMean(Vector& v) const
{
   double loc[2] = {v.Sum(), static_cast<double>(v.Size())}; // device Sum
   MPI_Allreduce(MPI_IN_PLACE, loc, 2, MPI_DOUBLE, MPI_SUM, comm_);
   v -= loc[0] / loc[1];
}

} // namespace incns
