#include "precond/cahouet_chabard.hpp"

#include "util/profiler.hpp"

#include <stdexcept>

namespace incns
{

using namespace mfem;

void CahouetChabardConfig::Validate(bool root) const
{
   // Hard errors: inconsistent physics or not-in-v1 settings. Thrown
   // identically on every rank (pure config, no mesh data involved).
   if (sigma < 0.0)
   {
      throw std::invalid_argument("cc config: sigma must be >= 0");
   }
   if (nu <= 0.0)
   {
      throw std::invalid_argument("cc config: nu must be > 0");
   }
   if (schur_model == SchurModel::LumpedBMB)
   {
      throw std::invalid_argument(
         "cc config: LumpedBMB is rejected in v1 -- with the collocated GLL "
         "basis the consistent M_v is already diagonal, and the |phi| lumping "
         "it would need is a simplex fallback with no callers here; use "
         "ConsistentBMB");
   }
   if (visc_form == ViscousForm::SymGradient)
   {
      throw std::invalid_argument(
         "cc config: SymGradient is not in the system operator yet "
         "(bessemer assembles the Laplacian form); lands with the NSE stage");
   }
   if (pc_mass_coeff == PcMassCoeff::ReciprocalNuField)
   {
      throw std::invalid_argument(
         "cc config: ReciprocalNuField (variable-viscosity LES mass) is a v2 "
         "hook; only ConstNu is implemented");
   }
   if (lp_pc == LpPC::PMG)
   {
      throw std::invalid_argument(
         "cc config: lp_pc = PMG (p-multigrid) is deferred to the GPU stage "
         "(human decision 2026-07-16); use LORAMG");
   }
   if (a_pc == APC::PMGChebyshev)
   {
      throw std::invalid_argument(
         "cc config: a_pc = PMGChebyshev (p-multigrid) is deferred to the GPU "
         "stage (human decision 2026-07-16); use LORAMG, JacobiChebyshev or "
         "JacobiPCG");
   }
   if (pc_precision == PcPrecision::FP32PC)
   {
      throw std::invalid_argument(
         "cc config: FP32PC is a reserved v2 hook; only FP64 is implemented");
   }
   if (inner_stop == InnerStop::FixedIters && n_inner < 1)
   {
      throw std::invalid_argument("cc config: n_inner must be >= 1");
   }
   if (inner_stop == InnerStop::RelTol &&
       (tol_inner <= 0.0 || tol_inner >= 1.0))
   {
      throw std::invalid_argument("cc config: tol_inner must be in (0, 1)");
   }
   if (a_pc == APC::JacobiPCG &&
       (a_pcg_rtol <= 0.0 || a_pcg_rtol >= 1.0 || a_pcg_max_iter < 1))
   {
      throw std::invalid_argument("cc config: JacobiPCG needs a_pcg_rtol in "
                                  "(0, 1) and a_pcg_max_iter >= 1");
   }
   if (lp_vcycles < 1 || a_vcycles < 1)
   {
      throw std::invalid_argument("cc config: vcycle counts must be >= 1");
   }
   if (k_reproj < 1)
   {
      throw std::invalid_argument("cc config: k_reproj must be >= 1");
   }
   if (pc_quadrature == PcQuadrature::GllCollocated &&
       (mv_inv == MassInvType::Chebyshev || mp_inv == MassInvType::Chebyshev ||
        mv_inv == MassInvType::AbsLumped || mp_inv == MassInvType::AbsLumped))
   {
      throw std::invalid_argument(
         "cc config: pc_quadrature = gll_collocated makes the PC masses "
         "exactly diagonal; mv_inv/mp_inv must be Auto or DiagDirect");
   }

   // Loud warnings (rank 0), not errors.
   if (root && schur_model == SchurModel::LaplacianLegacy)
   {
      mfem::out << "[cc] WARNING: LaplacianLegacy Schur model -- comparison "
                "mode, known non-robust on fine meshes (SPEC T5); never "
                "the production default." << std::endl;
   }
}

CahouetChabardSchurPC::CahouetChabardSchurPC(
   const CahouetChabardConfig& cfg, ParFiniteElementSpace& vfes,
   ParFiniteElementSpace& pfes,
   const RuleBook& rules, const Operator& B, const Operator& Mv,
   const Vector& mv_diag, const Array<int>& outflow_attrs, bool singular_auto)
   : Solver(pfes.GetTrueVSize()), cfg_(cfg), mp_form_(&pfes)
{
   INCNS_PROFILE("cc_schur::setup");
   iterative_mode = false;

   int rank = 0;
   MPI_Comm_rank(pfes.GetComm(), &rank);
   cfg_.Validate(rank == 0);

   // v1 is nodal H1 pressure only (SPEC par.12): the projector's all-ones
   // constant vector and the LOR L_p both assume it.
   if (!dynamic_cast<const H1_FECollection*>(pfes.FEColl()))
   {
      throw std::invalid_argument(
         "cc: L2/DG pressure spaces are not supported in v1 (SPEC par.12)");
   }

   switch (cfg_.nullspace)
   {
      case NullspaceMode::Auto:     singular_ = singular_auto; break;
      case NullspaceMode::ForceOn:  singular_ = true; break;
      case NullspaceMode::ForceOff: singular_ = false; break;
   }
   sigma_ = cfg_.sigma;
   nu_pc_ = cfg_.nu_pc < 0.0 ? cfg_.nu : cfg_.nu_pc;

   const bool colloc = (cfg_.pc_quadrature == PcQuadrature::GllCollocated);

   // --- pressure mass + its fixed inverse ------------------------------------
   // Inherit: the standard GL rule (matches the system's M_p). GllCollocated:
   // the collocated GLL rule -> exactly diagonal -> DiagDirect is REQUIRED
   // (a non-collocated basis fails loudly instead of silently degrading).
   {
      const int kp = pfes.FEColl()->GetOrder();
      const Geometry::Type geom = (pfes.GetParMesh()->Dimension() == 3)
                                  ? Geometry::CUBE : Geometry::SQUARE;
      auto* mi = new MassIntegrator;
      mi->SetIntRule(colloc ? &rules.CollocatedMass(geom, kp)
                     : &rules.Get(geom, 2 * kp));
      mp_form_.AddDomainIntegrator(mi);
      mp_form_.SetAssemblyLevel(AssemblyLevel::PARTIAL);
      mp_form_.Assemble();
      Array<int> empty;
      mp_form_.FormSystemMatrix(empty, Mp_);
      mp_diag_.SetSize(pfes.GetTrueVSize());
      mp_form_.AssembleDiagonal(mp_diag_);
      mp_inv_ = std::make_unique<MassInverse>(
                   *Mp_.Ptr(), mp_diag_, pfes.GetComm(),
                   colloc ? MassInvType::DiagDirect : cfg_.mp_inv,
                   cfg_.k_mp_chebyshev);
   }

   // --- the consistent mixed Poisson stack ------------------------------------
   // Inherit: the SYSTEM's velocity mass (spectrally faithful to A; Chebyshev
   // when consistent). GllCollocated: the PC builds its OWN collocated-GLL
   // vector mass (diagonal -> one fused multiply per inner-CG iteration); the
   // system operator is untouched -- only the preconditioner cheapens.
   if (colloc)
   {
      const int ku = vfes.FEColl()->GetOrder();
      const Geometry::Type geom = (vfes.GetParMesh()->Dimension() == 3)
                                  ? Geometry::CUBE : Geometry::SQUARE;
      mv_form_pc_ = std::make_unique<ParBilinearForm>(&vfes);
      auto* mvi = new VectorMassIntegrator;
      mvi->SetIntRule(&rules.CollocatedMass(geom, ku));
      mv_form_pc_->AddDomainIntegrator(mvi);
      mv_form_pc_->SetAssemblyLevel(AssemblyLevel::PARTIAL);
      mv_form_pc_->Assemble();
      Array<int> empty;
      mv_form_pc_->FormSystemMatrix(empty, Mv_pc_);
      mv_diag_pc_.SetSize(vfes.GetTrueVSize());
      mv_form_pc_->AssembleDiagonal(mv_diag_pc_);
      mv_inv_ = std::make_unique<MassInverse>(*Mv_pc_.Ptr(), mv_diag_pc_,
                                              pfes.GetComm(),
                                              MassInvType::DiagDirect);
   }
   else
   {
      mv_inv_ = std::make_unique<MassInverse>(Mv, mv_diag, pfes.GetComm(),
                                              cfg_.mv_inv, cfg_.k_mv_chebyshev);
   }
   bmb_ = std::make_unique<MixedPoissonOperator>(B, *mv_inv_);
   proj_ = std::make_unique<ConstantPressureProjector>(pfes);
   if (singular_)
   {
      bmb_proj_ = std::make_unique<ProjectedOperator>(*bmb_, *proj_);
   }
   lp_ = std::make_unique<LpSurrogate>(pfes, outflow_attrs, singular_,
                                       cfg_.lp_bc, cfg_.lp_dirichlet_attrs,
                                       cfg_.lp_vcycles);

   // --- the inner CG: SPD operator + symmetric PC => CG, never GMRES ----------
   inner_cg_ = std::make_unique<CGSolver>(pfes.GetComm());
   inner_cg_->SetOperator(singular_ ? static_cast<Operator&>(*bmb_proj_)
                          : static_cast<Operator&>(*bmb_));
   inner_cg_->SetPreconditioner(*lp_);
   inner_cg_->iterative_mode = false;
   inner_cg_->SetPrintLevel(-1);
   if (cfg_.inner_stop == InnerStop::FixedIters)
   {
      // Fixed iteration count: rtol = atol = 0 disables the residual test.
      inner_cg_->SetMaxIter(cfg_.n_inner);
      inner_cg_->SetRelTol(0.0);
      inner_cg_->SetAbsTol(0.0);
   }
   else
   {
      inner_cg_->SetMaxIter(200);
      inner_cg_->SetRelTol(cfg_.tol_inner);
      inner_cg_->SetAbsTol(0.0);
   }

   z_mass_.SetSize(pfes.GetTrueVSize());
   rhs_.SetSize(pfes.GetTrueVSize());
   w_.SetSize(pfes.GetTrueVSize());
   z_mass_.UseDevice(true);
   rhs_.UseDevice(true);
   w_.UseDevice(true);
}

void CahouetChabardSchurPC::Mult(const Vector& r, Vector& z) const
{
   INCNS_PROFILE("cc_schur::apply");

   // 1. Mass term: t = M_p^-1 r (diagonal multiply or fixed Chebyshev).
   mp_inv_->Mult(r, z_mass_);

   // 2. Poisson term (skipped entirely at sigma = 0: steady Stokes limit).
   if (sigma_ > 0.0)
   {
      rhs_ = r;
      if (singular_) { proj_->Project(rhs_); }
      w_ = 0.0;
      if (cfg_.schur_model == SchurModel::LaplacianLegacy)
      {
         // Comparison mode ONLY (T5): fixed V-cycle(s) on LOR L_p directly --
         // no inner CG, no B/B^T/M_v^-1 actions. Cheaper, non-robust.
         lp_->Mult(rhs_, w_);
      }
      else
      {
         INCNS_PROFILE("inner_cg");
         inner_cg_->Mult(rhs_, w_);
      }
   }
   else
   {
      w_ = 0.0;
   }

   // 3. Combine: z = nu_pc * t + sigma * w. The coefficients are LITERALLY
   // the viscosity and gamma0/dt (SPEC par.2.3 scaling guard: 1/dt lives
   // entirely in sigma on the Poisson term; the mass term's coefficient is
   // the dt-independent high-frequency plateau nu).
   z.Set(nu_pc_, z_mass_);
   if (sigma_ > 0.0) { z.Add(sigma_, w_); }
}

void CahouetChabardSchurPC::Reset(double sigma, double nu)
{
   if (sigma < 0.0)
   {
      throw std::invalid_argument("cc Reset: sigma must be >= 0");
   }
   if (nu <= 0.0)
   {
      throw std::invalid_argument("cc Reset: nu must be > 0");
   }
   sigma_ = sigma;
   // nu_pc tracks nu unless explicitly pinned by the config (an O(1) dial).
   nu_pc_ = cfg_.nu_pc < 0.0 ? nu : cfg_.nu_pc;
   // Everything structural (mass diagonals/Chebyshev, B, the L_p AMG
   // hierarchy) is sigma/nu-independent: reused always (SPEC par.9).
}

} // namespace incns
