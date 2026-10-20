/**
 * @file oseen_pcd.cpp
 * @brief OseenPcdSchurPC.
 */
#include "oseen/oseen_pcd.hpp"

#include <cmath>

namespace incns
{

using namespace mfem;

namespace
{

/// -(w.n) where w.n < 0, else 0: the inflow part of the boundary (host).
class InflowCoefficient : public Coefficient
{
public:
   explicit InflowCoefficient(const ParGridFunction& w) : w_(w) {}
   real_t Eval(ElementTransformation& T, const IntegrationPoint& ip) override
   {
      w_.GetVectorValue(T, ip, wv_);
      const int d = wv_.Size();
      nor_.SetSize(d);
      CalcOrtho(T.Jacobian(), nor_);
      const real_t wn = (wv_ * nor_) / nor_.Norml2();
      return wn < 0.0 ? -wn : 0.0;
   }

private:
   const ParGridFunction& w_;
   Vector wv_, nor_;
};

} // namespace

OseenPcdSchurPC::OseenPcdSchurPC(ParFiniteElementSpace& vfes,
                                 ParFiniteElementSpace& pfes,
                                 const RuleBook& rules, const Operator& B,
                                 const Array<int>& outflow_attrs, bool singular,
                                 const Options& opts)
   : Solver(pfes.GetTrueVSize()), vfes_(vfes), pfes_(pfes), opts_(opts),
     singular_(singular)
{
   const int dim = pfes.GetParMesh()->Dimension();
   const Geometry::Type geom = (dim == 3) ? Geometry::CUBE : Geometry::SQUARE;
   const int kp = pfes.GetMaxElementOrder();
   const int ku = vfes.GetMaxElementOrder();
   Array<int> none;

   // M_p (consistent, Gauss) and its Chebyshev inverse.
   mp_form_ = std::make_unique<ParBilinearForm>(&pfes);
   {
      auto* mi = new MassIntegrator;
      mi->SetIntRule(&rules.Get(geom, 2 * kp));
      mp_form_->AddDomainIntegrator(mi);
   }
   mp_form_->SetAssemblyLevel(AssemblyLevel::PARTIAL);
   mp_form_->Assemble();
   mp_form_->FormSystemMatrix(none, Mp_);
   mp_diag_.SetSize(pfes.GetTrueVSize());
   mp_form_->AssembleDiagonal(mp_diag_);
   mp_inv_ = std::make_unique<MassInverse>(*Mp_.Ptr(), mp_diag_, pfes.GetComm(),
                                           MassInvType::Chebyshev,
                                           opts_.mp_chebyshev);

   // M_v: the collocated (diagonal) velocity mass, for A_p = B M_v^-1 B^T.
   mv_form_ = std::make_unique<ParBilinearForm>(&vfes);
   {
      auto* mvi = new VectorMassIntegrator;
      mvi->SetIntRule(&rules.CollocatedMass(geom, ku));
      mv_form_->AddDomainIntegrator(mvi);
   }
   mv_form_->SetAssemblyLevel(AssemblyLevel::PARTIAL);
   mv_form_->Assemble();
   mv_form_->FormSystemMatrix(none, Mv_);
   mv_diag_.SetSize(vfes.GetTrueVSize());
   mv_form_->AssembleDiagonal(mv_diag_);
   mv_inv_ = std::make_unique<MassInverse>(*Mv_.Ptr(), mv_diag_, pfes.GetComm(),
                                           MassInvType::DiagDirect);

   bmb_ = std::make_unique<MixedPoissonOperator>(B, *mv_inv_);
   proj_ = std::make_unique<ConstantPressureProjector>(pfes);
   if (singular_) { bmb_proj_ = std::make_unique<ProjectedOperator>(*bmb_, *proj_); }
   lp_ = std::make_unique<LpSurrogate>(pfes, outflow_attrs, singular_);
   cg_ = std::make_unique<CGSolver>(pfes.GetComm());
   cg_->SetOperator(singular_ ? static_cast<Operator&>(*bmb_proj_)
                    : static_cast<Operator&>(*bmb_));
   cg_->SetPreconditioner(*lp_);
   cg_->iterative_mode = false;
   cg_->SetPrintLevel(-1);
   cg_->SetMaxIter(opts_.n_inner);
   cg_->SetRelTol(0.0);
   cg_->SetAbsTol(0.0);

   // N_p: the pressure-space Oseen operator (tau of the velocity order).
   wind_p_ = std::make_unique<OseenWindData>(vfes, rules, opts_.nu, kp);
   np_ = std::make_unique<ScalarOseenOperator>(pfes, *wind_p_, opts_.oss);

   for (Vector* v : {&t_, &s_, &w_}) { v->UseDevice(true); }
}

OseenPcdSchurPC::~OseenPcdSchurPC() = default;

void OseenPcdSchurPC::Update(const ParGridFunction& wind, double sigma)
{
   sigma_ = sigma;
   wind_p_->Update(wind);
   np_->Assemble();
   if (opts_.inflow_term && pfes_.GetParMesh()->bdr_attributes.Size() > 0)
   {
      wind.HostRead();
      InflowCoefficient q(wind);
      ParBilinearForm f(&pfes_);
      f.AddBoundaryIntegrator(new MassIntegrator(q));
      f.Assemble();
      f.Finalize();
      inflow_.reset(f.ParallelAssemble());
   }
}

void OseenPcdSchurPC::Mult(const Vector& r, Vector& z) const
{
   const int n = height;
   t_.SetSize(n);
   s_.SetSize(n);
   w_.SetSize(n);
   // t = M_p^-1 r; s = sigma r + N_p t (+ inflow t).
   mp_inv_->Mult(r, t_);
   np_->Mult(t_, s_);
   if (inflow_) { inflow_->AddMult(t_, s_); }
   s_.Add(sigma_, r);
   if (singular_) { proj_->Project(s_); }
   // w = A_p^-1 s; z = w + nu~ t.
   w_ = 0.0;
   cg_->Mult(s_, w_);
   add(w_, opts_.nu_tilde, t_, z);
}

} // namespace incns
