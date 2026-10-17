#include "operators/pa_operator.hpp"

using namespace mfem;

namespace incns
{

PaOperator::PaOperator(ParFiniteElementSpace& fes, ElementScratch& scratch)
   : Operator(fes.GetVSize()), fes_(fes)
{
   restr_ = fes_.GetElementRestriction(ElementDofOrdering::LEXICOGRAPHIC);
   MFEM_VERIFY(restr_, "pa_operator: tensor elements with an element "
               "restriction only");
   scratch.Lend(0, restr_->Height(), xe_);
   scratch.Lend(1, restr_->Height(), ye_);
}

void PaOperator::AddDomainIntegrator(BilinearFormIntegrator* integ)
{
   integs_.emplace_back(integ);
}

void PaOperator::Assemble()
{
   for (auto& integ : integs_) { integ->AssemblePA(fes_); }
}

void PaOperator::Mult(const Vector& x, Vector& y) const
{
   if (integs_.empty()) { y = 0.0; return; }
   restr_->Mult(x, xe_);
   ye_ = 0.0;
   for (const auto& integ : integs_) { integ->AddMultPA(xe_, ye_); }
   restr_->MultTranspose(ye_, y);
}

void PaOperator::MultTranspose(const Vector& x, Vector& y) const
{
   if (integs_.empty()) { y = 0.0; return; }
   restr_->Mult(x, xe_);
   ye_ = 0.0;
   for (const auto& integ : integs_) { integ->AddMultTransposePA(xe_, ye_); }
   restr_->MultTranspose(ye_, y);
}

PaMixedOperator::PaMixedOperator(ParFiniteElementSpace& trial,
                                 ParFiniteElementSpace& test,
                                 ElementScratch& scratch)
   : Operator(test.GetVSize(), trial.GetVSize()), trial_(trial), test_(test)
{
   restr_trial_ =
      trial_.GetElementRestriction(ElementDofOrdering::LEXICOGRAPHIC);
   restr_test_ = test_.GetElementRestriction(ElementDofOrdering::LEXICOGRAPHIC);
   MFEM_VERIFY(restr_trial_ && restr_test_, "pa_operator: tensor elements "
               "with element restrictions only");
   scratch.Lend(0, restr_trial_->Height(), trial_e_);
   scratch.Lend(1, restr_test_->Height(), test_e_);
}

void PaMixedOperator::AddDomainIntegrator(BilinearFormIntegrator* integ)
{
   integs_.emplace_back(integ);
}

void PaMixedOperator::Assemble()
{
   for (auto& integ : integs_) { integ->AssemblePA(trial_, test_); }
}

void PaMixedOperator::Mult(const Vector& x, Vector& y) const
{
   if (integs_.empty()) { y = 0.0; return; }
   restr_trial_->Mult(x, trial_e_);
   test_e_ = 0.0;
   for (const auto& integ : integs_) { integ->AddMultPA(trial_e_, test_e_); }
   restr_test_->MultTranspose(test_e_, y);
}

void PaMixedOperator::MultTranspose(const Vector& x, Vector& y) const
{
   if (integs_.empty()) { y = 0.0; return; }
   restr_test_->Mult(x, test_e_);
   trial_e_ = 0.0;
   for (const auto& integ : integs_)
   {
      integ->AddMultTransposePA(test_e_, trial_e_);
   }
   restr_trial_->MultTranspose(trial_e_, y);
}

} // namespace incns
