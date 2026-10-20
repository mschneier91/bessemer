/**
 * @file oseen_convection.cpp
 * @brief ScalarOseenOperator and ComponentwiseOperator.
 */
#include "oseen/oseen_convection.hpp"

// This TU defines device kernels (mfem::forall). A host compiler would build
// them as host loops over device pointers, which segfaults on a GPU;
// src/CMakeLists.txt marks this file LANGUAGE CUDA, and this line turns a lost
// entry there into a compile error.
#if defined(MFEM_USE_CUDA) && !defined(__CUDACC__)
#error "oseen_convection.cpp defines device kernels and must be compiled by nvcc -- see the nvcc TU list in src/CMakeLists.txt"
#endif

namespace incns
{

using namespace mfem;

ScalarOseenOperator::ScalarOseenOperator(ParFiniteElementSpace& sfes,
      OseenWindData& wind, bool oss)
   : Operator(sfes.GetTrueVSize()), sfes_(sfes), wind_(wind), oss_(oss)
{
   MFEM_VERIFY(sfes.GetVDim() == 1, "ScalarOseenOperator: a scalar space");
   winv_.UseDevice(true);
   t1_.UseDevice(true);
   t2_.UseDevice(true);
}

void ScalarOseenOperator::Assemble()
{
   Array<int> none;
   conv_ = std::make_unique<ParBilinearForm>(&sfes_);
   conv_->SetAssemblyLevel(AssemblyLevel::PARTIAL);
   {
      auto* c = new ConvectionIntegrator(wind_.Wind(), 1.0);
      c->SetIntRule(&wind_.ConvRule());
      conv_->AddDomainIntegrator(c);
      auto* m = new MassIntegrator(wind_.HalfDivWind());
      m->SetIntRule(&wind_.ConvRule());
      conv_->AddDomainIntegrator(m);
   }
   conv_->Assemble();
   conv_->FormSystemMatrix(none, Conv_);
   if (!oss_) { return; }

   ktau_ = std::make_unique<ParBilinearForm>(&sfes_);
   ktau_->SetAssemblyLevel(AssemblyLevel::PARTIAL);
   {
      auto* k = new DiffusionIntegrator(wind_.TauWW());
      k->SetIntRule(&wind_.OssRule());
      ktau_->AddDomainIntegrator(k);
   }
   ktau_->Assemble();
   ktau_->FormSystemMatrix(none, Ktau_);

   ctau_ = std::make_unique<ParBilinearForm>(&sfes_);
   ctau_->SetAssemblyLevel(AssemblyLevel::PARTIAL);
   {
      auto* c = new ConvectionIntegrator(wind_.TauW(), 1.0);
      c->SetIntRule(&wind_.OssRule());
      ctau_->AddDomainIntegrator(c);
   }
   ctau_->Assemble();
   ctau_->FormSystemMatrix(none, Ctau_);

   // W_tau = diag (tau_c phi_l, 1) at the GLL rule; its inverse where > 0.
   ParBilinearForm w(&sfes_);
   w.SetAssemblyLevel(AssemblyLevel::PARTIAL);
   {
      auto* m = new MassIntegrator(wind_.TauGll());
      m->SetIntRule(&wind_.GllRule());
      w.AddDomainIntegrator(m);
   }
   w.Assemble();
   winv_.SetSize(height);
   w.AssembleDiagonal(winv_);
   auto wi = winv_.ReadWrite();
   mfem::forall(height, [ = ] MFEM_HOST_DEVICE(int i)
   {
      wi[i] = (wi[i] > 0.0) ? 1.0 / wi[i] : 0.0;
   });
}

void ScalarOseenOperator::MultConvection(const Vector& x, Vector& y) const
{
   MFEM_VERIFY(Conv_.Ptr(), "ScalarOseenOperator: Assemble() first");
   Conv_->Mult(x, y);
}

void ScalarOseenOperator::MultStabilization(const Vector& x, Vector& y) const
{
   MFEM_VERIFY(oss_ && Ktau_.Ptr(), "ScalarOseenOperator: no OSS term assembled");
   Ktau_->Mult(x, y);
   t1_.SetSize(height);
   t2_.SetSize(height);
   Ctau_->Mult(x, t1_);
   t1_ *= winv_;
   Ctau_->MultTranspose(t1_, t2_);
   y -= t2_;
}

void ScalarOseenOperator::Mult(const Vector& x, Vector& y) const
{
   MFEM_VERIFY(Conv_.Ptr(), "ScalarOseenOperator: Assemble() first");
   Conv_->Mult(x, y);
   if (!oss_) { return; }
   t1_.SetSize(height);
   t2_.SetSize(height);
   Ktau_->AddMult(x, y);
   Ctau_->Mult(x, t1_);
   t1_ *= winv_;
   Ctau_->MultTranspose(t1_, t2_);
   y -= t2_;
}

void ComponentwiseOperator::Mult(const Vector& x, Vector& y) const
{
   const int n = op_.Height();
   // Write y's base first: blocks written through aliases while the base is
   // valid only on the host would be overwritten by the next base Read().
   x.Read();
   y.Write();
   Vector xc, yc;
   for (int c = 0; c < vdim_; c++)
   {
      xc.MakeRef(const_cast<Vector&>(x), c * n, n);
      yc.MakeRef(y, c * n, n);
      op_.Mult(xc, yc);
   }
}

} // namespace incns
