#include "operators/pressure_schur.hpp"

namespace incns
{

using namespace mfem;

PressureMassSchur::PressureMassSchur(ParFiniteElementSpace& pfes,
                                     const RuleBook& rules, double scale)
   : Solver(pfes.GetTrueVSize()), scale_(scale), mass_(&pfes)
{
   MFEM_VERIFY(scale_ > 0.0, "pressure_schur: scale must be positive");

   const int dim = pfes.GetParMesh()->Dimension();
   const Geometry::Type geom = (dim == 3) ? Geometry::CUBE : Geometry::SQUARE;
   const int kp = pfes.FEColl()->GetOrder();

   auto* mi = new MassIntegrator;
   mi->SetIntRule(&rules.Get(geom, 2 * kp));
   mass_.AddDomainIntegrator(mi);
   mass_.SetAssemblyLevel(AssemblyLevel::PARTIAL);
   mass_.Assemble();

   // Device-aware (no-op on CPU): keep the diagonal and its inverse resident on
   // the device so the per-apply Mult never triggers a host<->device copy.
   Vector diag(pfes.GetTrueVSize());
   diag.UseDevice(true);
   mass_.AssembleDiagonal(diag);
   MFEM_VERIFY(diag.Min() > 0.0, "pressure_schur: non-positive mass diagonal");

   // inv_diag = scale / diag, built on the device so it never leaves it (the
   // per-apply Mult, y = x .* inv_diag, is device-aware Vector arithmetic).
   //
   // Spelled with mfem::Vector operators rather than a hand-written forall on
   // purpose: bessemer is compiled by mpicxx, never nvcc, so MFEM_HOST_DEVICE
   // expands to nothing here and forall's CUDA dispatch is preprocessed out
   // (general/forall.hpp, guarded on __CUDACC__). The lambda would then run on
   // the HOST over the device pointers Read()/Write() hand back -- a segfault
   // with "Invalid permissions" on a real GPU. These operators live inside
   // libmfem, which IS nvcc-built, so they dispatch to genuine device kernels.
   inv_diag_.SetSize(diag.Size());
   inv_diag_.UseDevice(true);
   inv_diag_ = scale_;
   inv_diag_ /= diag;
}

void PressureMassSchur::Mult(const Vector& x, Vector& y) const
{
   y = x;
   y *= inv_diag_;
}

} // namespace incns
