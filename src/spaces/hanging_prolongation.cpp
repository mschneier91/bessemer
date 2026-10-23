/**
 * @file hanging_prolongation.cpp
 * @brief HangingProlongation: the maps from MFEM's assembled P, and the apply
 *        and transpose kernels.
 */
#include "spaces/hanging_prolongation.hpp"

#include "operators/true_dof_exchange.hpp"

// This TU defines device kernels (mfem::forall). A host compiler would build
// them as host loops over device pointers, which segfaults on a GPU;
// src/CMakeLists.txt marks this file LANGUAGE CUDA, and this line turns a lost
// entry there into a compile error.
#if defined(MFEM_USE_CUDA) && !defined(__CUDACC__)
#error "hanging_prolongation.cpp defines device kernels and must be compiled by nvcc -- see the nvcc TU list in src/CMakeLists.txt"
#endif

#include <algorithm>
#include <cmath>
#include <vector>

namespace incns
{

using namespace mfem;

HangingProlongation::HangingProlongation(const ParFiniteElementSpace& fes)
   : Operator(fes.GetVSize(), fes.GetTrueVSize())
{
   MFEM_VERIFY(!fes.Conforming(), "HangingProlongation: a nonconforming space "
               "(a conforming one keeps MFEM's prolongation)");
   const HypreParMatrix* P = fes.Dof_TrueDof_Matrix();
   MFEM_VERIFY(P && P->Height() == Height() && P->Width() == Width(),
               "HangingProlongation: the space's P is not local dofs x true dofs");
   P->HostRead();
   SparseMatrix diag, offd;
   HYPRE_BigInt* cmap = nullptr;
   P->GetDiag(diag);
   P->GetOffd(offd, cmap);
   nt_ = Width();
   ng_ = offd.Width();
   const int n = Height(), nc = nt_ + ng_;
   const int* DI = diag.HostReadI();
   const int* DJ = diag.HostReadJ();
   const real_t* DA = diag.HostReadData();
   const int* OI = offd.HostReadI();
   const int* OJ = offd.HostReadJ();
   const real_t* OA = offd.HostReadData();

   // The rows: identity (one entry, 1) or hanging; columns >= nt_ are ghosts.
   std::vector<int> src(n), hoff(1, 0), hcol, tcount(nc, 0);
   std::vector<real_t> hw;
   for (int i = 0; i < n; i++)
   {
      const int nd = DI[i + 1] - DI[i], no = OI[i + 1] - OI[i];
      for (int k = DI[i]; k < DI[i + 1]; k++) { tcount[DJ[k]]++; }
      for (int k = OI[i]; k < OI[i + 1]; k++) { tcount[nt_ + OJ[k]]++; }
      MFEM_VERIFY(nd + no > 0, "HangingProlongation: local dof " << i
                  << " has an empty row");
      if (nd + no == 1 && (nd == 1 ? DA[DI[i]] : OA[OI[i]]) == 1.0)
      {
         src[i] = nd == 1 ? DJ[DI[i]] : nt_ + OJ[OI[i]];
         continue;
      }
      src[i] = -1 - nh_++;
      for (int k = DI[i]; k < DI[i + 1]; k++)
      {
         hcol.push_back(DJ[k]);
         hw.push_back(DA[k]);
      }
      for (int k = OI[i]; k < OI[i + 1]; k++)
      {
         hcol.push_back(nt_ + OJ[k]);
         hw.push_back(OA[k]);
      }
      hoff.push_back(static_cast<int>(hcol.size()));
   }

   // The transpose: per column its rows, ascending (rows visited in order).
   std::vector<int> toff(nc + 1, 0);
   for (int c = 0; c < nc; c++) { toff[c + 1] = toff[c] + tcount[c]; }
   std::vector<int> trow(toff[nc]), fill(toff.begin(), toff.end() - 1);
   std::vector<real_t> tw(toff[nc]);
   for (int i = 0; i < n; i++)
   {
      for (int k = DI[i]; k < DI[i + 1]; k++)
      {
         const int q = fill[DJ[k]]++;
         trow[q] = i;
         tw[q] = DA[k];
      }
      for (int k = OI[i]; k < OI[i + 1]; k++)
      {
         const int q = fill[nt_ + OJ[k]]++;
         trow[q] = i;
         tw[q] = OA[k];
      }
   }

   auto copy = [](const auto & from, auto & to)
   {
      to.SetSize(static_cast<int>(from.size()));
      std::copy(from.begin(), from.end(), to.HostWrite());
   };
   copy(src, src_);
   copy(hoff, hoff_);
   copy(hcol, hcol_);
   copy(hw, hw_);
   copy(toff, toff_);
   copy(trow, trow_);
   copy(tw, tw_);
   for (Vector* v : {&hw_, &tw_}) { v->UseDevice(true); }

   int np = 1;
   MPI_Comm_size(fes.GetComm(), &np);
   if (np > 1)
   {
      ex_ = std::make_unique<TrueDofExchange>(fes.GetComm(), nt_,
                                              fes.GetMyTDofOffset(), cmap, ng_);
   }
   // Back to hypre's memory space, as MFEM's own host reads do
   // (MergeDiagAndOffd): HostRead() moved the space's P to the host, and its
   // next SpMV took hypre's host path with device vectors (a segfault on a GPU
   // only, job 49664616).
   P->HypreRead();
}

HangingProlongation::~HangingProlongation() = default;

MemoryClass HangingProlongation::GetMemoryClass() const
{
   return Device::GetDeviceMemoryClass();
}

std::size_t HangingProlongation::MemoryBytes() const
{
   return (hw_.Size() + tw_.Size()) * sizeof(real_t) +
          (src_.Size() + hoff_.Size() + hcol_.Size() + toff_.Size() +
           trow_.Size()) * sizeof(int) +
          (ex_ ? ex_->MemoryBytes() : 0);
}

// The kernels: free functions in a named namespace (nvcc rejects an extended
// lambda in a private member, and in internal-linkage scope).
namespace hangingp
{

/// y = P x (|P| x with ABS): copies, and the hanging rows' weighted sums.
/// @p G holds the ghosts (columns >= nt).
template <bool ABS>
void Apply(int nt, const Array<int>& src, const Array<int>& hoff,
           const Array<int>& hcol, const Vector& hw, const Vector& x,
           const real_t* G, Vector& y)
{
   const int n = src.Size();
   const auto S = src.Read(), HO = hoff.Read(), HC = hcol.Read();
   const auto HW = hw.Read();
   const auto X = x.Read();
   auto Y = y.Write();
   mfem::forall(n, [ = ] MFEM_HOST_DEVICE(int i)
   {
      const int s = S[i];
      if (s >= 0)
      {
         Y[i] = s < nt ? X[s] : G[s - nt];
         return;
      }
      const int h = -1 - s;
      real_t v = 0.0;
      for (int k = HO[h]; k < HO[h + 1]; k++)
      {
         const int c = HC[k];
         const real_t w = ABS ? fabs(HW[k]) : HW[k];
         v += w * (c < nt ? X[c] : G[c - nt]);
      }
      Y[i] = v;
   });
}

/// y = P^T x (|P|^T x with ABS) on the true dofs, and the ghosts' sums in
/// @p G (columns >= nt), per column in the order of its rows.
template <bool ABS>
void ApplyTranspose(int nt, const Array<int>& toff, const Array<int>& trow,
                    const Vector& tw, const Vector& x, real_t* G, Vector& y)
{
   const int nc = toff.Size() - 1;
   const auto TO = toff.Read(), TR = trow.Read();
   const auto TW = tw.Read();
   const auto X = x.Read();
   auto Y = y.Write();
   mfem::forall(nc, [ = ] MFEM_HOST_DEVICE(int c)
   {
      real_t v = 0.0;
      for (int k = TO[c]; k < TO[c + 1]; k++)
      {
         const real_t w = ABS ? fabs(TW[k]) : TW[k];
         v += w * X[TR[k]];
      }
      if (c < nt) { Y[c] = v; }
      else { G[c - nt] = v; }
   });
}

} // namespace hangingp

template <bool ABS>
void HangingProlongation::Apply(const Vector& x, Vector& y) const
{
   const real_t* G = nullptr;
   if (ex_)
   {
      ex_->GatherBegin(x);
      ex_->GatherEnd();
      if (ng_ > 0) { G = ex_->Ghosts().Read(); }
   }
   hangingp::Apply<ABS>(nt_, src_, hoff_, hcol_, hw_, x, G, y);
}

template <bool ABS>
void HangingProlongation::ApplyTranspose(const Vector& x, Vector& y) const
{
   real_t* G = (ex_ && ng_ > 0) ? ex_->GhostSums().Write() : nullptr;
   hangingp::ApplyTranspose<ABS>(nt_, toff_, trow_, tw_, x, G, y);
   if (ex_)
   {
      ex_->ReduceBegin();
      ex_->ReduceEnd(y);
   }
}

void HangingProlongation::Mult(const Vector& x, Vector& y) const
{
   Apply<false>(x, y);
}

void HangingProlongation::MultTranspose(const Vector& x, Vector& y) const
{
   ApplyTranspose<false>(x, y);
}

void HangingProlongation::AbsMult(const Vector& x, Vector& y) const
{
   Apply<true>(x, y);
}

void HangingProlongation::AbsMultTranspose(const Vector& x, Vector& y) const
{
   ApplyTranspose<true>(x, y);
}

} // namespace incns
