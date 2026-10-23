/**
 * @file true_dof_exchange.cpp
 * @brief TrueDofExchange: the shared dofs' gather and reduce of a conforming
 *        parallel space (pack and accumulate kernels, MPI point to point).
 */
#include "operators/true_dof_exchange.hpp"

// This TU defines device kernels (mfem::forall). A host compiler would build
// them as host loops over device pointers, which segfaults on a GPU;
// src/CMakeLists.txt marks this file LANGUAGE CUDA, and this line turns a lost
// entry there into a compile error.
#if defined(MFEM_USE_CUDA) && !defined(__CUDACC__)
#error "true_dof_exchange.cpp defines device kernels and must be compiled by nvcc -- see the nvcc TU list in src/CMakeLists.txt"
#endif

#include "util/device.hpp"

#include <algorithm>
#include <map>
#include <memory>
#include <tuple>

namespace incns
{

using namespace mfem;

namespace
{

constexpr int kGatherTag = 3917, kReduceTag = 3918;

/// Whether MPI takes device pointers here.
bool DeviceBuffers()
{
   return Device::Allows(Backend::DEVICE_MASK) && Device::GetGPUAwareMPI();
}

} // namespace

TrueDofExchange::TrueDofExchange(ParFiniteElementSpace& fes)
   : comm_(fes.GetComm()), vdim_(fes.GetVDim())
{
   MFEM_VERIFY(fes.Conforming(), "TrueDofExchange: a conforming space");
   MFEM_VERIFY(vdim_ == 1 || fes.GetOrdering() == Ordering::byNODES,
               "TrueDofExchange: vdim 1 or ordered by nodes");
   const int nd = fes.GetNDofs();
   ntd_ = fes.GetTrueVSize() / vdim_;

   // Owned dofs: their scalar true dofs (the components at the stride).
   std::vector<int> src(nd);
   std::vector<char> ghost(nd, 0);
   for (int j = 0; j < nd; j++)
   {
      const int t = fes.GetLocalTDofNumber(j);
      for (int c = 1; c < vdim_; c++)
      {
         MFEM_VERIFY(fes.GetLocalTDofNumber(j + c * nd) == (t < 0 ? -1 : t + c * ntd_),
                     "TrueDofExchange: the true dofs ordered by nodes");
      }
      src[j] = t;
   }

   // Groups: those this rank masters go to their other members, ascending;
   // the others come from their masters, in the masters' group order.
   std::unique_ptr<GroupCommunicator> gc(fes.ScalarGroupComm());
   const GroupTopology& gt = gc->GetGroupTopology();
   const Table& gl = gc->GroupLDofTable();
   std::map<int, std::vector<int>> send;          // rank -> groups
   std::vector<std::tuple<int, int, int>> recv;    // (rank, master group, group)
   for (int g = 1; g < gt.NGroups(); g++)
   {
      if (gl.RowSize(g) == 0) { continue; }
      if (gt.IAmMaster(g))
      {
         const int* lp = gt.GetGroup(g);
         for (int i = 0; i < gt.GetGroupSize(g); i++)
         {
            if (lp[i] != 0) { send[gt.GetNeighborRank(lp[i])].push_back(g); }
         }
      }
      else
      {
         recv.emplace_back(gt.GetGroupMasterRank(g), gt.GetGroupMasterGroup(g), g);
      }
   }
   std::sort(recv.begin(), recv.end());
   for (const auto& r : recv) { nbr_.push_back(std::get<0>(r)); }
   for (const auto& s : send) { nbr_.push_back(s.first); }
   std::sort(nbr_.begin(), nbr_.end());
   nbr_.erase(std::unique(nbr_.begin(), nbr_.end()), nbr_.end());
   const int nn = static_cast<int>(nbr_.size());
   soff_.assign(nn, 0);
   scount_.assign(nn, 0);
   goff_.assign(nn, 0);
   gcount_.assign(nn, 0);

   // Ghost numbers: by neighbor, then the master's group order.
   std::size_t r = 0;
   for (int i = 0; i < nn; i++)
   {
      goff_[i] = ng_;
      for (; r < recv.size() && std::get<0>(recv[r]) == nbr_[i]; r++)
      {
         const int g = std::get<2>(recv[r]);
         const int* row = gl.GetRow(g);
         for (int k = 0; k < gl.RowSize(g); k++)
         {
            const int j = row[k];
            MFEM_VERIFY(src[j] < 0 && !ghost[j], "TrueDofExchange: a ghost owned or "
                        "received twice");
            ghost[j] = 1;
            src[j] = -1 - ng_++;
         }
      }
      gcount_[i] = ng_ - goff_[i];
   }
   for (int j = 0; j < nd; j++)
   {
      MFEM_VERIFY(src[j] >= 0 || ghost[j], "TrueDofExchange: a dof neither owned "
                  "nor received");
   }

   // The send list: by neighbor, then group, the owned true dofs.
   std::vector<int> sdof;
   for (int i = 0; i < nn; i++)
   {
      soff_[i] = static_cast<int>(sdof.size());
      const auto it = send.find(nbr_[i]);
      if (it == send.end()) { continue; }
      for (const int g : it->second)
      {
         const int* row = gl.GetRow(g);
         for (int k = 0; k < gl.RowSize(g); k++)
         {
            MFEM_VERIFY(src[row[k]] >= 0, "TrueDofExchange: a sent dof not owned");
            sdof.push_back(src[row[k]]);
         }
      }
      scount_[i] = static_cast<int>(sdof.size()) - soff_[i];
   }
   lsrc_.SetSize(nd);
   std::copy(src.begin(), src.end(), lsrc_.HostWrite());
   Finish(sdof);
}

TrueDofExchange::TrueDofExchange(MPI_Comm comm, int ntd, HYPRE_BigInt first,
                                 const HYPRE_BigInt* ghosts, int ng)
   : comm_(comm), vdim_(1), ntd_(ntd), ng_(ng)
{
   int np = 1;
   MPI_Comm_size(comm_, &np);
   // Every rank's first true dof, and the total: the owners' ranges.
   std::vector<HYPRE_BigInt> starts(np + 1, 0);
   MPI_Allgather(&first, 1, HYPRE_MPI_BIG_INT, starts.data(), 1,
                 HYPRE_MPI_BIG_INT, comm_);
   HYPRE_BigInt mine = ntd, total = 0;
   MPI_Allreduce(&mine, &total, 1, HYPRE_MPI_BIG_INT, MPI_SUM, comm_);
   starts[np] = total;

   // The ghosts by owner (ascending numbers: ascending owners).
   std::vector<int> need(np, 0);
   for (int g = 0; g < ng; g++)
   {
      MFEM_VERIFY(g == 0 || ghosts[g] > ghosts[g - 1], "TrueDofExchange: the "
                  "ghosts must be ascending");
      const int o = static_cast<int>(std::upper_bound(starts.begin(), starts.end(),
                                     ghosts[g]) - starts.begin()) - 1;
      MFEM_VERIFY(o >= 0 && o < np && (ghosts[g] < first || ghosts[g] >= first + ntd),
                  "TrueDofExchange: ghost " << ghosts[g] << " out of range or owned");
      need[o]++;
   }
   // Who reads what from this rank.
   std::vector<int> give(np, 0);
   MPI_Alltoall(need.data(), 1, MPI_INT, give.data(), 1, MPI_INT, comm_);
   std::vector<int> ndispl(np, 0), gdispl(np, 0);
   for (int p = 1; p < np; p++)
   {
      ndispl[p] = ndispl[p - 1] + need[p - 1];
      gdispl[p] = gdispl[p - 1] + give[p - 1];
   }
   std::vector<HYPRE_BigInt> asked(gdispl[np - 1] + give[np - 1]);
   MPI_Alltoallv(const_cast<HYPRE_BigInt*>(ghosts), need.data(), ndispl.data(),
                 HYPRE_MPI_BIG_INT, asked.data(), give.data(), gdispl.data(),
                 HYPRE_MPI_BIG_INT, comm_);

   for (int p = 0; p < np; p++)
   {
      if (need[p] > 0 || give[p] > 0) { nbr_.push_back(p); }
   }
   const int nn = static_cast<int>(nbr_.size());
   soff_.assign(nn, 0);
   scount_.assign(nn, 0);
   goff_.assign(nn, 0);
   gcount_.assign(nn, 0);
   std::vector<int> sdof;
   for (int i = 0; i < nn; i++)
   {
      const int p = nbr_[i];
      goff_[i] = ndispl[p];
      gcount_[i] = need[p];
      soff_[i] = static_cast<int>(sdof.size());
      for (int k = gdispl[p]; k < gdispl[p] + give[p]; k++)
      {
         const HYPRE_BigInt t = asked[k] - first;
         MFEM_VERIFY(t >= 0 && t < ntd, "TrueDofExchange: asked for a dof not "
                     "owned here");
         sdof.push_back(static_cast<int>(t));
      }
      scount_[i] = give[p];
   }
   Finish(sdof);
}

void TrueDofExchange::Finish(const std::vector<int>& sdof)
{
   ns_ = static_cast<int>(sdof.size());

   // Reduce: per owned shared true dof, its send-list positions, ascending
   // (by neighbor): the order its contributions are added in.
   std::map<int, std::vector<int>> pos;
   for (int k = 0; k < ns_; k++) { pos[sdof[k]].push_back(k); }
   rdof_.SetSize(static_cast<int>(pos.size()));
   roff_.SetSize(rdof_.Size() + 1);
   ridx_.SetSize(ns_);
   {
      int* rd = rdof_.HostWrite();
      int* ro = roff_.HostWrite();
      int* ri = ridx_.HostWrite();
      int m = 0, k = 0;
      ro[0] = 0;
      for (const auto& p : pos)
      {
         rd[m] = p.first;
         for (const int q : p.second) { ri[k++] = q; }
         ro[++m] = k;
      }
   }
   sdof_.SetSize(ns_);
   std::copy(sdof.begin(), sdof.end(), sdof_.HostWrite());
   sbuf_.SetSize(vdim_ * ns_);
   gbuf_.SetSize(vdim_ * ng_);
   sbuf_.UseDevice(true);
   gbuf_.UseDevice(true);
   req_.reserve(2 * nbr_.size());
}

TrueDofExchange::~TrueDofExchange() = default;

std::size_t TrueDofExchange::MemoryBytes() const
{
   return (sbuf_.Size() + gbuf_.Size()) * sizeof(real_t) +
          (lsrc_.Size() + sdof_.Size() + rdof_.Size() + roff_.Size() + ridx_.Size()) *
          sizeof(int);
}

void TrueDofExchange::Start(real_t* recv, const int* rcount, const int* roff,
                            const real_t* send, const int* scount,
                            const int* soff, int tag) const
{
   req_.clear();
   const MPI_Datatype type = MPITypeMap<real_t>::mpi_type;
   const int nn = static_cast<int>(nbr_.size());
   for (int i = 0; i < nn; i++)
   {
      if (rcount[i] == 0) { continue; }
      req_.emplace_back();
      MPI_Irecv(recv + vdim_ * roff[i], vdim_ * rcount[i], type, nbr_[i], tag, comm_,
                &req_.back());
   }
   for (int i = 0; i < nn; i++)
   {
      if (scount[i] == 0) { continue; }
      req_.emplace_back();
      MPI_Isend(const_cast<real_t*>(send + vdim_ * soff[i]), vdim_ * scount[i], type,
                nbr_[i], tag, comm_, &req_.back());
   }
}

void TrueDofExchange::GatherBegin(const Vector& x) const
{
   if (nbr_.empty()) { return; }
   if (ns_ > 0)
   {
      const int vd = vdim_, ntd = ntd_;
      const auto S = sdof_.Read();
      const auto X = x.Read();
      auto B = sbuf_.Write();
      mfem::forall(ns_, [ = ] MFEM_HOST_DEVICE(int k)
      {
         for (int c = 0; c < vd; c++) { B[vd * k + c] = X[S[k] + ntd * c]; }
      });
   }
   const real_t* send = nullptr;
   real_t* recv = nullptr;
   if (DeviceBuffers())
   {
      DeviceSync(); // the pack kernel's results, before MPI reads them
      send = sbuf_.Read();
      recv = gbuf_.Write();
   }
   else
   {
      send = sbuf_.HostRead();
      recv = gbuf_.HostWrite();
   }
   Start(recv, gcount_.data(), goff_.data(), send, scount_.data(), soff_.data(),
         kGatherTag);
}

void TrueDofExchange::GatherEnd() const
{
   if (req_.empty()) { return; }
   MPI_Waitall(static_cast<int>(req_.size()), req_.data(), MPI_STATUSES_IGNORE);
   req_.clear();
}

void TrueDofExchange::ReduceBegin() const
{
   if (nbr_.empty()) { return; }
   const real_t* send = nullptr;
   real_t* recv = nullptr;
   if (DeviceBuffers())
   {
      DeviceSync(); // the caller's kernels' ghost sums, before MPI reads them
      send = gbuf_.Read();
      recv = sbuf_.Write();
   }
   else
   {
      send = gbuf_.HostRead();
      recv = sbuf_.HostWrite();
   }
   Start(recv, scount_.data(), soff_.data(), send, gcount_.data(), goff_.data(),
         kReduceTag);
}

void TrueDofExchange::ReduceEnd(Vector& y) const
{
   if (nbr_.empty()) { return; }
   if (!req_.empty())
   {
      MPI_Waitall(static_cast<int>(req_.size()), req_.data(), MPI_STATUSES_IGNORE);
      req_.clear();
   }
   const int m = rdof_.Size();
   if (m == 0) { return; }
   const int vd = vdim_, ntd = ntd_;
   const auto R = rdof_.Read(), O = roff_.Read(), I = ridx_.Read();
   const auto B = sbuf_.Read();
   auto Y = y.ReadWrite();
   mfem::forall(m, [ = ] MFEM_HOST_DEVICE(int i)
   {
      const int t = R[i];
      for (int c = 0; c < vd; c++)
      {
         real_t v = Y[t + ntd * c];
         for (int k = O[i]; k < O[i + 1]; k++) { v += B[vd * I[k] + c]; }
         Y[t + ntd * c] = v;
      }
   });
}

} // namespace incns
