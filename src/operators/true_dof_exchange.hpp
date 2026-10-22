/**
 * @file true_dof_exchange.hpp
 * @brief The halo exchange of a conforming parallel space for kernels that read
 *        and write true-dof vectors directly, without the L-vector that MFEM's
 *        parallel prolongation goes through.
 */
#pragma once

#include "mfem.hpp"

#include <cstddef>
#include <vector>

namespace incns
{

/**
 * @brief P and P^T of a conforming parallel space, split around the caller's
 *        kernels and without the full-vector copies.
 *
 * A local dof is owned (a true dof of this rank) or a ghost (a shared dof that
 * another rank owns). The caller's kernels read an owned dof's value from the
 * true-dof vector and a ghost's from Ghosts(), and write an owned dof's result
 * into the true-dof vector and a ghost's contribution into GhostSums().
 *
 * - Gather (P): the owners send their shared true dofs; Ghosts() receives them.
 * - Reduce (P^T): the ghost contributions go back to their owners, which add
 *   them to their true dofs in a fixed order (by neighbor rank, then message
 *   position): deterministic, no atomics.
 *
 * At np 2, MFEM's device prolongation spent most of its 0.15 ms (P) and
 * 0.17 ms (P^T) per apply at Q6 on the copies of all owned dofs between the
 * true-dof vector and the L-vector (SetSubArray, ExtractSubArray; nsys, job
 * 49573701). Here only the shared dofs move.
 *
 * Scalar maps for vector spaces ordered by nodes: component c of scalar true
 * dof t is t + c TrueStride(), of ghost g at vdim g + c (interleaved). Device
 * buffers go to MPI directly when it is GPU-aware, through the host otherwise.
 * One exchange in flight at a time.
 */
class TrueDofExchange
{
public:
   /// @param fes A conforming space, vdim 1 or ordered by nodes (borrowed;
   ///            the scalar group communicator is built from it).
   explicit TrueDofExchange(mfem::ParFiniteElementSpace& fes);
   ~TrueDofExchange();

   TrueDofExchange(const TrueDofExchange&) = delete;
   TrueDofExchange& operator=(const TrueDofExchange&) = delete;

   /// Per scalar local dof: its scalar true dof (owned, >= 0) or -1 - its
   /// ghost number.
   const mfem::Array<int>& LocalSource() const { return lsrc_; }
   /// The component stride of true-dof vectors (scalar true dofs).
   int TrueStride() const { return ntd_; }
   int NumGhosts() const { return ng_; }
   /// The owned true dofs that other ranks hold as ghosts (scalar, ascending).
   const mfem::Array<int>& OwnedShared() const { return rdof_; }
   int VDim() const { return vdim_; }

   /// Starts sending the owned shared dofs of @p x (true dofs).
   void GatherBegin(const mfem::Vector& x) const;
   /// Waits for the ghosts' values (Ghosts()).
   void GatherEnd() const;
   /// The ghosts' values after GatherEnd(): (vdim, ghosts).
   const mfem::Vector& Ghosts() const { return gbuf_; }

   /// The ghosts' contributions, written by the caller before ReduceBegin()
   /// (the same buffer as Ghosts()).
   mfem::Vector& GhostSums() const { return gbuf_; }
   /// Starts sending GhostSums() to the owners.
   void ReduceBegin() const;
   /// Waits; adds the received contributions to @p y (true dofs).
   void ReduceEnd(mfem::Vector& y) const;

   /// @return Bytes of device data this object owns.
   std::size_t MemoryBytes() const;

private:
   void Start(mfem::real_t* recv, const int* rcount, const int* roff,
              const mfem::real_t* send, const int* scount, const int* soff,
              int tag) const;

   MPI_Comm comm_;
   int vdim_ = 1, ntd_ = 0, ng_ = 0, ns_ = 0;
   mfem::Array<int> lsrc_;
   /// Neighbors with traffic, ascending rank; per neighbor the send segment
   /// (owned shared true dofs, offsets into sdof_) and the ghost segment.
   std::vector<int> nbr_, soff_, scount_, goff_, gcount_;
   mfem::Array<int> sdof_; ///< The send list's scalar true dofs.
   /// Reduce: each owned shared true dof (rdof_) and the send-list positions
   /// that receive its contributions (CSR roff_, ridx_; ascending).
   mfem::Array<int> rdof_, roff_, ridx_;
   mutable mfem::Vector sbuf_; ///< (vdim, ns): gather send, reduce receive.
   mutable mfem::Vector gbuf_; ///< (vdim, ng): gather receive, reduce send.
   mutable std::vector<MPI_Request> req_;
};

} // namespace incns
