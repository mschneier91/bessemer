/**
 * @file interface_accelerator.hpp
 * @brief Accelerators of the interface fixed-point iteration of a
 *        partitioned fluid-structure coupling: d_{k+1} from d_k and the
 *        solid's answer d~_k = G(d_k) (docs/fsi.md).
 */
#pragma once

#include "fsi/fsi_parameters.hpp"
#include "mfem.hpp"

#include <deque>
#include <memory>
#include <vector>

namespace incns
{

/**
 * @brief One subiteration's update of a fixed-point iteration d = G(d) on
 *        distributed vectors (true dofs; inner products are global).
 */
class InterfaceAccelerator
{
public:
   virtual ~InterfaceAccelerator() = default;

   /// A new time step begins (what the accelerator learnt may be kept).
   virtual void NewStep() = 0;

   /// The current step's attempt is abandoned (the coupled step is taken
   /// again at another dt): forget what it taught, keep the earlier steps'.
   virtual void DiscardStep() = 0;

   /// The abandoned attempt FAILED (its subiterations, solid or flow solve):
   /// what the earlier steps taught may have misled it, so forget that too
   /// (after DiscardStep(); a no-op for an accelerator that keeps nothing).
   virtual void ForgetKept() {}

   /// The input the last Next() proposed was moved back toward that call's
   /// input d: next = d + s (next - d) (FsiCase's backtrack after a failed
   /// solid solve). IQN-ILS needs nothing, its columns being the pairs
   /// actually evaluated; a relaxation scales its last factor.
   virtual void Shrink(double /*s*/) {}

   /**
    * @param d      The input of this subiteration, d_k.
    * @param dtilde The solid's answer G(d_k).
    * @param next   The next input d_{k+1}.
    * @return The relaxation factor used (NaN for a quasi-Newton update).
    */
   virtual double Next(const mfem::Vector& d, const mfem::Vector& dtilde,
                       mfem::Vector& next) = 0;

   /// What the accelerator carries from step to step, for a checkpoint: the
   /// sizes of its parts and their vectors (true dofs), in the method's
   /// layout. NewStep() rebuilds everything else.
   struct State
   {
      std::vector<int> counts;           ///< The parts' sizes.
      std::vector<mfem::Vector> vectors; ///< Their vectors, in order.
   };
   /// @return The state a restart needs (relaxation carries none).
   virtual State ExportState() const { return State(); }
   /// @param s A state from ExportState() of the same method.
   virtual void ImportState(const State& s)
   {
      MFEM_VERIFY(s.counts.empty() && s.vectors.empty(), "fsi: the "
                  "checkpoint's accelerator state is not this method's");
   }
};

/**
 * @brief Constant or dynamic Aitken relaxation (Kuttler and Wall 2008):
 *        d_{k+1} = d_k + omega_k r_k, r_k = G(d_k) - d_k.
 *
 * omega_0 is the given factor; then omega_k = -omega_{k-1} r_{k-1}^T (r_k -
 * r_{k-1}) / |r_k - r_{k-1}|^2, kept within [-omega_max, omega_max]. Every
 * step restarts from omega_0. With aitken off every factor is omega_0.
 */
class AitkenRelaxation : public InterfaceAccelerator
{
public:
   /**
    * @param omega0    The first factor (every factor when !aitken).
    * @param omega_max The bound on |omega|.
    * @param aitken    Dynamic factors (else constant).
    * @param comm      Communicator of the inner products.
    */
   AitkenRelaxation(double omega0, double omega_max, bool aitken, MPI_Comm comm);

   void NewStep() override;
   void DiscardStep() override { has_prev_ = false; }
   /// The last factor was effectively s omega (Aitken's next one uses it).
   void Shrink(double s) override { omega_ *= s; }

   double Next(const mfem::Vector& d, const mfem::Vector& dtilde,
               mfem::Vector& next) override;

private:
   double omega0_;         ///< First factor.
   double omega_max_;      ///< Bound on |omega|.
   bool aitken_;           ///< Dynamic factors.
   MPI_Comm comm_;         ///< Communicator.
   double omega_ = 0.0;    ///< The last factor.
   bool has_prev_ = false; ///< Whether r_prev_ holds a residual.
   mfem::Vector r_prev_;   ///< The previous residual.
};

/**
 * @brief The interface quasi-Newton method with an inverse Jacobian from a
 *        least-squares model, IQN-ILS (Degroote, Bathe and Vierendeels
 *        2009).
 *
 * With V the differences of the residuals r_i = G(d_i) - d_i and W those of
 * the outputs G(d_i), d_{k+1} = G(d_k) + W c, c the least-squares solution of
 * V c = -r_k (modified Gram-Schmidt, global inner products; a column nearly
 * dependent on the newer ones is dropped). The first subiteration of a step
 * without columns relaxes by omega_0. The columns of the last `reuse` steps
 * are kept (newest first), which often halves the subiterations.
 */
class IqnIls : public InterfaceAccelerator
{
public:
   /**
    * @param omega0 Relaxation of a subiteration with no columns.
    * @param reuse  Earlier steps whose columns are kept (0: none).
    * @param comm   Communicator of the inner products.
    */
   IqnIls(double omega0, int reuse, MPI_Comm comm);

   void NewStep() override;
   void DiscardStep() override;
   /// The kept steps' columns go: the retry starts its least squares afresh.
   void ForgetKept() override { past_.clear(); }

   double Next(const mfem::Vector& d, const mfem::Vector& dtilde,
               mfem::Vector& next) override;

   /// @return The columns in the last least-squares problem.
   int Columns() const { return columns_; }

   /// This step's columns and the kept steps', newest first (counts: the
   /// columns of each; vectors: each step's V, then its W).
   State ExportState() const override;
   void ImportState(const State& s) override;

private:
   /// One step's columns, newest first.
   struct Columns_
   {
      std::deque<mfem::Vector> v; ///< Residual differences.
      std::deque<mfem::Vector> w; ///< Output differences.
   };

   double omega0_;    ///< Relaxation without columns.
   int reuse_;        ///< Earlier steps kept.
   MPI_Comm comm_;    ///< Communicator.
   Columns_ current_; ///< This step's columns.
   std::deque<Columns_> past_; ///< Earlier steps' columns, newest first.
   bool has_prev_ = false;     ///< Whether the previous subiteration is kept.
   mfem::Vector r_prev_;       ///< Its residual.
   mfem::Vector dt_prev_;      ///< Its output.
   int columns_ = 0;           ///< See Columns().
};

/**
 * @param f    The fsi settings (acceleration, relaxation, omega_max, reuse).
 * @param comm Communicator.
 * @return The accelerator they ask for.
 */
std::unique_ptr<InterfaceAccelerator> MakeInterfaceAccelerator(
   const FsiParameters& f, MPI_Comm comm);

} // namespace incns
