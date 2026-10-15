/**
 * @file lean_vector_diffusion.hpp
 * @brief mfem::VectorDiffusionIntegrator with compact partial-assembly storage
 *        and a diagonal that needs no per-thread stack on a GPU.
 */
#pragma once

#include "mfem.hpp"

namespace incns
{

/**
 * @brief Vector diffusion (nu grad u, grad v) for a scalar coefficient, with
 *        two economies over mfem::VectorDiffusionIntegrator; anything else
 *        falls back to MFEM.
 *
 * @par Compact storage (Storage::Compact, the default)
 * MFEM's AssemblePA stores, at every quadrature point, the dim x dim geometric
 * matrix for EACH velocity component: 9 x 3 = 27 numbers per point in 3D, of
 * which 6 differ. For a scalar coefficient the components share one symmetric
 * matrix, so this stores 6 numbers per point in 3D (3 in 2D): 1.5 GB instead of
 * 6.75 GB for 64^3 Q3 elements. The apply is MFEM's own sum-factorized kernel
 * (SmemPAVectorDiffusionApply2D/3D) with the one change that every component
 * reads the shared matrix, and the setup is MFEM's arithmetic; the result
 * equals MFEM's to roundoff (lean_vector_diffusion_test). Storage::Mfem keeps
 * MFEM's AssemblePA and AddMultPA (deck `discretization.vector_diffusion:
 * mfem`), for comparison or should MFEM's kernel change.
 *
 * @par The diagonal
 * MFEM's 3D kernel (PAVectorDiffusionDiagonal3D, called without its template
 * sizes) declares two work arrays per thread at the device's largest supported
 * sizes, 44 kB of stack per thread under CUDA; the CUDA runtime reserves that
 * for every thread the device can run at once, about 11 GB of an H100 for the
 * life of the process (docs/upstream/mfem_vector_diffusion_diagonal.md). Here
 * the diagonal is a direct sum over the quadrature points with no local arrays
 * (O(p^6) per element; it runs at setup and on a Delta-t change only), in
 * either storage.
 *
 * Compact applies to a scalar (or no) coefficient on quads and hexes with
 * sdim == dim == vdim, without libCEED; other cases use MFEM's path whatever
 * the setting.
 */
class LeanVectorDiffusionIntegrator : public mfem::VectorDiffusionIntegrator
{
public:
   /// How the partial-assembly data is stored.
   enum class Storage
   {
      Compact, ///< One symmetric matrix per point shared by the components.
      Mfem     ///< MFEM's own (one dim x dim matrix per point per component).
   };

   /**
    * @brief Vector diffusion with a scalar coefficient.
    * @param q       The coefficient (borrowed).
    * @param storage Partial-assembly storage.
    */
   explicit LeanVectorDiffusionIntegrator(mfem::Coefficient& q,
                                          Storage storage = Storage::Compact)
      : mfem::VectorDiffusionIntegrator(q), storage_(storage) { }

   /// Vector diffusion with unit coefficient, compact storage.
   LeanVectorDiffusionIntegrator() = default;

   using mfem::BilinearFormIntegrator::AssemblePA;
   /**
    * @brief Partial-assembly setup: compact when eligible, else MFEM's.
    * @param fes Vector H1 space.
    */
   void AssemblePA(const mfem::FiniteElementSpace& fes) override;

   /**
    * @brief y += A x on E-vectors.
    * @param x Input E-vector.
    * @param y Output E-vector (accumulated into).
    */
   void AddMultPA(const mfem::Vector& x, mfem::Vector& y) const override;

   /**
    * @brief The operator is symmetric: forwards to AddMultPA.
    * @param x Input E-vector.
    * @param y Output E-vector (accumulated into).
    */
   void AddMultTransposePA(const mfem::Vector& x,
                           mfem::Vector& y) const override
   { AddMultPA(x, y); }

   /**
    * @brief Add the diagonal of the partially assembled operator.
    * @param diag E-vector in the PA diagonal's layout (accumulated into).
    */
   void AssembleDiagonalPA(mfem::Vector& diag) override;

   /// @return Whether the compact storage is in use (after AssemblePA).
   bool Compact() const { return compact_; }

   /// @return Numbers stored by the last AssemblePA (compact or MFEM's).
   long long PADataSize() const
   { return compact_ ? cpa_.Size() : pa_data.Size(); }

   /// Kernel signature: ne, B, G, compact data, x, y, d1d, q1d.
   using ApplyKernelType = void (*)(const int, const mfem::Array<mfem::real_t>&,
                                    const mfem::Array<mfem::real_t>&,
                                    const mfem::Vector&, const mfem::Vector&,
                                    mfem::Vector&, const int, const int);

   /// Kernel registry for the compact apply, keyed on (dim, d1d, q1d).
   MFEM_REGISTER_KERNELS(CompactApplyKernels, ApplyKernelType, (int, int, int));

   /// Register a compile-time (DIM, D1D, Q1D) specialization.
   template <int DIM, int D1D, int Q1D>
   static void AddSpecialization()
   { CompactApplyKernels::Specialization<DIM, D1D, Q1D>::Add(); }

private:
   Storage storage_ = Storage::Compact; ///< Requested storage.
   bool compact_ = false;               ///< Compact data in use.
   const mfem::DofToQuad* cmaps_ = nullptr; ///< Tensor maps (not owned).
   int cdim_ = 0; ///< Dimension (compact).
   int cne_ = 0;  ///< Elements (compact).
   int cd1d_ = 0; ///< Dofs per direction (compact).
   int cq1d_ = 0; ///< Points per direction (compact).
   /// Compact data, (nq, dim (dim + 1) / 2, ne): 3D xx, xy, xz, yy, yz, zz;
   /// 2D xx, xy, yy. Coefficient, weight and 1/det J included.
   mfem::Vector cpa_;
};

} // namespace incns
