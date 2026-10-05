#include "precond/point_block_jacobi.hpp"

#include "mfem/general/forall.hpp"

// This TU defines device kernels (mfem::forall); see src/CMakeLists.txt's
// nvcc list. A host compiler would build them as host loops over device
// pointers -- this line turns a lost entry there into a compile error.
#if defined(MFEM_USE_CUDA) && !defined(__CUDACC__)
#error "point_block_jacobi.cpp defines device kernels and must be compiled by nvcc -- see the nvcc TU list in src/CMakeLists.txt"
#endif

namespace incns
{

using namespace mfem;

namespace
{

/**
 * @brief Points MFEM's PA diagonal assembly (E-vector, then the element
 *        restriction's AbsMultTranspose, then P^T -- |P^T| on nonconforming
 *        meshes) at the rotation term's nodal skew.
 *
 * A separate form, because the rotation integrator's own AssembleDiagonalPA
 * must keep returning zeros for the real operator.
 */
class NodalSkewProxy : public BilinearFormIntegrator
{
   /// The rotation integrator whose nodal skew is assembled (not owned).
   const VectorRotationalConvectionIntegrator& rot_;

public:
   /**
    * @brief Proxy for @p r's nodal skew.
    * @param r Assembled rotation integrator; must outlive the proxy.
    */
   explicit NodalSkewProxy(const VectorRotationalConvectionIntegrator& r)
      : rot_(r) { }
   using BilinearFormIntegrator::AssemblePA;
   /// Nothing to set up: the data lives in the rotation integrator.
   void AssemblePA(const FiniteElementSpace&) override { }
   /**
    * @brief Adds the rotation term's nodal skew instead of a diagonal.
    * @param s_e Velocity E-vector (accumulated into).
    */
   void AssembleDiagonalPA(Vector& s_e) override { rot_.AddNodalSkewPA(s_e); }
};

} // namespace

// Device kernels live in free functions in a NAMED namespace: nvcc rejects
// extended lambdas in functions with internal linkage or in private members.
namespace pbj
{

// True-dof index of (component c, node a): byVDIM a*dim+c, byNODES c*n+a.
MFEM_HOST_DEVICE inline int Idx(int c, int a, int n, int dim, bool by_vdim)
{
   return by_vdim ? a * dim + c : c * n + a;
}

// d = 1 on essential true dofs (the constrained operator's identity rows).
void SetEssentialDiagonal(const Array<int>& ess, Vector& d)
{
   const int ne = ess.Size();
   if (ne == 0) { return; }
   const auto E = ess.Read();
   auto D = d.ReadWrite();
   mfem::forall(ne, [ = ] MFEM_HOST_DEVICE(int i) { D[E[i]] = 1.0; });
}

// Component c of node a essential: zero the s entries in row/column c of the
// node's block -- 3D s_k for k != c; 2D the single s. Two essential components
// of one node may write the same zero concurrently; benign.
void ZeroEssentialSkew(const Array<int>& ess, int dim, int n, bool by_vdim,
                       Vector& s)
{
   const int ne = ess.Size();
   if (ne == 0) { return; }
   const auto E = ess.Read();
   auto S = s.ReadWrite();
   mfem::forall(ne, [ = ] MFEM_HOST_DEVICE(int i)
   {
      const int k = E[i];
      const int c = by_vdim ? k % dim : k / n;
      const int a = by_vdim ? k / dim : k % n;
      if (dim == 2) { S[Idx(0, a, n, dim, by_vdim)] = 0.0; }
      else
      {
         for (int j = 0; j < 3; ++j)
         {
            if (j != c) { S[Idx(j, a, n, dim, by_vdim)] = 0.0; }
         }
      }
   });
}

// blocks(i, j, a) = diag(d_a) + [s_a]_x, column-major within each block.
void FillBlocks(int dim, int n, bool by_vdim, const Vector& d, const Vector& s,
                DenseTensor& blocks)
{
   const auto D = d.Read(), S = s.Read();
   auto Bt = Reshape(blocks.Write(), dim, dim, n);
   mfem::forall(n, [ = ] MFEM_HOST_DEVICE(int a)
   {
      if (dim == 3)
      {
         const real_t d0 = D[Idx(0, a, n, 3, by_vdim)];
         const real_t d1 = D[Idx(1, a, n, 3, by_vdim)];
         const real_t d2 = D[Idx(2, a, n, 3, by_vdim)];
         const real_t s0 = S[Idx(0, a, n, 3, by_vdim)];
         const real_t s1 = S[Idx(1, a, n, 3, by_vdim)];
         const real_t s2 = S[Idx(2, a, n, 3, by_vdim)];
         Bt(0, 0, a) = d0;  Bt(0, 1, a) = -s2; Bt(0, 2, a) = s1;
         Bt(1, 0, a) = s2;  Bt(1, 1, a) = d1;  Bt(1, 2, a) = -s0;
         Bt(2, 0, a) = -s1; Bt(2, 1, a) = s0;  Bt(2, 2, a) = d2;
      }
      else
      {
         const real_t s0 = S[Idx(0, a, n, 2, by_vdim)];
         Bt(0, 0, a) = D[Idx(0, a, n, 2, by_vdim)]; Bt(0, 1, a) = -s0;
         Bt(1, 0, a) = s0; Bt(1, 1, a) = D[Idx(1, a, n, 2, by_vdim)];
      }
   });
}

// byNODES true-dof vector <-> node-contiguous (dim, n) layout.
void GatherNodes(int dim, int n, const Vector& r, Vector& rn)
{
   const auto R = r.Read();
   auto RN = rn.Write();
   mfem::forall(dim * n, [ = ] MFEM_HOST_DEVICE(int i)
   {
      RN[i] = R[(i % dim) * n + i / dim];
   });
}

void ScatterNodes(int dim, int n, const Vector& zn, Vector& z)
{
   const auto ZN = zn.Read();
   auto Z = z.Write();
   mfem::forall(dim * n, [ = ] MFEM_HOST_DEVICE(int i)
   {
      Z[(i % dim) * n + i / dim] = ZN[i];
   });
}

} // namespace pbj

PointBlockJacobi::PointBlockJacobi(FiniteElementSpace& fes,
                                   const VectorRotationalConvectionIntegrator& rot,
                                   const Array<int>& ess_tdofs)
   : Solver(fes.GetTrueVSize()), dim_(fes.GetMesh()->Dimension()), n_(0),
     by_vdim_(fes.GetOrdering() == Ordering::byVDIM), ess_(ess_tdofs)
{
   MFEM_VERIFY(dim_ == 2 || dim_ == 3, "PointBlockJacobi: dim must be 2 or 3");
   MFEM_VERIFY(fes.GetVDim() == dim_, "PointBlockJacobi: vdim must equal dim");
   MFEM_VERIFY(rot.GetAssembledMesh() == fes.GetMesh(), "PointBlockJacobi: "
               "rot must be AssemblePA()'d on the velocity space's mesh");
   n_ = fes.GetTrueVSize() / dim_;

   if (auto* pfes = dynamic_cast<ParFiniteElementSpace*>(&fes))
   {
      skew_form_ = std::make_unique<ParBilinearForm>(pfes);
   }
   else
   {
      skew_form_ = std::make_unique<BilinearForm>(&fes);
   }
   skew_form_->SetAssemblyLevel(AssemblyLevel::PARTIAL);
   skew_form_->AddDomainIntegrator(new NodalSkewProxy(rot));
   skew_form_->Assemble();

   d_.SetSize(dim_ * n_);
   d_.UseDevice(true);
   d_ = 1.0;
   s_.SetSize(dim_ * n_);
   s_.UseDevice(true);
   s_ = 0.0;
   if (!by_vdim_)
   {
      r_node_.SetSize(dim_ * n_);
      r_node_.UseDevice(true);
      z_node_.SetSize(dim_ * n_);
      z_node_.UseDevice(true);
   }
   RebuildBlocks();
}

PointBlockJacobi::~PointBlockJacobi() = default;

void PointBlockJacobi::SetDiagonal(const Vector& d)
{
   MFEM_VERIFY(d.Size() == d_.Size(), "PointBlockJacobi::SetDiagonal: "
               "expected a true-dof vector of size " << d_.Size());
   d_ = d;
   pbj::SetEssentialDiagonal(ess_, d_);
   RebuildBlocks();
}

void PointBlockJacobi::UpdateSkew()
{
   skew_form_->AssembleDiagonal(s_);
   pbj::ZeroEssentialSkew(ess_, dim_, n_, by_vdim_, s_);
   RebuildBlocks();
}

void PointBlockJacobi::RebuildBlocks()
{
   // INVERSE mode: each apply is then one batched small mat-vec, which
   // parallelizes better than triangular solves. The blocks are always
   // invertible: positive diagonal gives det > 0 for any vorticity, and
   // essential nodes get identity rows/columns. BatchedDirectSolver has no
   // SetOperator, so it is rebuilt -- once per step.
   if (n_ == 0) { blocks_inv_.reset(); return; }
   DenseTensor blocks(dim_, dim_, n_);
   pbj::FillBlocks(dim_, n_, by_vdim_, d_, s_, blocks);
   blocks_inv_ = std::make_unique<BatchedDirectSolver>(
                    blocks, BatchedDirectSolver::INVERSE);
}

void PointBlockJacobi::Mult(const Vector& r, Vector& z) const
{
   if (!blocks_inv_) { return; } // no local dofs
   if (by_vdim_) { blocks_inv_->Mult(r, z); return; }
   pbj::GatherNodes(dim_, n_, r, r_node_);
   blocks_inv_->Mult(r_node_, z_node_);
   pbj::ScatterNodes(dim_, n_, z_node_, z);
}

} // namespace incns
