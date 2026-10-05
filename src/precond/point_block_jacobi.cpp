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

// inv(i, j, a) = B_a^{-1}, B_a = diag(d_a) + [s_a]_x, column-major per node,
// in closed form (adjugate / determinant), written in place: no allocation,
// no batched-library call. det B_a = d0 d1 d2 + d0 s0^2 + d1 s1^2 + d2 s2^2
// (2D: d0 d1 + s^2) > 0 for a positive diagonal, for any vorticity; essential
// nodes carry identity rows/columns.
void InvertBlocks(int dim, int n, bool by_vdim, const Vector& d,
                  const Vector& s, Vector& inv)
{
   const auto D = d.Read(), S = s.Read();
   auto I = Reshape(inv.Write(), dim, dim, n);
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
         // B = [[d0, -s2, s1], [s2, d1, -s0], [-s1, s0, d2]].
         const real_t b00 = d0, b01 = -s2, b02 = s1;
         const real_t b10 = s2, b11 = d1, b12 = -s0;
         const real_t b20 = -s1, b21 = s0, b22 = d2;
         // adj(B) (B adj(B) = det(B) I).
         const real_t a00 = b11 * b22 - b12 * b21;
         const real_t a01 = b02 * b21 - b01 * b22;
         const real_t a02 = b01 * b12 - b02 * b11;
         const real_t a10 = b12 * b20 - b10 * b22;
         const real_t a11 = b00 * b22 - b02 * b20;
         const real_t a12 = b02 * b10 - b00 * b12;
         const real_t a20 = b10 * b21 - b11 * b20;
         const real_t a21 = b01 * b20 - b00 * b21;
         const real_t a22 = b00 * b11 - b01 * b10;
         const real_t r = 1.0 / (b00 * a00 + b01 * a10 + b02 * a20);
         I(0, 0, a) = r * a00; I(0, 1, a) = r * a01; I(0, 2, a) = r * a02;
         I(1, 0, a) = r * a10; I(1, 1, a) = r * a11; I(1, 2, a) = r * a12;
         I(2, 0, a) = r * a20; I(2, 1, a) = r * a21; I(2, 2, a) = r * a22;
      }
      else
      {
         // B = [[d0, -s], [s, d1]] -> B^{-1} = [[d1, s], [-s, d0]] / det.
         const real_t d0 = D[Idx(0, a, n, 2, by_vdim)];
         const real_t d1 = D[Idx(1, a, n, 2, by_vdim)];
         const real_t s0 = S[Idx(0, a, n, 2, by_vdim)];
         const real_t r = 1.0 / (d0 * d1 + s0 * s0);
         I(0, 0, a) = r * d1;  I(0, 1, a) = r * s0;
         I(1, 0, a) = -r * s0; I(1, 1, a) = r * d0;
      }
   });
}

// z_a = B_a^{-1} r_a for every node, reading and writing the true-dof layout
// directly (byNODES or byVDIM): one kernel, no gather/scatter pass. For
// byNODES each component access is contiguous across nodes (coalesced).
void ApplyBlocks(int dim, int n, bool by_vdim, const Vector& inv,
                 const Vector& r, Vector& z)
{
   const auto I = Reshape(inv.Read(), dim, dim, n);
   const auto R = r.Read();
   auto Z = z.Write();
   mfem::forall(n, [ = ] MFEM_HOST_DEVICE(int a)
   {
      real_t ra[3] = {0.0, 0.0, 0.0};
      for (int c = 0; c < dim; ++c) { ra[c] = R[Idx(c, a, n, dim, by_vdim)]; }
      for (int i = 0; i < dim; ++i)
      {
         real_t zi = 0.0;
         for (int j = 0; j < dim; ++j) { zi += I(i, j, a) * ra[j]; }
         Z[Idx(i, a, n, dim, by_vdim)] = zi;
      }
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
   inv_.SetSize(dim_ * dim_ * n_);
   inv_.UseDevice(true);
   RebuildBlocks();
}

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
   // Closed-form inverses, in place: once per step (UpdateSkew) and per
   // Delta-t change (SetDiagonal), no allocation after construction.
   pbj::InvertBlocks(dim_, n_, by_vdim_, d_, s_, inv_);
}

void PointBlockJacobi::Mult(const Vector& r, Vector& z) const
{
   pbj::ApplyBlocks(dim_, n_, by_vdim_, inv_, r, z);
}

} // namespace incns
