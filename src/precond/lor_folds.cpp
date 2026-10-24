#include "precond/lor_folds.hpp"

#include "mfem/general/forall.hpp"

// This TU defines a device kernel (mfem::forall). A host compiler would build
// it as a host loop over device pointers, which segfaults on a GPU;
// src/CMakeLists.txt marks this file LANGUAGE CUDA, and this line turns a lost
// entry there into a compile error.
#if defined(MFEM_USE_CUDA) && !defined(__CUDACC__)
#error "lor_folds.cpp defines device kernels and must be compiled by nvcc -- see the nvcc TU list in src/CMakeLists.txt"
#endif

namespace incns
{

// The kernel's helper in a NAMED namespace (as time/cfl.cpp's): nvcc rejects
// extended lambdas whose enclosing function has internal linkage.
namespace lor_fold_kernels
{

/// Index of LOR vertex (i, j, k) of element e in the batched LOR vertex
/// coordinates (byVDIM: sdim x n1^dim x ne, lexicographic, x fastest).
MFEM_HOST_DEVICE inline int LorVertex(int sdim, int n1, int nv, int e, int i,
                                      int j, int k)
{
   return sdim * (nv * e + i + n1 * (j + n1 * k));
}

} // namespace lor_fold_kernels

long long CountFoldedLorElements(mfem::ParFiniteElementSpace& fes)
{
   mfem::ParMesh& mesh = *fes.GetParMesh();
   const int dim = mesh.Dimension();
   int covered = (mfem::UsesTensorBasis(fes) && !fes.IsDGSpace() &&
                  mesh.SpaceDimension() == dim && (dim == 2 || dim == 3))
                 ? 1 : 0;
   MPI_Allreduce(MPI_IN_PLACE, &covered, 1, MPI_INT, MPI_MIN, fes.GetComm());
   if (!covered) { return 0; }
   const int p = fes.GetMaxElementOrder(), n1 = p + 1;
   const int nv = dim == 2 ? n1 * n1 : n1 * n1 * n1;
   const int nsub = dim == 2 ? p * p : p * p * p;
   const int ne = fes.GetNE(), sdim = dim;
   // The LOR vertices: the mesh's nodes at the n1-point Gauss-Lobatto
   // tensor rule (lexicographic, x fastest), as the batched assembly's
   // FormLORVertexCoordinates finds them -- but on a rule with a fixed
   // address. MFEM caches quadrature maps by the rule's address, and that
   // function's rule is on its stack: the pressure's and the velocity's
   // orders met each other's stale maps there (a 25-point map wrote into a
   // 16-point buffer).
   static mfem::IntegrationRules lobatto(0, mfem::Quadrature1D::GaussLobatto);
   const mfem::IntegrationRule& ir =
      lobatto.Get(mesh.GetTypicalElementGeometry(), 2 * n1 - 3);
   MFEM_VERIFY(ir.GetNPoints() == nv, "lor_folds: the vertex rule has "
               << ir.GetNPoints() << " points, not " << nv);
   mesh.EnsureNodes();
   const mfem::GridFunction& nodes = *mesh.GetNodes();
   const mfem::FiniteElementSpace& nfes = *nodes.FESpace();
   const mfem::Operator* restriction =
      nfes.GetElementRestriction(mfem::ElementDofOrdering::LEXICOGRAPHIC);
   mfem::Vector nodal_e(restriction->Height());
   nodal_e.UseDevice(true);
   restriction->Mult(nodes, nodal_e);
   mfem::Vector x_vert(sdim * nv * ne);
   x_vert.UseDevice(true);
   const mfem::QuadratureInterpolator* qi = nfes.GetQuadratureInterpolator(ir);
   qi->SetOutputLayout(mfem::QVectorLayout::byVDIM);
   qi->Values(nodal_e, x_vert);
   mfem::Vector folded(ne * nsub);
   folded.UseDevice(true);
   const double* X = x_vert.Read();
   double* F = folded.Write();
   if (dim == 2)
   {
      mfem::forall(ne * nsub, [ = ] MFEM_HOST_DEVICE(int t)
      {
         const int e = t / nsub, s = t % nsub, ix = s % p, iy = s / p;
         double fold = 0.0;
         for (int b = 0; b < 2; ++b)
         {
            for (int a = 0; a < 2; ++a)
            {
               // The edges through vertex (a, b) along xi and eta.
               const int x0 = lor_fold_kernels::LorVertex(sdim, n1, nv, e, ix, iy + b, 0);
               const int x1 = lor_fold_kernels::LorVertex(sdim, n1, nv, e, ix + 1, iy + b, 0);
               const int y0 = lor_fold_kernels::LorVertex(sdim, n1, nv, e, ix + a, iy, 0);
               const int y1 = lor_fold_kernels::LorVertex(sdim, n1, nv, e, ix + a, iy + 1, 0);
               const double det = (X[x1] - X[x0]) * (X[y1 + 1] - X[y0 + 1]) -
                                  (X[x1 + 1] - X[x0 + 1]) * (X[y1] - X[y0]);
               if (!(det > 0.0)) { fold = 1.0; }
            }
         }
         F[t] = fold;
      });
   }
   else
   {
      mfem::forall(ne * nsub, [ = ] MFEM_HOST_DEVICE(int t)
      {
         const int e = t / nsub, s = t % nsub;
         const int ix = s % p, iy = (s / p) % p, iz = s / (p * p);
         double fold = 0.0;
         for (int c = 0; c < 2; ++c)
         {
            for (int b = 0; b < 2; ++b)
            {
               for (int a = 0; a < 2; ++a)
               {
                  // The edges through vertex (a, b, c) along xi, eta, zeta.
                  double u[3], v[3], w[3];
                  const int x0 = lor_fold_kernels::LorVertex(sdim, n1, nv, e, ix, iy + b, iz + c);
                  const int x1 = lor_fold_kernels::LorVertex(sdim, n1, nv, e, ix + 1, iy + b, iz + c);
                  const int y0 = lor_fold_kernels::LorVertex(sdim, n1, nv, e, ix + a, iy, iz + c);
                  const int y1 = lor_fold_kernels::LorVertex(sdim, n1, nv, e, ix + a, iy + 1, iz + c);
                  const int z0 = lor_fold_kernels::LorVertex(sdim, n1, nv, e, ix + a, iy + b, iz);
                  const int z1 = lor_fold_kernels::LorVertex(sdim, n1, nv, e, ix + a, iy + b, iz + 1);
                  for (int d = 0; d < 3; ++d)
                  {
                     u[d] = X[x1 + d] - X[x0 + d];
                     v[d] = X[y1 + d] - X[y0 + d];
                     w[d] = X[z1 + d] - X[z0 + d];
                  }
                  const double det = u[0] * (v[1] * w[2] - v[2] * w[1]) -
                                     u[1] * (v[0] * w[2] - v[2] * w[0]) +
                                     u[2] * (v[0] * w[1] - v[1] * w[0]);
                  if (!(det > 0.0)) { fold = 1.0; }
               }
            }
         }
         F[t] = fold;
      });
   }
   long long n = static_cast<long long>(folded.Sum() + 0.5);
   MPI_Allreduce(MPI_IN_PLACE, &n, 1, MPI_LONG_LONG, MPI_SUM, fes.GetComm());
   return n;
}

} // namespace incns
